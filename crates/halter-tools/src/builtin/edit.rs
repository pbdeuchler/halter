// pattern: Imperative Shell

use std::fmt::Display;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolSpec};
use serde::Serialize;
use serde_json::{Value, json};
use tracing::debug;

use crate::policy::CanonicalPath;
use crate::{ToolContext, ToolOutput, TypedTool};

use super::common::{
    ToolScope, ensure_not_cancelled, hash_text, optional_bool, optional_string, required_string,
    resolve_path,
};

/// The input shape, quoted by every error that rejects the shape itself.
const SHAPE: &str = r#"{"edits": [{"path": "...", "old_string": "...", "new_string": "...", "replace_all": false}], "expected_sha256": "..."}"#;

#[derive(Debug)]
/// Built-in tool for exact-string file edits, several per call.
pub struct EditTool;

/// One replacement, as the model asked for it.
struct Edit {
    path: PathBuf,
    old_string: String,
    new_string: String,
    replace_all: bool,
}

/// The error for edit `index` on `path`. Every error before the writes
/// says so: nothing has been written when it is raised.
fn refused(index: usize, path: &Path, reason: impl Display) -> anyhow::Error {
    anyhow::anyhow!(
        "failed to execute edit tool: edits[{index}] ('{}'): {reason}; nothing was written",
        path.display()
    )
}

/// The `edits` array of `input`, each path resolved against `working_dir`.
fn parse_edits(working_dir: &Path, input: &Value) -> anyhow::Result<Vec<Edit>> {
    let edits = match input.get("edits") {
        Some(Value::Array(edits)) if !edits.is_empty() => edits,
        Some(Value::Array(_)) => {
            anyhow::bail!("invalid tool input: `edits` is empty; send at least one edit: {SHAPE}")
        }
        _ if ["path", "old_string", "new_string"]
            .iter()
            .any(|key| input.get(key).is_some()) =>
        {
            anyhow::bail!(
                "invalid tool input: put path, old_string, new_string and replace_all in an \
                 `edits` array, one object per edit, with expected_sha256 beside it: {SHAPE}"
            )
        }
        _ => anyhow::bail!("invalid tool input: missing array field `edits`: {SHAPE}"),
    };
    edits
        .iter()
        .enumerate()
        .map(|(index, edit)| {
            let field = |key| {
                required_string(edit, key)
                    .map(str::to_owned)
                    .map_err(|error| anyhow::anyhow!("{error} in edits[{index}]: {SHAPE}"))
            };
            let path = resolve_path(working_dir, &field("path")?);
            let old_string = field("old_string")?;
            if old_string.is_empty() {
                return Err(refused(index, &path, "old_string must not be empty"));
            }
            Ok(Edit {
                new_string: field("new_string")?,
                replace_all: optional_bool(edit, "replace_all")
                    .map_err(|error| refused(index, &path, error))?
                    .unwrap_or(false),
                path,
                old_string,
            })
        })
        .collect()
}

#[async_trait]
impl TypedTool for EditTool {
    type Output = EditOutput;

    fn describe(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("edit"),
            description: "Replace exact strings in UTF-8 files. Send every change for this step \
                in one call as `edits`, a list of {path, old_string, new_string, replace_all}; \
                they can span files. Edits apply in order, so a later edit to the same file \
                matches the text the earlier ones left. `old_string` must match the file's raw \
                text exactly, including whitespace, and be unique — keep it minimal, just enough \
                context to be unique. If it is not unique the edit fails: add surrounding \
                context, or set `replace_all` to change every occurrence (handy for renames). An \
                empty `new_string` deletes the match. If any edit fails, no file is written. \
                Pass `expected_sha256` (the sha256 from `read` or a previous `edit`) to fail \
                fast if the file changed underneath you; it is only allowed when every edit \
                targets that one file. Omit it otherwise."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": { "type": "string" },
                                "old_string": { "type": "string" },
                                "new_string": { "type": "string" },
                                "replace_all": { "type": "boolean" }
                            },
                            "required": ["path", "old_string", "new_string"]
                        }
                    },
                    "expected_sha256": { "type": "string" }
                },
                "required": ["edits"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: true,
                requires_approval: false,
                cancellable: false,
                long_running: false,
            },
            provider_aliases: Default::default(),
        }
    }

    /// All-or-nothing up to the writes: every path passes the write policy,
    /// every file is read and every edit applied in memory before the first
    /// write. Each file then gets one atomic write, back to back. A failed
    /// write restores the files written before it; a crash between two
    /// writes leaves the earlier files edited and the later ones not.
    async fn run(&self, context: ToolContext, input: Value) -> anyhow::Result<Self::Output> {
        let _scope = ToolScope::new(&context, "edit");
        ensure_not_cancelled(&context.cancel)?;
        let edits = parse_edits(&context.working_dir, &input)?;
        // Anything but a SHA-256 hex digest is a placeholder ("", "unused"),
        // never a hash that could match, so it is treated as absent.
        let expected_sha256 = optional_string(&input, "expected_sha256")
            .filter(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .map(str::to_ascii_lowercase);

        // Each distinct file once, in order of first mention, and for each
        // edit the position of its file there.
        let mut files: Vec<CanonicalPath> = Vec::new();
        let mut file_of = Vec::with_capacity(edits.len());
        for (index, edit) in edits.iter().enumerate() {
            let canonical = context
                .policy
                .check_write_path(&edit.path)
                .await
                .map_err(|error| refused(index, &edit.path, error))?;
            file_of.push(
                match files
                    .iter()
                    .position(|file| file.path() == canonical.path())
                {
                    Some(at) => at,
                    None => {
                        files.push(canonical);
                        files.len() - 1
                    }
                },
            );
        }
        if expected_sha256.is_some() && files.len() > 1 {
            let paths = files.iter().map(|file| file.path().display().to_string());
            anyhow::bail!(
                "failed to execute edit tool: expected_sha256 names one file, but these edits \
                 touch {} ({}); omit expected_sha256, or split the call so each call edits one \
                 file; nothing was written",
                files.len(),
                paths.collect::<Vec<_>>().join(", ")
            );
        }
        debug!(
            session_id = %context.session_id,
            edits = edits.len(),
            files = files.len(),
            "editing files"
        );

        let path_locks = context.path_locks.clone();
        tokio::task::spawn_blocking(move || {
            use std::io::Read;

            // Distinct (a second lock on one file would wait forever) and sorted
            // (so two calls locking overlapping files cannot deadlock).
            let mut lock_order: Vec<&Path> = files.iter().map(CanonicalPath::path).collect();
            lock_order.sort_unstable();
            let _locks = lock_order
                .into_iter()
                .map(|path| path_locks.acquire_write(path))
                .collect::<anyhow::Result<Vec<_>>>()?;

            let originals: Vec<anyhow::Result<String>> = files
                .iter()
                .map(|file| {
                    let mut text = String::new();
                    file.open_read_blocking()?.read_to_string(&mut text)?;
                    Ok(text)
                })
                .collect();
            // Array order throughout, so the error names the first edit that
            // fails, and a file's read error is its first edit's.
            let mut texts: Vec<Option<String>> = vec![None; files.len()];
            let mut applied = Vec::with_capacity(edits.len());
            for (index, (edit, &at)) in edits.iter().zip(&file_of).enumerate() {
                let path = files[at].path();
                let text = match texts[at].take() {
                    Some(text) => text,
                    None => {
                        let original = originals[at]
                            .as_ref()
                            .map_err(|error| refused(index, path, error))?;
                        let hash_before = hash_text(original);
                        if let Some(expected_sha256) = expected_sha256.as_deref()
                            && expected_sha256 != hash_before
                        {
                            anyhow::bail!(
                                "failed to execute edit tool: '{}' changed since expected_sha256 \
                                 was taken (its sha256 is now {hash_before}); read it again, or \
                                 omit expected_sha256; nothing was written",
                                path.display()
                            );
                        }
                        original.clone()
                    }
                };
                let matches = text.match_indices(&edit.old_string).count();
                if matches == 0 {
                    return Err(refused(index, path, "old_string was not found"));
                }
                if matches > 1 && !edit.replace_all {
                    return Err(refused(
                        index,
                        path,
                        format!(
                            "old_string matched {matches} times; add surrounding context to \
                             make it unique, or set replace_all=true to replace every one"
                        ),
                    ));
                }
                let replacements = if edit.replace_all { matches } else { 1 };
                texts[at] = Some(text.replacen(&edit.old_string, &edit.new_string, replacements));
                applied.push(EditApplied {
                    path: path.to_path_buf(),
                    occurrences_in_file: matches,
                    replacements_applied: replacements,
                });
            }
            // Every file has an edit, so every read succeeded and every text is set.
            let originals = originals.into_iter().collect::<anyhow::Result<Vec<_>>>()?;
            let texts: Vec<String> = texts.into_iter().flatten().collect();

            for (written, (file, updated)) in files.iter().zip(&texts).enumerate() {
                if let Err(error) = file.atomic_write_blocking(updated.as_bytes()) {
                    let restored = files.iter().zip(&originals).take(written).try_for_each(
                        |(file, original)| file.atomic_write_blocking(original.as_bytes()),
                    );
                    anyhow::bail!(
                        "failed to execute edit tool: writing '{}' failed ({error}); {}",
                        file.path().display(),
                        match restored {
                            Ok(()) => format!(
                                "restored the {written} file(s) this call had already written, \
                                 so no file was changed"
                            ),
                            Err(error) => format!(
                                "restoring the {written} file(s) this call had already written \
                                 failed too ({error}), so they may hold the edited text"
                            ),
                        }
                    );
                }
            }

            Ok(EditOutput {
                files: files
                    .iter()
                    .zip(originals.iter().zip(&texts))
                    .map(|(file, (original, updated))| FileEdited {
                        path: file.path().to_path_buf(),
                        file_hash_before: hash_text(original),
                        file_hash_after: hash_text(updated),
                    })
                    .collect(),
                edits: applied,
            })
        })
        .await?
    }
}

#[derive(Debug, Serialize)]
/// Result of an `edit` call: each file written, in order of first mention,
/// and each edit, in array order.
pub struct EditOutput {
    files: Vec<FileEdited>,
    edits: Vec<EditApplied>,
}

#[derive(Debug, Serialize)]
struct FileEdited {
    path: PathBuf,
    file_hash_before: String,
    file_hash_after: String,
}

#[derive(Debug, Serialize)]
struct EditApplied {
    path: PathBuf,
    /// Occurrences of `old_string` in the text this edit ran against.
    occurrences_in_file: usize,
    replacements_applied: usize,
}

impl ToolOutput for EditOutput {
    /// One line per file, with each of its edits' replacement counts in
    /// order. The new hash is the `expected_sha256` for a follow-up edit.
    fn to_model_string(&self) -> String {
        let line = |file: &FileEdited| {
            let counts = self
                .edits
                .iter()
                .filter(|edit| edit.path == file.path)
                .map(|edit| edit.replacements_applied.to_string());
            format!(
                "Edited {} ({} replaced); sha256={}",
                file.path.display(),
                counts.collect::<Vec<_>>().join(" + "),
                file.file_hash_after
            )
        };
        self.files.iter().map(line).collect::<Vec<_>>().join("\n")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolPolicy};

    use super::*;
    use crate::Tool;
    use halter_protocol::ToolResult;

    fn tool_context(root: &std::path::Path) -> ToolContext {
        ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: root.to_path_buf(),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(crate::ToolSessionStore::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
                allowed_write_roots: vec![root.to_path_buf()],
                ..PolicySettings::default()
            })) as Arc<dyn ToolPolicy>,
            shell_timeout_secs: 30,
            subagent_parent: None,
        }
    }

    const A: &str = "one two two\n";
    const B: &str = "alpha\n";
    /// `.secrets` matches a default sensitive-path pattern: writing it is denied.
    const FILES: [(&str, &str); 3] = [("a.txt", A), ("b.txt", B), (".secrets", "s\n")];

    fn edit(path: &str, old_string: &str, new_string: &str) -> Value {
        json!({ "path": path, "old_string": old_string, "new_string": new_string })
    }

    /// Each case runs in a fresh directory holding `FILES`. A success lists
    /// every edit's replacement count and the files it changed; a failure
    /// lists fragments of the error, and must leave every file's bytes as
    /// they were.
    #[tokio::test]
    async fn edit_applies_every_edit_or_writes_nothing() {
        let a_hash = hash_text(A);
        // Ok: each edit's replacement count, and each changed file's text.
        // Err: fragments of the error.
        type Expected<'a> = Result<(Vec<usize>, Vec<(&'a str, &'a str)>), Vec<&'a str>>;
        let cases: Vec<(&str, Value, Expected)> = vec![
            (
                "single edit",
                json!({"edits": [edit("a.txt", "one", "1")]}),
                Ok((vec![1], vec![("a.txt", "1 two two\n")])),
            ),
            (
                "a later edit matches the text an earlier one left",
                json!({"edits": [edit("a.txt", "one", "uno"), edit("a.txt", "uno two", "dos")]}),
                Ok((vec![1, 1], vec![("a.txt", "dos two\n")])),
            ),
            (
                "several files",
                json!({"edits": [edit("a.txt", "one", "1"), edit("b.txt", "alpha", "beta")]}),
                Ok((
                    vec![1, 1],
                    vec![("a.txt", "1 two two\n"), ("b.txt", "beta\n")],
                )),
            ),
            (
                "replace_all",
                json!({"edits": [{"path": "a.txt", "old_string": "two", "new_string": "2", "replace_all": true}]}),
                Ok((vec![2], vec![("a.txt", "one 2 2\n")])),
            ),
            (
                "an empty new_string deletes",
                json!({"edits": [edit("a.txt", " two two", "")]}),
                Ok((vec![1], vec![("a.txt", "one\n")])),
            ),
            (
                "hash ok, upper case, across two spellings of one file",
                json!({
                    "edits": [edit("a.txt", "one", "1"), edit("./a.txt", "1", "I")],
                    "expected_sha256": a_hash.to_ascii_uppercase()
                }),
                Ok((vec![1, 1], vec![("a.txt", "I two two\n")])),
            ),
            (
                "a malformed hash is ignored, even across files",
                json!({
                    "edits": [edit("a.txt", "one", "1"), edit("b.txt", "alpha", "beta")],
                    "expected_sha256": "unused"
                }),
                Ok((
                    vec![1, 1],
                    vec![("a.txt", "1 two two\n"), ("b.txt", "beta\n")],
                )),
            ),
            (
                "unknown fields and placeholders",
                json!({
                    "edits": [
                        {"path": "a.txt", "old_string": "one", "new_string": "1", "replace_all": null, "zz": 1},
                        {"path": "b.txt", "old_string": "alpha", "new_string": "beta", "replace_all": ""},
                        {"path": "b.txt", "old_string": "beta", "new_string": "gamma", "replace_all": 0}
                    ],
                    "expected_sha256": "",
                    "zz_unknown": "unused"
                }),
                Ok((
                    vec![1, 1, 1],
                    vec![("a.txt", "1 two two\n"), ("b.txt", "gamma\n")],
                )),
            ),
            (
                "a failure in the middle",
                json!({"edits": [
                    edit("a.txt", "one", "1"),
                    edit("b.txt", "missing", "x"),
                    edit("a.txt", "two", "2"),
                ]}),
                Err(vec![
                    "edits[1] (",
                    "b.txt",
                    "old_string was not found",
                    "nothing was written",
                ]),
            ),
            (
                "an ambiguous match after a good edit",
                json!({"edits": [edit("b.txt", "alpha", "beta"), edit("a.txt", "two", "2")]}),
                Err(vec![
                    "edits[1] (",
                    "matched 2 times",
                    "replace_all=true",
                    "nothing was written",
                ]),
            ),
            (
                "an empty old_string",
                json!({"edits": [edit("a.txt", "", "x")]}),
                Err(vec![
                    "edits[0] (",
                    "old_string must not be empty",
                    "nothing was written",
                ]),
            ),
            (
                "an unreadable file after a good edit",
                json!({"edits": [edit("a.txt", "one", "1"), edit("missing.txt", "x", "y")]}),
                Err(vec!["edits[1] (", "missing.txt", "nothing was written"]),
            ),
            (
                "policy denies the second path",
                json!({"edits": [edit("a.txt", "one", "1"), edit(".secrets", "s", "t")]}),
                Err(vec!["edits[1] (", ".secrets", "nothing was written"]),
            ),
            (
                "a stale hash",
                json!({"edits": [edit("a.txt", "one", "1")], "expected_sha256": hash_text("before")}),
                Err(vec![
                    "changed since expected_sha256",
                    "read it again",
                    &a_hash,
                    "nothing was written",
                ]),
            ),
            (
                "a hash with edits to two files",
                json!({
                    "edits": [edit("a.txt", "one", "1"), edit("b.txt", "alpha", "beta")],
                    "expected_sha256": a_hash
                }),
                Err(vec![
                    "touch 2 (",
                    "omit expected_sha256",
                    "split the call",
                    "nothing was written",
                ]),
            ),
            (
                "the old flat shape",
                json!({"path": "a.txt", "old_string": "one", "new_string": "1"}),
                Err(vec!["in an `edits` array", SHAPE]),
            ),
            (
                "an empty edits array",
                json!({"edits": [], "path": "a.txt"}),
                Err(vec!["`edits` is empty", SHAPE]),
            ),
            (
                "no edits",
                json!({"edits": "a.txt"}),
                Err(vec!["missing array field `edits`", SHAPE]),
            ),
            (
                "an edit missing a field",
                json!({"edits": [edit("a.txt", "one", "1"), {"path": "a.txt", "old_string": "two"}]}),
                Err(vec!["missing string field 'new_string' in edits[1]", SHAPE]),
            ),
            (
                "a replace_all that is not a boolean",
                json!({"edits": [{"path": "a.txt", "old_string": "two", "new_string": "2", "replace_all": "yes"}]}),
                Err(vec![
                    "edits[0] (",
                    "must be a boolean",
                    "nothing was written",
                ]),
            ),
        ];

        for (name, input, expected) in cases {
            let temp = tempfile::tempdir().expect("tempdir");
            for (file, content) in FILES {
                std::fs::write(temp.path().join(file), content).expect("write");
            }
            let result = EditTool.execute(tool_context(temp.path()), input).await;
            let on_disk =
                |file: &str| std::fs::read_to_string(temp.path().join(file)).expect("read");
            match (result, expected) {
                (Ok(ToolResult::Rendered { value, .. }), Ok((replaced, changed))) => {
                    let counts: Vec<usize> = value["edits"]
                        .as_array()
                        .expect("edits")
                        .iter()
                        .map(|edit| edit["replacements_applied"].as_u64().expect("count") as usize)
                        .collect();
                    assert_eq!(counts, replaced, "{name}");
                    assert_eq!(
                        value["files"].as_array().expect("files").len(),
                        changed.len(),
                        "{name}"
                    );
                    for (file, original) in FILES {
                        let want = changed
                            .iter()
                            .find(|(path, _)| *path == file)
                            .map_or(original, |(_, text)| text);
                        assert_eq!(on_disk(file), want, "{name}: {file}");
                    }
                }
                (Err(error), Err(fragments)) => {
                    let error = error.to_string();
                    for fragment in fragments {
                        assert!(error.contains(fragment), "{name}: {error} lacks {fragment}");
                    }
                    for (file, original) in FILES {
                        assert_eq!(on_disk(file), original, "{name}: {file} was written");
                    }
                }
                (result, expected) => panic!("{name}: got {result:?}, expected {expected:?}"),
            }
        }
    }

    /// A write that fails after another file was written puts that file back.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_failed_write_restores_the_files_written_before_it() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().expect("tempdir");
        let locked = temp.path().join("locked");
        std::fs::create_dir(&locked).expect("mkdir");
        std::fs::write(temp.path().join("a.txt"), A).expect("write");
        std::fs::write(locked.join("b.txt"), B).expect("write");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).expect("chmod");
        if std::fs::write(locked.join("probe"), "").is_ok() {
            eprintln!("skipped: a read-only directory is writable here (running as root?)");
            return;
        }

        let error = EditTool
            .execute(
                tool_context(temp.path()),
                json!({"edits": [edit("a.txt", "one", "1"), edit("locked/b.txt", "alpha", "beta")]}),
            )
            .await
            .expect_err("the second write fails");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("chmod");

        let error = error.to_string();
        assert!(
            error.contains("writing '") && error.contains("b.txt' failed"),
            "{error}"
        );
        assert!(error.contains("restored the 1 file(s)"), "{error}");
        assert_eq!(
            std::fs::read_to_string(temp.path().join("a.txt")).expect("read"),
            A
        );
        assert_eq!(
            std::fs::read_to_string(locked.join("b.txt")).expect("read"),
            B
        );
    }

    #[test]
    fn edit_output_renders_a_line_per_file_with_each_edits_count() {
        let edit = |path: &str, occurrences_in_file, replacements_applied| EditApplied {
            path: PathBuf::from(path),
            occurrences_in_file,
            replacements_applied,
        };
        let file = |path: &str, after: &str| FileEdited {
            path: PathBuf::from(path),
            file_hash_before: "old".to_owned(),
            file_hash_after: after.to_owned(),
        };
        let output = EditOutput {
            files: vec![file("/w/a \"é\".rs", "new"), file("/w/b.rs", "newer")],
            edits: vec![
                edit("/w/a \"é\".rs", 3, 3),
                edit("/w/b.rs", 1, 1),
                edit("/w/a \"é\".rs", 2, 1),
            ],
        };

        assert_eq!(
            output.to_model_string(),
            "Edited /w/a \"é\".rs (3 + 1 replaced); sha256=new\nEdited /w/b.rs (1 replaced); sha256=newer"
        );
        assert_eq!(
            serde_json::to_value(&output).expect("serialize"),
            json!({
                "files": [
                    {"path": "/w/a \"é\".rs", "file_hash_before": "old", "file_hash_after": "new"},
                    {"path": "/w/b.rs", "file_hash_before": "old", "file_hash_after": "newer"},
                ],
                "edits": [
                    {"path": "/w/a \"é\".rs", "occurrences_in_file": 3, "replacements_applied": 3},
                    {"path": "/w/b.rs", "occurrences_in_file": 1, "replacements_applied": 1},
                    {"path": "/w/a \"é\".rs", "occurrences_in_file": 2, "replacements_applied": 1},
                ],
            })
        );
    }
}
