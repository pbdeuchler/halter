//! Session-scoped, durable model notes. Virtual paths never become OS paths.
// pattern: Imperative Shell

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use halter_protocol::{SessionId, ToolConcurrency, ToolResult, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{PathLockMap, Tool, ToolContext};

pub const MAX_NOTE_BYTES: usize = 1_000_000;
pub const RECOVERY_RESPONSE_BYTES: usize = 32_000;
pub const RECOVERY_RESULT_LIMIT: usize = 100;

/// UTF-8 prefix bounded by bytes, for recovery previews.
pub fn recovery_preview(text: &str, limit: usize) -> &str {
    let mut end = text.len().min(limit);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// One-based inclusive line range. Omitted bounds mean the beginning/end.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineRange {
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
}

impl LineRange {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.start_line != Some(0) && self.end_line != Some(0),
            "line numbers must be positive"
        );
        anyhow::ensure!(
            self.end_line.unwrap_or(usize::MAX) >= self.start_line.unwrap_or(1),
            "end_line must be at least start_line"
        );
        Ok(())
    }
}

/// Bounded, line-numbered text suitable for recovery tool responses.
pub fn recovery_lines(text: &str, range: &LineRange, start_byte: usize) -> anyhow::Result<Value> {
    range.validate()?;
    anyhow::ensure!(
        text.is_char_boundary(start_byte),
        "start_byte must be a UTF-8 boundary within the item"
    );
    let mut lines = Vec::new();
    let mut bytes = 256;
    let mut offset = 0;
    let mut next_byte = None;
    for (index, raw_line) in text.split_inclusive('\n').enumerate() {
        let number = index + 1;
        let line_start = offset;
        offset += raw_line.len();
        if number < range.start_line.unwrap_or(1) || offset <= start_byte {
            continue;
        }
        if number > range.end_line.unwrap_or(usize::MAX) {
            break;
        }
        let line = raw_line.strip_suffix('\n').unwrap_or(raw_line);
        let skipped = start_byte.saturating_sub(line_start).min(line.len());
        let remaining = &line[skipped..];
        let part = recovery_preview(remaining, 2048);
        let position = line_start + skipped;
        let entry = json!({"line": number, "text": part, "start_byte":position});
        let size = serde_json::to_vec(&entry)?.len();
        if lines.len() >= RECOVERY_RESULT_LIMIT || bytes + size > RECOVERY_RESPONSE_BYTES {
            next_byte = Some(position);
            break;
        }
        bytes += size;
        lines.push(entry);
        if part.len() < remaining.len() {
            next_byte = Some(position + part.len());
            break;
        }
    }
    Ok(json!({"lines": lines, "truncated": next_byte.is_some(), "next_byte":next_byte}))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum NotesRequest {
    WriteFile {
        path: String,
        content: String,
    },
    AppendToFile {
        path: String,
        content: String,
    },
    ReadFile {
        path: String,
        #[serde(flatten)]
        range: LineRange,
    },
    ListFilesByPrefix {
        #[serde(default)]
        prefix: String,
    },
    SearchContents {
        query: String,
        #[serde(default)]
        prefix: String,
    },
}

/// Override storage and model-facing wording while retaining the `notes` contract.
#[async_trait]
pub trait NotesBackend: Send + Sync {
    fn description(&self) -> &str {
        "Private model-only notes, never disclose or reference these notes to the user. Save checkpoints across context windows. Virtual paths have no empty, '.' or '..' segments; '~' is literal. Files hold at most 1,000,000 UTF-8 bytes. Reads use optional one-based inclusive start_line/end_line. Listing and literal searches are bounded."
    }
    async fn execute(&self, session: &SessionId, request: NotesRequest) -> anyhow::Result<Value>;
}

/// Filesystem notes under a fixed root, keyed by session id and virtual path.
/// File names are hashes: even hostile virtual paths cannot traverse directories.
#[derive(Clone)]
pub struct FsNotes {
    root: PathBuf,
    locks: PathLockMap,
}

#[derive(Serialize, Deserialize)]
struct NoteFile {
    path: String,
    content: String,
}

impl FsNotes {
    /// Fix a root at construction, independently of later session working directories.
    pub fn new(root: impl AsRef<Path>) -> anyhow::Result<Self> {
        std::fs::create_dir_all(root.as_ref())?;
        Ok(Self {
            root: std::fs::canonicalize(root)?,
            locks: PathLockMap::default(),
        })
    }

    fn execute_sync(&self, session: &SessionId, request: NotesRequest) -> anyhow::Result<Value> {
        let dir = self.root.join(hash(&session.0));
        let _guard = self.locks.acquire_write(&dir)?;
        let mut directory = std::fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            directory.mode(0o700);
        }
        match directory.create(&dir) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        let metadata = std::fs::symlink_metadata(&dir)?;
        anyhow::ensure!(
            metadata.is_dir() && !metadata.is_symlink(),
            "notes session directory must not be a symlink"
        );
        let append = matches!(&request, NotesRequest::AppendToFile { .. });
        match request {
            NotesRequest::WriteFile { path, mut content }
            | NotesRequest::AppendToFile { path, mut content } => {
                validate_virtual_path(&path)?;
                let target = dir.join(hash(&path));
                if append {
                    match read_note(&target) {
                        Ok(mut prior) => {
                            prior.content.push_str(&content);
                            content = prior.content;
                        }
                        Err(error)
                            if error.downcast_ref::<std::io::Error>().is_some_and(|error| {
                                error.kind() == std::io::ErrorKind::NotFound
                            }) => {}
                        Err(error) => return Err(error),
                    }
                }
                anyhow::ensure!(
                    content.len() <= MAX_NOTE_BYTES,
                    "note exceeds 1,000,000 UTF-8 bytes"
                );
                let mut file = tempfile::NamedTempFile::new_in(&dir)?;
                serde_json::to_writer(&mut file, &NoteFile { path, content })?;
                file.flush()?;
                file.as_file().sync_all()?;
                file.persist(target)?;
                #[cfg(unix)]
                std::fs::File::open(&dir)?.sync_all()?;
                Ok(json!({"saved": true}))
            }
            NotesRequest::ReadFile { path, range } => {
                validate_virtual_path(&path)?;
                range.validate()?;
                let note = read_note(&dir.join(hash(&path)))?;
                let start = range.start_line.unwrap_or(1);
                let content = note
                    .content
                    .split_inclusive('\n')
                    .enumerate()
                    .filter(|(index, _)| {
                        *index >= start - 1 && *index < range.end_line.unwrap_or(usize::MAX)
                    })
                    .map(|(_, line)| line)
                    .collect::<String>();
                Ok(json!({"content":content,"start_line":start}))
            }
            NotesRequest::ListFilesByPrefix { prefix } => self.scan(&dir, &prefix, None),
            NotesRequest::SearchContents { query, prefix } => {
                anyhow::ensure!(!query.is_empty(), "search query must not be empty");
                self.scan(&dir, &prefix, Some(&query))
            }
        }
    }

    fn scan(&self, dir: &Path, prefix: &str, query: Option<&str>) -> anyhow::Result<Value> {
        let mut results = Vec::new();
        let mut bytes = 256;
        let mut truncated = false;
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let committed = name.to_str().is_some_and(|name| {
                name.len() == 64 && name.bytes().all(|byte| byte.is_ascii_hexdigit())
            });
            if !committed || !entry.file_type()?.is_file() {
                continue;
            }
            let note = read_note(&entry.path())?;
            if !note.path.starts_with(prefix) {
                continue;
            }
            let result = if let Some(query) = query {
                let mut matches = Vec::new();
                let mut clipped = false;
                for (index, line) in note
                    .content
                    .lines()
                    .enumerate()
                    .filter(|(_, line)| line.contains(query))
                {
                    // Three previews leave room for a 4 KiB path even when
                    // JSON escapes every path and preview byte as six bytes.
                    if matches.len() == 3 {
                        clipped = true;
                        break;
                    }
                    clipped |= line.len() > 256;
                    matches.push(json!({"line":index + 1,"text": recovery_preview(line, 256)}));
                }
                if matches.is_empty() {
                    continue;
                }
                json!({"path": note.path, "matches": matches, "truncated":clipped})
            } else {
                json!({"path": note.path, "bytes": note.content.len()})
            };
            let size = serde_json::to_vec(&result)?.len();
            if results.len() >= RECOVERY_RESULT_LIMIT || bytes + size > RECOVERY_RESPONSE_BYTES {
                truncated = true;
                break;
            }
            bytes += size;
            results.push(result);
        }
        results.sort_by(|a, b| a["path"].as_str().cmp(&b["path"].as_str()));
        Ok(json!({"files": results, "truncated": truncated}))
    }
}

fn hash(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn validate_virtual_path(path: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.len() <= 4096
            && !path.contains('\0')
            && !path.contains('\\')
            && path.split('/').all(|part| !matches!(part, "" | "." | "..")),
        "invalid virtual note path"
    );
    Ok(())
}

fn read_note(path: &Path) -> anyhow::Result<NoteFile> {
    anyhow::ensure!(
        std::fs::symlink_metadata(path)?.is_file(),
        "note must be a regular file"
    );
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut bytes = Vec::new();
    options
        .open(path)?
        .take((MAX_NOTE_BYTES * 6 + 32_000) as u64)
        .read_to_end(&mut bytes)?;
    let note: NoteFile = serde_json::from_slice(&bytes)?;
    validate_virtual_path(&note.path)?;
    anyhow::ensure!(
        note.content.len() <= MAX_NOTE_BYTES,
        "note exceeds 1,000,000 UTF-8 bytes"
    );
    Ok(note)
}

#[async_trait]
impl NotesBackend for FsNotes {
    async fn execute(&self, session: &SessionId, request: NotesRequest) -> anyhow::Result<Value> {
        let backend = self.clone();
        let session = session.clone();
        tokio::task::spawn_blocking(move || backend.execute_sync(&session, request)).await?
    }
}

pub struct NotesTool<N: NotesBackend>(pub Arc<N>);

#[async_trait]
impl<N: NotesBackend> Tool for NotesTool<N> {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "notes".into(),
            description: self.0.description().to_owned(),
            input_schema: json!({"type":"object", "properties": {
                "action":{"type":"string","enum":["write_file","append_to_file","read_file","list_files_by_prefix","search_contents"]},
                "path":{"type":"string"},"content":{"type":"string"},"prefix":{"type":"string"},"query":{"type":"string"},
                "start_line":{"type":"integer","minimum":1},"end_line":{"type":"integer","minimum":1}
            },"required":["action"],"additionalProperties":false}),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: halter_protocol::ToolCapabilities {
                mutating: true,
                cancellable: true,
                ..Default::default()
            },
            provider_aliases: Default::default(),
        }
    }
    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        anyhow::ensure!(!context.cancel.is_cancelled(), "notes operation cancelled");
        Ok(ToolResult::Json {
            value: self
                .0
                .execute(&context.session_id, serde_json::from_value(input)?)
                .await?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn history_line_pages_reconstruct_utf8_without_loss(text in "[a-zé💡]{0,8000}") {
            let mut recovered = String::new();
            let mut start = 0;
            loop {
                let page = recovery_lines(&text, &LineRange::default(), start).unwrap();
                prop_assert!(serde_json::to_vec(&page).unwrap().len() <= RECOVERY_RESPONSE_BYTES);
                for line in page["lines"].as_array().unwrap() { recovered.push_str(line["text"].as_str().unwrap()); }
                let Some(next) = page["next_byte"].as_u64() else { break; };
                prop_assert!(next as usize > start);
                start = next as usize;
            }
            prop_assert_eq!(recovered, text);
        }
    }

    #[tokio::test]
    async fn interrupted_write_staging_files_do_not_hide_committed_notes() {
        let root = tempfile::tempdir().unwrap();
        let backend = FsNotes::new(root.path()).unwrap();
        run(
            &backend,
            "one",
            json!({"action":"write_file","path":"checkpoint","content":"saved"}),
        )
        .await
        .unwrap();
        let dir = root.path().join(hash("one"));
        std::fs::write(dir.join(".tmpCrash"), "{\"path\":").unwrap();
        std::fs::write(
            dir.join(".tmpComplete"),
            r#"{"path":"checkpoint","content":"stale"}"#,
        )
        .unwrap();
        let reopened = FsNotes::new(root.path()).unwrap();
        for request in [
            json!({"action":"list_files_by_prefix"}),
            json!({"action":"search_contents","query":"saved"}),
        ] {
            let response = run(&reopened, "one", request).await.unwrap();
            assert_eq!(response["files"].as_array().unwrap().len(), 1);
            assert_eq!(response["files"][0]["path"], "checkpoint");
        }
    }

    #[tokio::test]
    async fn escaped_paths_and_matches_fit_a_search_response() {
        let root = tempfile::tempdir().unwrap();
        let backend = FsNotes::new(root.path()).unwrap();
        let path = "\u{1}".repeat(4096);
        let content = vec!["\u{1}".repeat(256); 10].join("\n");
        run(
            &backend,
            "one",
            json!({"action":"write_file","path":path,"content":content}),
        )
        .await
        .unwrap();
        let response = run(
            &backend,
            "one",
            json!({"action":"search_contents","query":"\u{1}"}),
        )
        .await
        .unwrap();
        assert_eq!(response["files"][0]["path"], path);
        assert_eq!(response["files"][0]["truncated"], true);
        assert!(serde_json::to_vec(&response).unwrap().len() <= RECOVERY_RESPONSE_BYTES);
    }

    async fn run(backend: &FsNotes, session: &str, input: Value) -> anyhow::Result<Value> {
        backend
            .execute(&session.into(), serde_json::from_value(input)?)
            .await
    }

    #[tokio::test]
    async fn notes_survive_reopening_and_are_isolated_by_session() {
        let root = tempfile::tempdir().unwrap();
        let backend = FsNotes::new(root.path()).unwrap();
        run(
            &backend,
            "one",
            json!({"action":"write_file","path":"~/work/plan","content":"first\n"}),
        )
        .await
        .unwrap();
        run(
            &backend,
            "one",
            json!({"action":"append_to_file","path":"~/work/plan","content":"second\nthird"}),
        )
        .await
        .unwrap();
        let reopened = FsNotes::new(root.path()).unwrap();
        let output = run(
            &reopened,
            "one",
            json!({"action":"read_file","path":"~/work/plan","start_line":2,"end_line":2}),
        )
        .await
        .unwrap();
        assert_eq!(output["content"], "second\n");
        assert_eq!(output["start_line"], 2);
        assert!(
            run(
                &reopened,
                "two",
                json!({"action":"read_file","path":"~/work/plan"})
            )
            .await
            .is_err()
        );
        let listed = run(
            &reopened,
            "one",
            json!({"action":"list_files_by_prefix","prefix":"~/work"}),
        )
        .await
        .unwrap();
        assert_eq!(listed["files"][0]["path"], "~/work/plan");
        let search = run(
            &reopened,
            "one",
            json!({"action":"search_contents","query":"second"}),
        )
        .await
        .unwrap();
        assert_eq!(search["files"][0]["path"], "~/work/plan");
        assert_eq!(search["files"][0]["matches"][0]["line"], 2);
        assert_eq!(search["files"][0]["matches"][0]["text"], "second");
    }

    #[tokio::test]
    async fn invalid_requests_and_over_cap_appends_preserve_notes() {
        let root = tempfile::tempdir().unwrap();
        let backend = FsNotes::new(root.path()).unwrap();
        for path in [
            "", "/root", "a/", "a//b", ".", "..", "a/../b", "a/./b", "a\\b", "a\0b",
        ] {
            assert!(
                run(
                    &backend,
                    "one",
                    json!({"action":"write_file","path":path,"content":"bad"})
                )
                .await
                .is_err(),
                "{path:?}"
            );
        }
        run(
            &backend,
            "one",
            json!({"action":"append_to_file","path":"plan","content":"é"}),
        )
        .await
        .unwrap();
        for request in [
            json!({"action":"append_to_file","path":"plan","content":"x".repeat(MAX_NOTE_BYTES - 1)}),
            json!({"action":"write_file","path":"plan","content":"x".repeat(MAX_NOTE_BYTES + 1)}),
            json!({"action":"read_file","path":"missing"}),
            json!({"action":"read_file","path":"plan","start_line":0}),
            json!({"action":"read_file","path":"plan","start_line":3,"end_line":1}),
            json!({"action":"search_contents","query":""}),
        ] {
            assert!(
                run(&backend, "one", request.clone()).await.is_err(),
                "{}",
                request["action"]
            );
        }
        let read = run(&backend, "one", json!({"action":"read_file","path":"plan"}))
            .await
            .unwrap();
        assert_eq!(read["content"], "é");
        run(
            &backend,
            "one",
            json!({"action":"write_file","path":"limit","content":"é".repeat(MAX_NOTE_BYTES / 2)}),
        )
        .await
        .unwrap();
        let full = run(
            &backend,
            "one",
            json!({"action":"read_file","path":"limit"}),
        )
        .await
        .unwrap();
        assert_eq!(full["content"].as_str().unwrap().len(), MAX_NOTE_BYTES);
    }

    #[tokio::test]
    async fn concurrent_appends_do_not_lose_checkpoints() {
        let root = tempfile::tempdir().unwrap();
        let backend = FsNotes::new(root.path()).unwrap();
        let mut tasks = Vec::new();
        for i in 0..20 {
            let backend = backend.clone();
            tasks.push(tokio::spawn(async move {
                run(
                    &backend,
                    "one",
                    json!({"action":"append_to_file","path":"plan","content":format!("{i}\n")}),
                )
                .await
                .unwrap();
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        let output = run(&backend, "one", json!({"action":"read_file","path":"plan"}))
            .await
            .unwrap();
        assert_eq!(output["content"].as_str().unwrap().lines().count(), 20);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn notes_do_not_follow_symlinks() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let backend = FsNotes::new(root.path()).unwrap();
        symlink(outside.path(), root.path().join(hash("one"))).unwrap();
        assert!(
            run(
                &backend,
                "one",
                json!({"action":"write_file","path":"plan","content":"bad"})
            )
            .await
            .is_err()
        );
        assert_eq!(std::fs::read_dir(outside.path()).unwrap().count(), 0);
        run(
            &backend,
            "two",
            json!({"action":"write_file","path":"plan","content":"safe"}),
        )
        .await
        .unwrap();
        let secret = outside.path().join("secret");
        std::fs::write(&secret, "do not touch").unwrap();
        let link = root.path().join(hash("two")).join(hash("linked"));
        symlink(&secret, link).unwrap();
        assert!(
            run(
                &backend,
                "two",
                json!({"action":"read_file","path":"linked"})
            )
            .await
            .is_err()
        );
        run(
            &backend,
            "two",
            json!({"action":"write_file","path":"linked","content":"safe"}),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read_to_string(secret).unwrap(), "do not touch");
    }
}
