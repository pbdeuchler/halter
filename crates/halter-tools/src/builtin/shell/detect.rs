// pattern: Functional Core
//
// Shell detection shared by the `shell`, `background`, and `pty` tools and by
// command hooks in the runtime. Resolution, argv
// construction, and the model-facing description are pure; only `detected`
// reads the process environment and runs the one-time version probe.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::LazyLock;

use base64::Engine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Shell families the tools can drive. Each differs in syntax the model must know about.
pub enum ShellKind {
    Bash,
    Zsh,
    /// A plain POSIX sh: dash, ash, ksh, or `/bin/sh` itself.
    Posix,
    PowerShell,
}

impl ShellKind {
    /// Classify a shell binary by its file stem. Non-POSIX interactive shells
    /// such as fish or nu are not candidates: models write POSIX-family syntax.
    #[must_use]
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.file_stem()?.to_str()?.to_ascii_lowercase().as_str() {
            "bash" => Some(Self::Bash),
            "zsh" => Some(Self::Zsh),
            "sh" | "dash" | "ash" | "ksh" | "mksh" | "yash" => Some(Self::Posix),
            "pwsh" | "powershell" => Some(Self::PowerShell),
            _ => None,
        }
    }

    /// Arguments that run `command` once, non-interactively.
    #[must_use]
    pub fn argv(self, command: &str) -> Vec<String> {
        match self {
            Self::Bash | Self::Zsh | Self::Posix => vec!["-c".to_owned(), command.to_owned()],
            // -EncodedCommand (base64 of UTF-16LE) sidesteps Windows
            // command-line quoting, which PowerShell parses inconsistently.
            Self::PowerShell => {
                let utf16: Vec<u8> = command.encode_utf16().flat_map(u16::to_le_bytes).collect();
                [
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-EncodedCommand",
                ]
                .into_iter()
                .map(str::to_owned)
                .chain([base64::engine::general_purpose::STANDARD.encode(utf16)])
                .collect()
            }
        }
    }

    const fn version_probe(self) -> Option<&'static str> {
        match self {
            Self::Bash => Some("printf %s \"$BASH_VERSION\""),
            Self::Zsh => Some("printf %s \"$ZSH_VERSION\""),
            Self::Posix => None,
            Self::PowerShell => Some("$PSVersionTable.PSVersion.ToString()"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// The shell commands run under, as detected for this process.
pub struct Shell {
    pub kind: ShellKind,
    pub path: PathBuf,
    pub version: Option<String>,
}

/// Names searched on `PATH`, in preference order, when `$SHELL` is unusable.
#[cfg(not(windows))]
const FALLBACKS: &[&str] = &["bash", "zsh", "sh"];
#[cfg(windows)]
const FALLBACKS: &[&str] = &["bash", "pwsh", "powershell"];

/// Used when nothing else resolves; present on every supported system.
#[cfg(not(windows))]
const LAST_RESORT: (ShellKind, &str) = (ShellKind::Posix, "/bin/sh");
#[cfg(windows)]
const LAST_RESORT: (ShellKind, &str) = (ShellKind::PowerShell, "powershell.exe");

/// Pick the shell: `$SHELL` when it is an existing, supported shell, else the
/// first fallback `find` locates, else the platform's last resort.
pub fn resolve(
    env_shell: Option<&OsStr>,
    find: impl Fn(&str) -> Option<PathBuf>,
) -> (ShellKind, PathBuf) {
    env_shell
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .into_iter()
        .chain(FALLBACKS.iter().filter_map(|name| find(name)))
        .find_map(|path| ShellKind::from_path(&path).map(|kind| (kind, path)))
        .unwrap_or_else(|| (LAST_RESORT.0, PathBuf::from(LAST_RESORT.1)))
}

/// First executable named `name` in a `PATH`-style list. On Windows this skips
/// `System32\bash.exe`, which launches WSL rather than a local Bash.
fn find_on_path(path_var: &OsStr, name: &str) -> Option<PathBuf> {
    std::env::split_paths(path_var)
        .map(|dir| {
            dir.join(name)
                .with_extension(std::env::consts::EXE_EXTENSION)
        })
        .find(|candidate| {
            candidate.is_file()
                && !(cfg!(windows)
                    && candidate
                        .parent()
                        .and_then(Path::file_name)
                        .is_some_and(|dir| dir.eq_ignore_ascii_case("System32")))
        })
}

/// The shell for this process, resolved and version-probed on first use.
pub fn detected() -> &'static Shell {
    static SHELL: LazyLock<Shell> = LazyLock::new(|| {
        let path_var = std::env::var_os("PATH").unwrap_or_default();
        let (kind, path) = resolve(std::env::var_os("SHELL").as_deref(), |name| {
            find_on_path(&path_var, name)
        });
        let version = probe_version(kind, &path);
        Shell {
            kind,
            path,
            version,
        }
    });
    &SHELL
}

fn probe_version(kind: ShellKind, path: &Path) -> Option<String> {
    let output = std::process::Command::new(path)
        .args(kind.argv(kind.version_probe()?))
        .env_clear()
        .envs(super::inherited_env())
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    let version = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (output.status.success() && !version.is_empty()).then_some(version)
}

const USAGE: &str = "Each call starts a fresh non-interactive shell, so `cd`, exported \
    variables, aliases and functions do not carry over: chain dependent steps in one command \
    (`cd sub && make`) or pass `cwd`. Commands start in the session working directory. stdin \
    is empty, so pass non-interactive flags (`--yes`, `git --no-pager`) instead of running \
    commands that prompt. Only essential variables (PATH, HOME, locale) are inherited; pass \
    others in `env`. Processes still running when the command exits, including ones started \
    with `&`, are killed; use the `background` tool for servers and watchers. Output past \
    1 MiB is truncated. The default timeout comes from configuration; raise `timeout_ms` for \
    long builds.";

impl Shell {
    fn name(&self) -> String {
        self.path.file_stem().map_or_else(
            || self.path.display().to_string(),
            |stem| stem.to_string_lossy().into_owned(),
        )
    }

    fn major_version(&self) -> Option<u32> {
        self.version.as_deref()?.split('.').next()?.parse().ok()
    }

    /// Syntax pitfalls of this shell that a model writing bash would hit.
    fn kind_note(&self) -> Option<String> {
        match self.kind {
            ShellKind::Bash => self.major_version().filter(|major| *major < 4).map(|_| {
                format!(
                    "Bash {} predates 4.0: there are no associative arrays (`declare -A`), \
                     `mapfile`/`readarray`, `${{var,,}}` case conversion, `globstar` or `&>>`.",
                    self.version.as_deref().unwrap_or_default()
                )
            }),
            ShellKind::Zsh => Some(
                "This is zsh, not bash: a glob that matches nothing is an error (`no matches \
                 found`), so quote patterns and URLs containing `*`, `?` or `[`; unquoted \
                 `$var` is not word-split; arrays are 1-indexed; options use `setopt`, not \
                 `shopt`."
                    .to_owned(),
            ),
            ShellKind::Posix => Some(format!(
                "This is {}, a POSIX sh without bash extensions: no `[[ ]]`, arrays, `source` \
                 (use `.`), `{{a,b}}` brace expansion, `function` keyword or `$'...'` quoting.",
                self.name()
            )),
            ShellKind::PowerShell => Some(
                "This is PowerShell, not a POSIX shell: use PowerShell syntax and cmdlets, set \
                 variables with `$env:NAME = 'value'`, and note `&&`/`||` need PowerShell 7+."
                    .to_owned(),
            ),
        }
    }

    /// Platform userland differences that affect common commands.
    fn os_note(&self, os: &str) -> Option<&'static str> {
        match (os, self.kind) {
            ("macos", _) => Some(
                "macOS userland is BSD, not GNU: `sed -i` needs an explicit suffix \
                 (`sed -i ''`), `grep -P` is unavailable, and `date`, `stat`, `find` and \
                 `xargs` take BSD flags.",
            ),
            ("windows", ShellKind::Bash | ShellKind::Zsh | ShellKind::Posix) => Some(
                "This is a POSIX shell on Windows (Git Bash/MSYS): use forward slashes; \
                 `C:\\path` is `/c/path`.",
            ),
            _ => None,
        }
    }

    /// The `shell` tool description: which shell runs, how calls behave, and
    /// the shell- and platform-specific pitfalls worth knowing up front.
    #[must_use]
    pub fn description(&self, os: &str) -> String {
        let label = match &self.version {
            Some(version) => format!("{} {version}", self.name()),
            None => self.name(),
        };
        let header = format!(
            "Run a command with {label} (`{}`) on {os}. {USAGE}",
            self.path.display()
        );
        [
            Some(header),
            self.kind_note(),
            self.os_note(os).map(str::to_owned),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(kind: ShellKind, path: &str, version: Option<&str>) -> Shell {
        Shell {
            kind,
            path: PathBuf::from(path),
            version: version.map(str::to_owned),
        }
    }

    #[test]
    fn classifies_shells_by_file_stem() {
        for (path, expected) in [
            ("/bin/bash", Some(ShellKind::Bash)),
            ("/usr/local/bin/zsh", Some(ShellKind::Zsh)),
            ("/bin/dash", Some(ShellKind::Posix)),
            ("/bin/sh", Some(ShellKind::Posix)),
            ("pwsh.exe", Some(ShellKind::PowerShell)),
            ("powershell.exe", Some(ShellKind::PowerShell)),
            ("BASH.EXE", Some(ShellKind::Bash)),
            ("/usr/bin/fish", None),
            ("/usr/bin/nu", None),
            ("", None),
        ] {
            assert_eq!(ShellKind::from_path(Path::new(path)), expected, "{path}");
        }
    }

    #[test]
    fn argv_passes_posix_commands_verbatim_and_encodes_powershell() {
        assert_eq!(ShellKind::Zsh.argv("echo 'a b'"), ["-c", "echo 'a b'"]);
        // Never a login shell: `-l` sources rc files (~/.bash_profile,
        // /etc/profile) that run user-controlled startup code.
        for kind in [ShellKind::Bash, ShellKind::Zsh, ShellKind::Posix] {
            assert!(!kind.argv("x").iter().any(|arg| arg.starts_with("-l")));
        }
        let argv = ShellKind::PowerShell.argv("echo \"é\"");
        assert_eq!(
            argv[..4],
            [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand"
            ]
        );
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&argv[4])
            .expect("valid base64");
        let units: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();
        assert_eq!(String::from_utf16(&units).unwrap(), "echo \"é\"");
    }

    #[test]
    fn resolve_prefers_supported_env_shell_then_fallbacks_then_last_resort() {
        let dir = tempfile::tempdir().unwrap();
        let file = |name: &str| {
            let path = dir.path().join(name);
            std::fs::write(&path, "").unwrap();
            path
        };
        let zsh = file("zsh");
        let fish = file("fish");
        let fallback = file(FALLBACKS[0]);
        let find_fallback = |name: &str| (name == FALLBACKS[0]).then(|| fallback.clone());
        let find_none = |_: &str| None;
        let fallback_kind = ShellKind::from_path(&fallback).unwrap();

        assert_eq!(
            resolve(Some(zsh.as_os_str()), find_none),
            (ShellKind::Zsh, zsh.clone()),
            "a supported $SHELL wins"
        );
        assert_eq!(
            resolve(Some(fish.as_os_str()), find_fallback),
            (fallback_kind, fallback.clone()),
            "an unsupported $SHELL falls through"
        );
        assert_eq!(
            resolve(
                Some(dir.path().join("missing/zsh").as_os_str()),
                find_fallback
            ),
            (fallback_kind, fallback.clone()),
            "a nonexistent $SHELL falls through"
        );
        assert_eq!(resolve(None, find_fallback), (fallback_kind, fallback));
        assert_eq!(
            resolve(None, find_none),
            (LAST_RESORT.0, PathBuf::from(LAST_RESORT.1))
        );
    }

    #[test]
    fn find_on_path_returns_first_existing_match_or_none() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        let name = Path::new("tool").with_extension(std::env::consts::EXE_EXTENSION);
        std::fs::write(second.path().join(&name), "").unwrap();
        let path_var = std::env::join_paths([first.path(), second.path()]).unwrap();
        assert_eq!(
            find_on_path(&path_var, "tool"),
            Some(second.path().join(&name))
        );
        assert_eq!(find_on_path(&path_var, "absent"), None);
        assert_eq!(find_on_path(OsStr::new(""), "tool"), None);
    }

    #[test]
    fn description_names_the_shell_and_its_pitfalls() {
        let cases = [
            (
                shell(ShellKind::Zsh, "/bin/zsh", Some("5.9")),
                "macos",
                vec![
                    "with zsh 5.9 (`/bin/zsh`) on macos",
                    "no matches found",
                    "BSD",
                ],
                vec!["predates"],
            ),
            (
                shell(ShellKind::Bash, "/bin/bash", Some("3.2.57(1)-release")),
                "linux",
                vec!["Bash 3.2.57(1)-release predates 4.0", "declare -A"],
                vec!["BSD", "zsh"],
            ),
            (
                shell(ShellKind::Bash, "/usr/bin/bash", Some("5.2.21(1)-release")),
                "linux",
                vec!["with bash 5.2.21(1)-release"],
                vec!["predates", "BSD"],
            ),
            (
                shell(ShellKind::Bash, "/usr/bin/bash", None),
                "linux",
                vec!["with bash (`/usr/bin/bash`)"],
                vec!["predates"],
            ),
            (
                shell(ShellKind::Posix, "/bin/dash", None),
                "linux",
                vec!["This is dash, a POSIX sh", "`[[ ]]`"],
                vec![],
            ),
            (
                shell(ShellKind::PowerShell, "pwsh.exe", Some("7.4.1")),
                "windows",
                vec!["with pwsh 7.4.1", "not a POSIX shell"],
                vec!["Git Bash"],
            ),
            (
                shell(ShellKind::Bash, "C:\\Git\\bin\\bash.exe", Some("5.2.0")),
                "windows",
                vec!["Git Bash/MSYS"],
                vec![],
            ),
        ];
        for (shell, os, present, absent) in cases {
            let description = shell.description(os);
            assert!(description.contains("fresh non-interactive shell"));
            for needle in present {
                assert!(description.contains(needle), "{needle:?} in {description}");
            }
            for needle in absent {
                assert!(!description.contains(needle), "{needle:?} in {description}");
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn detected_shell_reports_a_version_exactly_when_it_has_a_probe() {
        let shell = detected();
        assert_eq!(
            shell.version.is_some(),
            shell.kind.version_probe().is_some(),
            "{shell:?}"
        );
    }
}
