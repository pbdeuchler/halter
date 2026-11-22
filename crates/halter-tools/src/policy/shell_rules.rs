// pattern: Functional Core
//
// Shell command allowlist. tree-sitter-bash parses a command line, and every
// simple command it would run must match an allow pattern: each stage of a
// pipeline or list, commands in subshells, substitutions and function bodies,
// the scripts of literal `sh -c` / `eval` strings, and the commands that
// transparent wrappers such as `sudo` or `env` run. A wrapper is checked as
// well as the command it wraps, so unwrapping only ever adds checks.

use std::path::Path;

use globset::{Glob, GlobMatcher};
use tree_sitter::{Node, Parser};

use super::PolicyError;

/// One argv word: `Some` when its value is static, `None` when expansion
/// (variables, substitutions, globs) decides it at run time.
type Word = Option<String>;

/// Nesting limit for re-parsing `sh -c` and `eval` scripts.
const MAX_DEPTH: usize = 8;

/// Shells whose `-c` script is parsed and checked like the outer command.
const SHELLS: &[&str] = &["sh", "bash", "zsh", "dash", "ksh"];

/// Wrappers that run a trailing command: name, options that take a separate
/// value, and positional arguments before the command (`timeout`'s duration).
const WRAPPERS: &[(&str, &[&str], usize)] = &[
    ("builtin", &[], 0),
    ("command", &[], 0),
    ("env", &["-u", "--unset", "-C", "--chdir"], 0),
    ("exec", &["-a"], 0),
    ("nice", &["-n", "--adjustment"], 0),
    ("nohup", &[], 0),
    (
        "sudo",
        &["-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U"],
        0,
    ),
    ("time", &["-f", "--format", "-o", "--output"], 0),
    ("timeout", &["-s", "--signal", "-k", "--kill-after"], 1),
];

/// Authorize `command` against `allow`. A lone `*` pattern allows anything,
/// even a command line that cannot be parsed. Otherwise the shell must be
/// `posix` (the dialect tree-sitter-bash parses), the line must parse cleanly,
/// and every command it runs must match a pattern.
pub(crate) fn check(allow: &[String], command: &str, posix: bool) -> Result<(), PolicyError> {
    if allow.iter().any(|pattern| pattern.trim() == "*") {
        return Ok(());
    }
    let rejected = |reason, fragment: &str| PolicyError::ShellCommandRejected {
        reason,
        fragment: fragment.to_owned(),
    };
    let patterns = allow
        .iter()
        .map(|pattern| {
            Pattern::compile(pattern)
                .map_err(|_| rejected("invalid policy.shell.allow pattern", pattern))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !posix {
        return Err(rejected(
            "the detected shell is not POSIX, so only a `*` allow pattern permits commands",
            command,
        ));
    }
    let commands =
        commands(command, 0).ok_or_else(|| rejected("could not be parsed statically", command))?;
    match commands
        .iter()
        .find(|argv| !patterns.iter().any(|pattern| pattern.matches(argv)))
    {
        Some(argv) => Err(rejected(
            "no policy.shell.allow pattern matches",
            &render(argv),
        )),
        None => Ok(()),
    }
}

/// An allow pattern: whitespace-separated tokens, each a glob over one static
/// argv word. A trailing `*` token matches any remaining words, including
/// ones only known at run time; every other token needs a static word.
struct Pattern {
    words: Vec<GlobMatcher>,
    rest: bool,
}

impl Pattern {
    fn compile(pattern: &str) -> Result<Self, globset::Error> {
        let tokens: Vec<&str> = pattern.split_whitespace().collect();
        let (tokens, rest) = match tokens.split_last() {
            Some((&"*", fixed)) => (fixed, true),
            _ => (tokens.as_slice(), false),
        };
        let words = tokens
            .iter()
            .map(|token| Glob::new(token).map(|glob| glob.compile_matcher()))
            .collect::<Result<_, _>>()?;
        Ok(Self { words, rest })
    }

    fn matches(&self, argv: &[Word]) -> bool {
        let arity = if self.rest {
            argv.len() >= self.words.len()
        } else {
            argv.len() == self.words.len()
        };
        arity
            && self
                .words
                .iter()
                .zip(argv)
                .all(|(glob, word)| word.as_deref().is_some_and(|word| glob.is_match(word)))
    }
}

/// Every command `source` runs, or `None` when it does not parse cleanly or
/// nests `sh -c` / `eval` deeper than [`MAX_DEPTH`].
fn commands(source: &str, depth: usize) -> Option<Vec<Vec<Word>>> {
    if depth > MAX_DEPTH {
        return None;
    }
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_bash::LANGUAGE.into())
        .ok()?;
    let tree = parser.parse(source, None)?;
    if tree.root_node().has_error() {
        return None;
    }
    // Iterative, so deeply nested input cannot overflow the stack.
    let mut pending = vec![tree.root_node()];
    std::iter::from_fn(|| {
        let node = pending.pop()?;
        let mut cursor = node.walk();
        let children: Vec<Node> = node.named_children(&mut cursor).collect();
        pending.extend(children.into_iter().rev());
        Some(argv(node, source))
    })
    .flatten()
    .map(|argv| expand(argv, depth))
    .collect::<Option<Vec<_>>>()
    .map(|nested| nested.concat())
}

/// The argv of a command node; `None` for every other node.
fn argv(node: Node, source: &str) -> Option<Vec<Word>> {
    let text = |node: Node| unquote(&source[node.byte_range()]);
    let mut cursor = node.walk();
    match node.kind() {
        "command" => {
            let name = node.child_by_field_name("name")?;
            let args = node.children_by_field_name("argument", &mut cursor);
            Some(std::iter::once(name).chain(args).map(text).collect())
        }
        "declaration_command" | "unset_command" => Some(
            node.children(&mut cursor)
                .filter(|child| child.kind() != "comment")
                .map(text)
                .collect(),
        ),
        _ => None,
    }
}

/// `argv` followed by every command it runs on its behalf, or `None` when
/// those cannot be determined statically.
fn expand(argv: Vec<Word>, depth: usize) -> Option<Vec<Vec<Word>>> {
    let program = argv[0]
        .as_deref()
        .and_then(|name| Path::new(name).file_name()?.to_str())
        .unwrap_or_default();
    let args = &argv[1..];
    let inner = if let Some(&(_, value_flags, positionals)) =
        WRAPPERS.iter().find(|(name, ..)| *name == program)
    {
        match unwrap(program, args, value_flags, positionals) {
            wrapped if wrapped.is_empty() => Vec::new(),
            wrapped => expand(wrapped, depth)?,
        }
    } else if SHELLS.contains(&program) {
        shell_script(args)?.map_or(Some(Vec::new()), |script| commands(script, depth + 1))?
    } else if program == "eval" {
        let words = args
            .iter()
            .map(Option::as_deref)
            .collect::<Option<Vec<_>>>()?;
        commands(&words.join(" "), depth + 1)?
    } else {
        Vec::new()
    };
    Some(std::iter::once(argv).chain(inner).collect())
}

/// The command a wrapper runs: `args` past its options (and, for `env`,
/// variable assignments) and `positionals`. A run-time word among the skipped
/// ones, or `env --split-string`, hides the command, which is then reported
/// as a single unknown word that no pattern matches.
fn unwrap(program: &str, args: &[Word], value_flags: &[&str], positionals: usize) -> Vec<Word> {
    let env = program == "env";
    let splits = |word: &str| {
        word.starts_with("--split-string")
            || (word.starts_with('-') && !word.starts_with("--") && word.contains('S'))
    };
    let mut options = true;
    let mut index = 0;
    while let Some(word) = args.get(index) {
        let Some(word) = word.as_deref() else {
            return vec![None];
        };
        index += match word {
            "--" if options => {
                options = false;
                1
            }
            _ if options && env && splits(word) => return vec![None],
            _ if options && word.len() > 1 && word.starts_with('-') => {
                1 + usize::from(value_flags.contains(&word))
            }
            _ if env && word.contains('=') => 1,
            _ => break,
        };
    }
    args.get(index + positionals..)
        .map_or_else(Vec::new, <[Word]>::to_vec)
}

/// The `-c` script a shell invocation runs: `Some(None)` when it runs no
/// inline script, `None` when a run-time word hides which it is.
fn shell_script(args: &[Word]) -> Option<Option<&str>> {
    let mut command_flag = false;
    let mut words = args.iter();
    while let Some(word) = words.next() {
        match word.as_deref()? {
            "-o" | "+o" | "-O" | "+O" | "--rcfile" | "--init-file" => {
                words.next();
            }
            "--" => return Some(command_flag.then(|| words.next()?.as_deref()).flatten()),
            word if word.len() > 1 && (word.starts_with('-') || word.starts_with('+')) => {
                command_flag |= !word.starts_with("--") && word.contains('c');
            }
            word => return Some(command_flag.then_some(word)),
        }
    }
    Some(None)
}

/// The static value of a shell word with its quoting removed, or `None` when
/// it expands at run time (`$`, backquotes, globs, brace expansion, or a
/// process substitution).
fn unquote(raw: &str) -> Word {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    let mut double = false;
    while let Some(c) = chars.next() {
        match (c, double) {
            ('\'', false) => out.extend(chars.by_ref().take_while(|&c| c != '\'')),
            ('"', _) => double = !double,
            ('\\', false) => match chars.next()? {
                '\n' => {}
                c => out.push(c),
            },
            ('\\', true) => match chars.next()? {
                c @ ('$' | '`' | '"' | '\\') => out.push(c),
                '\n' => {}
                c => out.extend(['\\', c]),
            },
            ('$' | '`', _) | ('*' | '?' | '(' | ')' | '<' | '>', false) => return None,
            ('[', false) if chars.as_str().contains(']') => return None,
            ('{', false) if chars.as_str().contains('}') => return None,
            (c, _) => out.push(c),
        }
    }
    Some(out)
}

fn render(argv: &[Word]) -> String {
    argv.iter()
        .map(|word| word.as_deref().unwrap_or("<expansion>"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allow(patterns: &[&str]) -> Vec<String> {
        patterns.iter().map(|&pattern| pattern.to_owned()).collect()
    }

    fn rejection(result: Result<(), PolicyError>) -> Option<(&'static str, String)> {
        match result {
            Ok(()) => None,
            Err(PolicyError::ShellCommandRejected { reason, fragment }) => Some((reason, fragment)),
            Err(other) => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    fn commands_extracts_every_command_with_static_words() {
        let words = |words: &[&str]| -> Vec<Word> {
            words
                .iter()
                .map(|&word| (word != "?").then(|| word.to_owned()))
                .collect()
        };
        for (source, expected) in [
            ("git status", vec![words(&["git", "status"])]),
            (
                "ls -la | grep x && echo 'a b' || (cd sub; make) ; true &",
                vec![
                    words(&["ls", "-la"]),
                    words(&["grep", "x"]),
                    words(&["echo", "a b"]),
                    words(&["cd", "sub"]),
                    words(&["make"]),
                    words(&["true"]),
                ],
            ),
            (
                "echo \"$(rm -rf /)\" `id`",
                vec![
                    words(&["echo", "?", "?"]),
                    words(&["rm", "-rf", "/"]),
                    words(&["id"]),
                ],
            ),
            (
                "FOO=$(whoami) \\rm r''m \"q\\\"x\" $HOME *.rs {a,b} <(cat f) ~/x",
                vec![
                    words(&["rm", "rm", "q\"x", "?", "?", "?", "?", "~/x"]),
                    words(&["whoami"]),
                    words(&["cat", "f"]),
                ],
            ),
            (
                "f() { curl x; }; for i in $(seq 3); do f $i; done",
                vec![
                    words(&["curl", "x"]),
                    words(&["seq", "3"]),
                    words(&["f", "?"]),
                ],
            ),
            (
                "export A=1 B; unset A",
                vec![words(&["export", "A=1", "B"]), words(&["unset", "A"])],
            ),
            // `[ … ]` and `[[ … ]]` are test syntax, not programs; their substitutions still count.
            (
                "[ -f $(id) ] && test -d y",
                vec![words(&["id"]), words(&["test", "-d", "y"])],
            ),
            (
                "cat <<EOF\n$(id)\nEOF",
                vec![words(&["cat"]), words(&["id"])],
            ),
            ("", vec![]),
        ] {
            assert_eq!(commands(source, 0), Some(expected), "{source}");
        }
    }

    #[test]
    fn commands_unwraps_wrappers_and_parses_inline_scripts() {
        let names = |source| {
            commands(source, 0).map(|commands| {
                commands
                    .into_iter()
                    .map(|argv| render(&argv))
                    .collect::<Vec<_>>()
            })
        };
        for (source, expected) in [
            ("sudo -u root rm x", vec!["sudo -u root rm x", "rm x"]),
            (
                "env -i A=1 -- nice -n 5 git push",
                vec![
                    "env -i A=1 -- nice -n 5 git push",
                    "nice -n 5 git push",
                    "git push",
                ],
            ),
            (
                "timeout -s KILL 5 make",
                vec!["timeout -s KILL 5 make", "make"],
            ),
            ("/usr/bin/env A=1", vec!["/usr/bin/env A=1"]),
            ("env -S'rm -rf /'", vec!["env -Srm -rf /", "<expansion>"]),
            ("sudo $CMD", vec!["sudo <expansion>", "<expansion>"]),
            (
                "bash -lc 'git push && ls'",
                vec!["bash -lc git push && ls", "git push", "ls"],
            ),
            (
                "sh -o pipefail -c -- 'make'",
                vec!["sh -o pipefail -c -- make", "make"],
            ),
            ("bash script.sh -c x", vec!["bash script.sh -c x"]),
            ("eval 'rm' x", vec!["eval rm x", "rm x"]),
        ] {
            assert_eq!(
                names(source),
                Some(expected.iter().map(|&s| s.to_owned()).collect()),
                "{source}"
            );
        }
    }

    #[test]
    fn commands_rejects_what_it_cannot_resolve_statically() {
        let nested =
            (0..=MAX_DEPTH).fold("true".to_owned(), |script, _| format!("eval {script:?}"));
        for source in [
            "echo 'unterminated",
            "if then fi",
            "bash -c \"$SCRIPT\"",
            "eval $X",
            "bash $FLAGS -c x",
            nested.as_str(),
        ] {
            assert_eq!(commands(source, 0), None, "{source}");
        }
        // Deep syntactic nesting is walked without recursion.
        let deep = format!("{}true{}", "( ".repeat(5_000), " )".repeat(5_000));
        assert!(commands(&deep, 0).is_some_and(|commands| commands.len() == 1));
    }

    #[test]
    fn patterns_match_word_by_word_with_a_trailing_rest() {
        let matches = |pattern: &str, source: &str| {
            let argv = commands(source, 0).unwrap().remove(0);
            Pattern::compile(pattern).unwrap().matches(&argv)
        };
        for (pattern, source, expected) in [
            ("git status", "git status", true),
            ("git status", "git status -s", false),
            ("git status *", "git status", true),
            ("git status *", "git status -s $X", true),
            ("git *", "git", true),
            ("git * --force", "git push --force", true),
            ("git * --force", "git $REMOTE --force", false),
            ("cargo test:*", "cargo test:unit", true),
            ("cargo --features=*", "cargo --features=a,b", true),
            ("git", "/usr/bin/git", false),
            ("*", "$CMD", true),
            ("", "true", false),
        ] {
            assert_eq!(matches(pattern, source), expected, "{pattern} / {source}");
        }
        assert!(Pattern::compile("git [").is_err());
    }

    #[test]
    fn check_requires_every_command_to_match() {
        let rules = allow(&["git status *", "ls *", "cd *", "sudo *", "bash *"]);
        for (command, expected) in [
            ("git status && ls -la; cd src", None),
            ("sudo ls", None),
            ("bash -c 'ls'", None),
            (
                "ls | rm -rf /",
                Some(("no policy.shell.allow pattern matches", "rm -rf /")),
            ),
            (
                "ls $(git push)",
                Some(("no policy.shell.allow pattern matches", "git push")),
            ),
            (
                "sudo rm x",
                Some(("no policy.shell.allow pattern matches", "rm x")),
            ),
            (
                "bash -c 'git push'",
                Some(("no policy.shell.allow pattern matches", "git push")),
            ),
            (
                "$CMD",
                Some(("no policy.shell.allow pattern matches", "<expansion>")),
            ),
            ("ls 'x", Some(("could not be parsed statically", "ls 'x"))),
        ] {
            let expected = expected.map(|(reason, fragment)| (reason, fragment.to_owned()));
            assert_eq!(
                rejection(check(&rules, command, true)),
                expected,
                "{command}"
            );
        }
    }

    #[test]
    fn check_handles_wildcards_non_posix_shells_and_bad_patterns() {
        for (rules, command, posix, expected) in [
            (allow(&["*"]), "anything 'unparseable", false, None),
            (allow(&[" * ", "ls"]), "rm -rf /", true, None),
            (
                allow(&[]),
                "true",
                true,
                Some("no policy.shell.allow pattern matches"),
            ),
            (
                allow(&["ls *"]),
                "ls",
                false,
                Some(
                    "the detected shell is not POSIX, so only a `*` allow pattern permits commands",
                ),
            ),
            (
                allow(&["ls [*"]),
                "ls",
                true,
                Some("invalid policy.shell.allow pattern"),
            ),
        ] {
            assert_eq!(
                rejection(check(&rules, command, posix)).map(|(reason, _)| reason),
                expected,
                "{rules:?} / {command}"
            );
        }
    }
}
