// pattern: Functional Core

//! Typed, once-compiled matcher for hook event-matcher strings.
//!
//! Patterns are compiled at config-load time via `CompiledMatcher::compile`.
//! Invalid patterns fail at that boundary; the engine consumes the compiled
//! matcher and never sees a raw string (H22/H27, defense-in-depth). Glob
//! patterns use `globset` with subtree semantics: `*.example.com` matches
//! `api.example.com` and `api.prod.example.com` alike (H14). The hand-rolled
//! `wildcard_match_impl` is retired in favor of the same globset path (M30).

use globset::{Glob, GlobBuilder, GlobMatcher};
use regex::Regex;
use thiserror::Error;

/// A matcher pattern compiled once at config-parse time.
#[derive(Debug, Clone)]
pub enum CompiledMatcher {
    /// Case-insensitive literal match. No wildcards, no regex metacharacters.
    Exact(String),
    /// `globset`-backed glob match. `*.example.com` matches subtree.
    Glob(GlobMatcher),
    /// Regular-expression match.
    Regex(Regex),
}

#[derive(Debug, Error)]
pub enum MatcherCompileError {
    #[error("invalid regex pattern '{pattern}': {source}")]
    Regex {
        pattern: String,
        #[source]
        source: regex::Error,
    },
    #[error("invalid glob pattern '{pattern}': {source}")]
    Glob {
        pattern: String,
        #[source]
        source: globset::Error,
    },
}

impl CompiledMatcher {
    /// Compile an event-matcher pattern. Patterns containing regex
    /// metacharacters are treated as regex; patterns containing only glob
    /// wildcards (`*`, `?`) are compiled as globs; everything else is a
    /// case-insensitive literal.
    pub fn compile(pattern: &str) -> Result<Self, MatcherCompileError> {
        let trimmed = pattern.trim();
        if trimmed.is_empty() {
            // Empty matcher compiles to an "Exact empty" marker. Callers
            // generally filter empty matchers before reaching here, but we
            // accept it gracefully.
            return Ok(Self::Exact(String::new()));
        }
        if looks_like_regex(trimmed) {
            let regex = Regex::new(trimmed).map_err(|source| MatcherCompileError::Regex {
                pattern: trimmed.to_owned(),
                source,
            })?;
            return Ok(Self::Regex(regex));
        }
        if looks_like_glob(trimmed) {
            return Self::compile_glob(trimmed);
        }
        Ok(Self::Exact(trimmed.to_owned()))
    }

    /// Force-compile a pattern as a regex, regardless of shape. Used when
    /// the caller wants regex semantics explicitly (event matcher strings in
    /// `hooks.json`).
    pub fn compile_regex(pattern: &str) -> Result<Self, MatcherCompileError> {
        let trimmed = pattern.trim();
        let regex = Regex::new(trimmed).map_err(|source| MatcherCompileError::Regex {
            pattern: trimmed.to_owned(),
            source,
        })?;
        Ok(Self::Regex(regex))
    }

    /// Force-compile a pattern as a glob. `*.example.com` becomes a subtree
    /// match. The bare `*` is retained as a universal match.
    pub fn compile_glob(pattern: &str) -> Result<Self, MatcherCompileError> {
        let trimmed = pattern.trim();
        if trimmed == "*" {
            // Any input matches. Express as a glob that matches any sequence,
            // including the empty string.
            let glob = Glob::new("*").map_err(|source| MatcherCompileError::Glob {
                pattern: trimmed.to_owned(),
                source,
            })?;
            return Ok(Self::Glob(glob.compile_matcher()));
        }
        let rewritten = rewrite_glob_for_subtree(trimmed);
        let glob = GlobBuilder::new(&rewritten)
            .case_insensitive(true)
            .literal_separator(false)
            .build()
            .map_err(|source| MatcherCompileError::Glob {
                pattern: trimmed.to_owned(),
                source,
            })?;
        Ok(Self::Glob(glob.compile_matcher()))
    }

    pub fn is_match(&self, input: &str) -> bool {
        match self {
            Self::Exact(literal) => literal.eq_ignore_ascii_case(input),
            Self::Glob(matcher) => matcher.is_match(input),
            Self::Regex(regex) => regex.is_match(input),
        }
    }
}

/// `*.example.com` should be a subtree match (matching `a.b.example.com` too).
/// The ASCII shell-glob-style `*` in globset matches a single path segment by
/// default; rewrite a leading `*.` to `**.` to cross the `.` separator.
fn rewrite_glob_for_subtree(pattern: &str) -> String {
    if let Some(rest) = pattern.strip_prefix("*.") {
        format!("**.{rest}")
    } else {
        pattern.to_owned()
    }
}

fn looks_like_regex(pattern: &str) -> bool {
    pattern.chars().any(|ch| {
        matches!(
            ch,
            '[' | ']' | '(' | ')' | '{' | '}' | '+' | '^' | '$' | '\\' | '|'
        )
    })
}

fn looks_like_glob(pattern: &str) -> bool {
    pattern.chars().any(|ch| matches!(ch, '*' | '?'))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AC3.1: `*.example.com` matches `api.example.com`,
    /// `api.prod.example.com`, and deeply nested subdomains.
    #[test]
    fn review_hook_runtime_ac3_1_glob_matches_subtree() {
        let matcher = CompiledMatcher::compile_glob("*.example.com").expect("compile");
        assert!(matcher.is_match("api.example.com"));
        assert!(matcher.is_match("api.prod.example.com"));
        assert!(matcher.is_match("deeply.nested.example.com"));
    }

    /// AC3.2: `*` matches any host/event string.
    #[test]
    fn review_hook_runtime_ac3_2_bare_star_matches_anything() {
        let matcher = CompiledMatcher::compile_glob("*").expect("compile");
        assert!(matcher.is_match(""));
        assert!(matcher.is_match("literally anything"));
        assert!(matcher.is_match("api.example.com"));
    }

    /// AC3.3: literal patterns match only themselves.
    #[test]
    fn review_hook_runtime_ac3_3_literal_matches_only_exact() {
        let matcher = CompiledMatcher::compile("api.example.com").expect("compile");
        assert!(matcher.is_match("api.example.com"));
        assert!(!matcher.is_match("api2.example.com"));
        assert!(!matcher.is_match("api.example.com.evil.com"));
        // Case-insensitive by design (HTTP hosts and hook event identifiers
        // are both case-insensitive).
        assert!(matcher.is_match("API.Example.Com"));
    }

    /// AC3.4: invalid regex fails at compile time with a parse error.
    #[test]
    fn review_hook_runtime_ac3_4_invalid_regex_rejected_at_compile() {
        let error = CompiledMatcher::compile_regex("(").expect_err("invalid regex must reject");
        assert!(matches!(error, MatcherCompileError::Regex { .. }));
    }

    /// AC3.6: glob parity with the retired `wildcard_match_impl`.
    #[test]
    fn review_hook_runtime_ac3_6_glob_parity_with_legacy_matcher() {
        // Legacy wildcard semantics: case-insensitive, `*` matches any run of
        // characters, literal tokens match exactly.
        let pairs: &[(&str, &str, bool)] = &[
            ("git *", "git status", true),
            ("git *", "cargo test", false),
            ("shell", "Shell", true),
            ("*", "anything", true),
            ("read*", "readtokens", true),
            ("read*", "write", false),
        ];
        for (pattern, candidate, expected) in pairs {
            let matcher = CompiledMatcher::compile(pattern).expect("compile");
            assert_eq!(
                matcher.is_match(candidate),
                *expected,
                "pattern {pattern} vs {candidate}",
            );
        }
    }
}
