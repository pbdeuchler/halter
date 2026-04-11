// pattern: Functional Core

use std::path::Path;

use ast_grep_core::MatchStrictness;
use ast_grep_language::SupportLang;

const SUPPORTED_LANGUAGES: &[&str] = &[
    "bash",
    "c",
    "cpp",
    "csharp",
    "css",
    "elixir",
    "go",
    "haskell",
    "hcl",
    "html",
    "java",
    "javascript",
    "json",
    "kotlin",
    "lua",
    "nix",
    "php",
    "python",
    "ruby",
    "rust",
    "scala",
    "solidity",
    "swift",
    "tsx",
    "typescript",
    "yaml",
];

pub(super) fn parse_strictness(value: Option<&str>) -> anyhow::Result<MatchStrictness> {
    match value.unwrap_or("smart") {
        "cst" => Ok(MatchStrictness::Cst),
        "smart" => Ok(MatchStrictness::Smart),
        "ast" => Ok(MatchStrictness::Ast),
        "relaxed" => Ok(MatchStrictness::Relaxed),
        "signature" => Ok(MatchStrictness::Signature),
        "template" => Ok(MatchStrictness::Template),
        other => anyhow::bail!("failed to execute ast_grep tool: unsupported strictness '{other}'"),
    }
}

pub(super) fn resolve_language(
    explicit_lang: Option<&str>,
    file_path: &Path,
) -> anyhow::Result<SupportLang> {
    if let Some(lang) = explicit_lang.map(str::trim).filter(|lang| !lang.is_empty()) {
        return resolve_explicit_language(lang);
    }

    infer_language_from_path(file_path).ok_or_else(|| {
        anyhow::anyhow!(
            "failed to execute ast_grep tool: unable to infer language for '{}'; set `lang` explicitly",
            file_path.display()
        )
    })
}

pub(super) fn infer_language_from_path(path: &Path) -> Option<SupportLang> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    let alias = match extension.as_str() {
        "sh" | "bash" | "zsh" => "bash",
        "c" | "h" => "c",
        "cc" | "cp" | "cpp" | "cxx" | "hh" | "hpp" | "hxx" => "cpp",
        "cs" => "csharp",
        "css" => "css",
        "ex" | "exs" => "elixir",
        "go" => "go",
        "hs" => "haskell",
        "hcl" | "tf" | "tfvars" => "hcl",
        "htm" | "html" => "html",
        "java" => "java",
        "cjs" | "js" | "jsx" | "mjs" => "javascript",
        "json" => "json",
        "kt" | "kts" => "kotlin",
        "lua" => "lua",
        "nix" => "nix",
        "php" => "php",
        "py" => "python",
        "rb" => "ruby",
        "rs" => "rust",
        "scala" | "sc" => "scala",
        "sol" => "solidity",
        "swift" => "swift",
        "tsx" => "tsx",
        "cts" | "mts" | "ts" => "typescript",
        "yaml" | "yml" => "yaml",
        _ => return None,
    };
    resolve_explicit_language(alias).ok()
}

pub(super) fn canonical_name(language: SupportLang) -> &'static str {
    match language {
        SupportLang::Bash => "bash",
        SupportLang::C => "c",
        SupportLang::Cpp => "cpp",
        SupportLang::CSharp => "csharp",
        SupportLang::Css => "css",
        SupportLang::Elixir => "elixir",
        SupportLang::Go => "go",
        SupportLang::Haskell => "haskell",
        SupportLang::Hcl => "hcl",
        SupportLang::Html => "html",
        SupportLang::Java => "java",
        SupportLang::JavaScript => "javascript",
        SupportLang::Json => "json",
        SupportLang::Kotlin => "kotlin",
        SupportLang::Lua => "lua",
        SupportLang::Nix => "nix",
        SupportLang::Php => "php",
        SupportLang::Python => "python",
        SupportLang::Ruby => "ruby",
        SupportLang::Rust => "rust",
        SupportLang::Scala => "scala",
        SupportLang::Solidity => "solidity",
        SupportLang::Swift => "swift",
        SupportLang::Tsx => "tsx",
        SupportLang::TypeScript => "typescript",
        SupportLang::Yaml => "yaml",
        _ => "unknown",
    }
}

fn resolve_explicit_language(value: &str) -> anyhow::Result<SupportLang> {
    match value.to_ascii_lowercase().as_str() {
        "bash" | "sh" | "shell" | "zsh" => Ok(SupportLang::Bash),
        "c" => Ok(SupportLang::C),
        "cpp" | "c++" | "cc" | "cxx" => Ok(SupportLang::Cpp),
        "csharp" | "c#" | "cs" => Ok(SupportLang::CSharp),
        "css" => Ok(SupportLang::Css),
        "elixir" | "ex" => Ok(SupportLang::Elixir),
        "go" | "golang" => Ok(SupportLang::Go),
        "haskell" | "hs" => Ok(SupportLang::Haskell),
        "hcl" | "terraform" | "tf" => Ok(SupportLang::Hcl),
        "html" | "htm" => Ok(SupportLang::Html),
        "java" => Ok(SupportLang::Java),
        "javascript" | "js" | "jsx" => Ok(SupportLang::JavaScript),
        "json" => Ok(SupportLang::Json),
        "kotlin" | "kt" | "kts" => Ok(SupportLang::Kotlin),
        "lua" => Ok(SupportLang::Lua),
        "nix" => Ok(SupportLang::Nix),
        "php" => Ok(SupportLang::Php),
        "python" | "py" => Ok(SupportLang::Python),
        "ruby" | "rb" => Ok(SupportLang::Ruby),
        "rust" | "rs" => Ok(SupportLang::Rust),
        "scala" | "sc" => Ok(SupportLang::Scala),
        "solidity" | "sol" => Ok(SupportLang::Solidity),
        "swift" => Ok(SupportLang::Swift),
        "tsx" => Ok(SupportLang::Tsx),
        "typescript" | "ts" => Ok(SupportLang::TypeScript),
        "yaml" | "yml" => Ok(SupportLang::Yaml),
        other => anyhow::bail!(
            "failed to execute ast_grep tool: unsupported language '{other}'; supported languages: {}",
            SUPPORTED_LANGUAGES.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_language_from_extension() {
        assert_eq!(
            infer_language_from_path(Path::new("src/lib.rs")),
            Some(SupportLang::Rust)
        );
        assert_eq!(
            infer_language_from_path(Path::new("src/app.tsx")),
            Some(SupportLang::Tsx)
        );
        assert_eq!(
            infer_language_from_path(Path::new("config.yaml")),
            Some(SupportLang::Yaml)
        );
        assert_eq!(infer_language_from_path(Path::new("README.md")), None);
    }
}
