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

/// Canonical extension-to-language aliases. Ordered roughly by expected
/// frequency so the linear scan in `infer_language_from_path` hits common
/// cases first. Using a const table with `eq_ignore_ascii_case` avoids the
/// per-file `to_ascii_lowercase` allocation that the prior `match` required
/// (finding L26).
const EXTENSION_ALIASES: &[(&str, &str)] = &[
    ("rs", "rust"),
    ("ts", "typescript"),
    ("cts", "typescript"),
    ("mts", "typescript"),
    ("tsx", "tsx"),
    ("js", "javascript"),
    ("cjs", "javascript"),
    ("jsx", "javascript"),
    ("mjs", "javascript"),
    ("py", "python"),
    ("go", "go"),
    ("rb", "ruby"),
    ("java", "java"),
    ("kt", "kotlin"),
    ("kts", "kotlin"),
    ("swift", "swift"),
    ("scala", "scala"),
    ("sc", "scala"),
    ("c", "c"),
    ("h", "c"),
    ("cc", "cpp"),
    ("cp", "cpp"),
    ("cpp", "cpp"),
    ("cxx", "cpp"),
    ("hh", "cpp"),
    ("hpp", "cpp"),
    ("hxx", "cpp"),
    ("cs", "csharp"),
    ("css", "css"),
    ("ex", "elixir"),
    ("exs", "elixir"),
    ("hs", "haskell"),
    ("hcl", "hcl"),
    ("tf", "hcl"),
    ("tfvars", "hcl"),
    ("htm", "html"),
    ("html", "html"),
    ("json", "json"),
    ("lua", "lua"),
    ("nix", "nix"),
    ("php", "php"),
    ("sh", "bash"),
    ("bash", "bash"),
    ("zsh", "bash"),
    ("sol", "solidity"),
    ("yaml", "yaml"),
    ("yml", "yaml"),
];

pub(super) fn infer_language_from_path(path: &Path) -> Option<SupportLang> {
    let extension = path.extension()?.to_str()?;
    let alias = EXTENSION_ALIASES
        .iter()
        .find(|(candidate, _)| extension.eq_ignore_ascii_case(candidate))
        .map(|(_, alias)| *alias)?;
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
