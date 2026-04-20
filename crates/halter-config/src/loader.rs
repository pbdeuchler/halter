// pattern: Imperative Shell

use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::Context;
use schemars::schema_for;
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use tokio::fs;
use tracing::{debug, info};

use crate::schema::{HarnessConfig, resolve_provider_runtime_config};

#[derive(Debug, Clone, Default)]
pub struct LayeredConfigPaths {
    pub user_config: Option<PathBuf>,
    pub project_config: Option<PathBuf>,
    pub explicit_config: Option<PathBuf>,
}

pub async fn load_path(path: impl AsRef<Path>) -> anyhow::Result<HarnessConfig> {
    let path = path.as_ref();
    debug!(path = %path.display(), "loading config file");
    let contents = fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read config at {}", path.display()))?;
    let mut value = parse_toml(&contents)
        .with_context(|| format!("failed to parse config at {}", path.display()))?;
    apply_env_overrides(&mut value)?;
    let config: HarnessConfig = value
        .try_into()
        .context("failed to decode configuration after applying overrides")?;
    config.validate()?;
    validate_runtime_requirements(&config)?;
    let model = config.default_model()?;
    let subagent_model = config.subagent_model().unwrap_or(model);
    info!(
        path = %path.display(),
        default_provider = %model.provider,
        default_model = %model.model,
        subagent_provider = %subagent_model.provider,
        subagent_model = %subagent_model.model,
        session_backend = ?config.sessions.backend,
        "loaded config"
    );
    Ok(config)
}

pub async fn load_layered(paths: LayeredConfigPaths) -> anyhow::Result<HarnessConfig> {
    debug!(
        user_config = ?paths.user_config.as_ref().map(|path| path.display().to_string()),
        project_config = ?paths.project_config.as_ref().map(|path| path.display().to_string()),
        explicit_config = ?paths.explicit_config.as_ref().map(|path| path.display().to_string()),
        "loading layered config"
    );
    let mut merged = toml::Value::try_from(HarnessConfig::default())
        .context("failed to serialize built-in config defaults")?;

    for path in [
        paths.user_config,
        paths.project_config,
        paths.explicit_config,
    ]
    .into_iter()
    .flatten()
    {
        if !path.exists() {
            debug!(path = %path.display(), "skipping missing config layer");
            continue;
        }
        debug!(path = %path.display(), "merging config layer");
        let raw = fs::read_to_string(&path)
            .await
            .with_context(|| format!("failed to read config at {}", path.display()))?;
        let patch = parse_toml(&raw)
            .with_context(|| format!("failed to parse config at {}", path.display()))?;
        merge_toml(&mut merged, patch);
    }

    apply_env_overrides(&mut merged)?;

    let config: HarnessConfig = merged
        .try_into()
        .context("failed to decode layered configuration")?;
    config.validate()?;
    validate_runtime_requirements(&config)?;
    let model = config.default_model()?;
    let subagent_model = config.subagent_model().unwrap_or(model);
    info!(
        default_provider = %model.provider,
        default_model = %model.model,
        subagent_provider = %subagent_model.provider,
        subagent_model = %subagent_model.model,
        session_backend = ?config.sessions.backend,
        "loaded layered config"
    );
    Ok(config)
}

pub fn apply_env_overrides(config: &mut toml::Value) -> anyhow::Result<()> {
    apply_env_overrides_with(config, |name| env::var_os(name))
}

/// Renders the `HarnessConfig` JSON Schema as formatted JSON text for CLI and SDK callers.
pub fn export_json_schema() -> anyhow::Result<String> {
    let schema = schema_for!(HarnessConfig);
    serde_json::to_string_pretty(&schema).context("failed to render json schema")
}

pub fn generate_starter_config() -> String {
    let sqlite_comment = if cfg!(feature = "sqlite") {
        "\n[sessions]\nbackend = \"memory\"\n# sqlite_path = \"/tmp/halter/sessions.db\"\n"
    } else {
        "\n[sessions]\nbackend = \"memory\"\n"
    };

    format!(
        r#"version = 1

[models.default]
provider = "openai"
model = "gpt-5"
reasoning = "medium"

# [models.subagent]
# provider = "openai"
# model = "gpt-5-mini"
# reasoning = "medium"

[resources.skills]
roots = ["./.agent/skills"]

[resources.plugins]
roots = ["./.agent/plugins"]

[policy]
allowed_write_roots = ["./", "/tmp/halter"]
max_read_bytes = 1048576
max_subagent_depth = 3
max_concurrent_subagents = 8

[policy.shell]
enabled = true
allow = ["git", "cargo", "rg", "ls", "find"]
timeout_secs = 30

[policy.network]
enabled = false
allowed_hosts = []
{sqlite_comment}"#
    )
}

fn validate_runtime_requirements(config: &HarnessConfig) -> anyhow::Result<()> {
    validate_runtime_requirements_with(config, |name| env::var_os(name))
}

fn parse_toml(contents: &str) -> anyhow::Result<toml::Value> {
    toml::from_str(contents).context("invalid TOML")
}

fn apply_env_overrides_with<F>(config: &mut toml::Value, mut lookup: F) -> anyhow::Result<()>
where
    F: FnMut(&str) -> Option<OsString>,
{
    for spec in env_override_specs() {
        let Some(raw) = lookup(spec.env_var) else {
            continue;
        };
        debug!(
            env_var = spec.env_var,
            path = spec.path.join("."),
            "applying env override"
        );
        let raw = raw
            .into_string()
            .map_err(|value| anyhow::anyhow!("invalid utf-8 in env override {:?}", value))?;
        let value = (spec.parse)(&raw)?;
        set_toml_path(config, spec.path, value);
    }

    Ok(())
}

fn validate_runtime_requirements_with<F>(
    config: &HarnessConfig,
    mut lookup: F,
) -> anyhow::Result<()>
where
    F: FnMut(&str) -> Option<OsString>,
{
    for provider in [
        Some(config.default_model()?.provider),
        config.subagent_model().map(|model| model.provider),
    ]
    .into_iter()
    .flatten()
    {
        resolve_provider_runtime_config(provider, config.provider_config(provider), |name| {
            let Some(raw) = lookup(name) else {
                return Ok(None);
            };
            let value = raw
                .into_string()
                .map_err(|_| anyhow::anyhow!("invalid utf-8 in {}", name))?;
            Ok(Some(value))
        })?;
    }
    Ok(())
}

fn parse_bool(value: &str) -> anyhow::Result<bool> {
    value
        .parse::<bool>()
        .with_context(|| format!("invalid boolean override value '{value}'"))
}

fn parse_string(value: &str) -> anyhow::Result<toml::Value> {
    Ok(toml::Value::String(value.to_owned()))
}

fn parse_bool_value(value: &str) -> anyhow::Result<toml::Value> {
    Ok(toml::Value::Boolean(parse_bool(value)?))
}

fn parse_colon_list(value: &str) -> anyhow::Result<toml::Value> {
    parse_list(value, ':')
}

fn parse_comma_list(value: &str) -> anyhow::Result<toml::Value> {
    parse_list(value, ',')
}

fn parse_list(value: &str, separator: char) -> anyhow::Result<toml::Value> {
    let values = value
        .split(separator)
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| toml::Value::String(entry.to_owned()))
        .collect::<Vec<_>>();
    Ok(toml::Value::Array(values))
}

struct EnvOverrideSpec {
    env_var: &'static str,
    path: &'static [&'static str],
    parse: fn(&str) -> anyhow::Result<toml::Value>,
}

fn env_override_specs() -> &'static [EnvOverrideSpec] {
    &[
        EnvOverrideSpec {
            env_var: "HALTER_SESSION_BACKEND",
            path: &["sessions", "backend"],
            parse: parse_string,
        },
        EnvOverrideSpec {
            env_var: "HALTER_POLICY_SHELL_ENABLED",
            path: &["policy", "shell", "enabled"],
            parse: parse_bool_value,
        },
        EnvOverrideSpec {
            env_var: "HALTER_POLICY_NETWORK_ENABLED",
            path: &["policy", "network", "enabled"],
            parse: parse_bool_value,
        },
        EnvOverrideSpec {
            env_var: "HALTER_SKILL_ROOTS",
            path: &["resources", "skills", "roots"],
            parse: parse_colon_list,
        },
        EnvOverrideSpec {
            env_var: "HALTER_PLUGIN_ROOTS",
            path: &["resources", "plugins", "roots"],
            parse: parse_colon_list,
        },
        EnvOverrideSpec {
            env_var: "HALTER_POLICY_SHELL_ALLOW",
            path: &["policy", "shell", "allow"],
            parse: parse_comma_list,
        },
        EnvOverrideSpec {
            env_var: "HALTER_POLICY_ALLOWED_HOSTS",
            path: &["policy", "network", "allowed_hosts"],
            parse: parse_comma_list,
        },
        EnvOverrideSpec {
            env_var: "HALTER_TOOLS_ENABLED",
            path: &["tools", "enabled"],
            parse: parse_comma_list,
        },
    ]
}

fn set_toml_path(root: &mut toml::Value, path: &[&str], value: toml::Value) {
    if path.is_empty() {
        *root = value;
        return;
    }

    let mut cursor = root;
    for segment in &path[..path.len() - 1] {
        let table = cursor
            .as_table_mut()
            .expect("toml path traversal requires a table");
        cursor = table
            .entry((*segment).to_owned())
            .or_insert_with(|| toml::Value::Table(Default::default()));
    }

    let table = cursor
        .as_table_mut()
        .expect("terminal toml path traversal requires a table");
    table.insert(path[path.len() - 1].to_owned(), value);
}

fn merge_toml(base: &mut toml::Value, patch: toml::Value) {
    merge_toml_at_path(base, patch, &mut Vec::new());
}

fn merge_toml_at_path(base: &mut toml::Value, patch: toml::Value, path: &mut Vec<String>) {
    match (base, patch) {
        (toml::Value::Table(base_table), toml::Value::Table(patch_table)) => {
            for (key, value) in patch_table {
                path.push(key.clone());
                match base_table.get_mut(&key) {
                    Some(existing) => merge_toml_at_path(existing, value, path),
                    None => {
                        base_table.insert(key, value);
                    }
                }
                path.pop();
            }
        }
        (toml::Value::Array(base_array), toml::Value::Array(patch_array)) => {
            match array_merge_policy(path) {
                ArrayMergePolicy::AppendDedupe => {
                    append_unique_values(base_array, patch_array);
                }
                ArrayMergePolicy::Replace => {
                    *base_array = patch_array;
                }
            }
        }
        (base_slot, patch_value) => *base_slot = patch_value,
    }
}

fn append_unique_values(base: &mut Vec<toml::Value>, patch: Vec<toml::Value>) {
    for value in patch {
        if !base.iter().any(|existing| existing == &value) {
            base.push(value);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrayMergePolicy {
    AppendDedupe,
    Replace,
}

fn array_merge_policy(path: &[String]) -> ArrayMergePolicy {
    match path
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["resources", "skills", "roots"] | ["resources", "plugins", "roots"] => {
            ArrayMergePolicy::AppendDedupe
        }
        ["policy", "shell", "allow"]
        | ["policy", "allowed_write_roots"]
        | ["policy", "network", "allowed_hosts"]
        | ["tools", "enabled"] => ArrayMergePolicy::Replace,
        _ => ArrayMergePolicy::Replace,
    }
}

/// Returns a stable hash of the effective config for change detection and caches.
#[must_use]
pub fn config_fingerprint(config: &HarnessConfig) -> String {
    let json = serde_json::to_vec(config).expect("config fingerprint requires serialization");
    let mut hasher = Sha256::new();
    hasher.update(json);
    format!("{:x}", hasher.finalize())
}

/// Expands a path's leading `~/` against `$HOME`. No other substitutions are
/// performed — `$VAR`, `${VAR}`, `~user/`, `%USERPROFILE%`, and shell escapes
/// all pass through unchanged. This is deliberate: expanding env-var
/// references on a user-supplied config path is a footgun (an attacker-
/// controlled environment variable could redirect resource loads). Callers
/// that *want* shell-like expansion must call a shell-expansion crate
/// explicitly with an explicit threat model.
#[must_use]
pub fn expand_path(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if let Some(stripped) = raw.strip_prefix("~/")
        && let Ok(home) = env::var("HOME")
    {
        return PathBuf::from(home).join(stripped);
    }
    PathBuf::from(raw.as_ref())
}

/// Renders the `HarnessConfig` JSON Schema as a serde value for callers that need to inspect it.
#[must_use]
pub fn schema_as_json_value() -> JsonValue {
    serde_json::to_value(schema_for!(HarnessConfig)).expect("schema conversion must succeed")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    #[tokio::test]
    async fn layered_config_merges_predictably() {
        let dir = tempfile::tempdir().expect("tempdir");
        let explicit = dir.path().join("halter.toml");
        fs::write(
            &explicit,
            r#"
[models.default]
provider = "openai"
model = "gpt-5"

[providers.openai]
api_key = "test-key"

[policy]
max_read_bytes = 99
"#,
        )
        .await
        .expect("write config");

        let loaded = load_layered(LayeredConfigPaths {
            explicit_config: Some(explicit),
            ..LayeredConfigPaths::default()
        })
        .await
        .expect("load layered config");

        assert_eq!(loaded.policy.max_read_bytes, 99);
        assert_eq!(
            loaded.default_model().expect("default model").model,
            "gpt-5"
        );
    }

    #[test]
    fn starter_config_is_parseable() {
        let parsed = parse_toml(&generate_starter_config()).expect("parse starter config");
        let config: HarnessConfig = parsed.try_into().expect("decode starter config");
        config.validate().expect("starter config should validate");
        validate_runtime_requirements_with(&config, |name| {
            Some(OsString::from(match name {
                "OPENAI_API_KEY" => "test-key",
                _ => unreachable!("unexpected env var"),
            }))
        })
        .expect("starter config runtime requirements should validate");
    }

    #[test]
    fn example_config_is_parseable() {
        let parsed = parse_toml(include_str!("../../../examples/halter.example.toml"))
            .expect("parse example config");
        let config: HarnessConfig = parsed.try_into().expect("decode example config");
        config.validate().expect("example config should validate");
        validate_runtime_requirements_with(&config, |name| {
            Some(OsString::from(match name {
                "OPENAI_API_KEY" | "OPENROUTER_API_KEY" => "test-key",
                _ => unreachable!("unexpected env var: {name}"),
            }))
        })
        .expect("example config runtime requirements should validate");
    }

    #[test]
    fn runtime_requirements_include_subagent_provider_credentials() {
        let parsed = parse_toml(
            r#"
version = 1

[models.default]
provider = "openai"
model = "gpt-5"

[models.subagent]
provider = "openrouter"
model = "openai/gpt-5-mini"

[providers.openai]
api_key = "openai-key"
"#,
        )
        .expect("parse config");
        let config: HarnessConfig = parsed.try_into().expect("decode config");
        config.validate().expect("config should validate");

        let error = validate_runtime_requirements_with(&config, |name| match name {
            "OPENAI_API_KEY" => Some(OsString::from("openai-key")),
            _ => None,
        })
        .expect_err("runtime requirements should fail");

        assert!(error.to_string().contains("OPENROUTER_API_KEY"));
    }

    #[test]
    fn env_overrides_apply_from_table() {
        let mut value = toml::Value::try_from(HarnessConfig::default()).expect("defaults");
        let overrides = BTreeMap::from([
            ("HALTER_SESSION_BACKEND", "memory"),
            ("HALTER_POLICY_SHELL_ENABLED", "false"),
            ("HALTER_SKILL_ROOTS", "./skills:./vendor/skills"),
            ("HALTER_POLICY_SHELL_ALLOW", "git,just"),
            ("HALTER_TOOLS_ENABLED", "read,glob"),
        ]);

        apply_env_overrides_with(&mut value, |name| overrides.get(name).map(OsString::from))
            .expect("apply overrides");

        let decoded: HarnessConfig = value.try_into().expect("decode config");
        assert_eq!(
            decoded.sessions.backend,
            crate::schema::SessionBackend::Memory
        );
        assert!(!decoded.policy.shell.enabled);
        assert_eq!(
            decoded.resources.skills.roots,
            vec![PathBuf::from("./skills"), PathBuf::from("./vendor/skills")]
        );
        assert_eq!(decoded.policy.shell.allow, vec!["git", "just"]);
        assert_eq!(decoded.tools.enabled, vec!["read", "glob"]);
    }

    #[cfg(feature = "sqlite")]
    #[test]
    fn env_overrides_can_select_sqlite_backend() {
        let mut value = toml::Value::try_from(HarnessConfig::default()).expect("defaults");
        let overrides = BTreeMap::from([("HALTER_SESSION_BACKEND", "sqlite")]);

        apply_env_overrides_with(&mut value, |name| overrides.get(name).map(OsString::from))
            .expect("apply overrides");

        let decoded: HarnessConfig = value.try_into().expect("decode config");
        assert_eq!(
            decoded.sessions.backend,
            crate::schema::SessionBackend::Sqlite
        );
    }

    #[test]
    fn unsupported_session_backend_is_rejected_during_decode() {
        let parsed = parse_toml(
            r#"
version = 1

[models.default]
provider = "openai"
model = "gpt-5"

[providers.openai]
api_key = "test-key"

[sessions]
backend = "flat_file"
"#,
        )
        .expect("parse config");

        let error: Result<HarnessConfig, _> = parsed.try_into();
        let error = error.expect_err("unsupported backend should fail");

        assert!(error.to_string().contains("unknown variant `flat_file`"));
    }

    #[tokio::test]
    async fn layered_config_appends_roots_but_replaces_allowlists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let user = dir.path().join("user.toml");
        let project = dir.path().join("project.toml");

        fs::write(
            &user,
            r#"
[models.default]
provider = "openai"
model = "gpt-5"

[providers.openai]
api_key = "test-key"

[resources.skills]
roots = ["./user-skills"]

[policy.shell]
allow = ["git"]
"#,
        )
        .await
        .expect("write user config");
        fs::write(
            &project,
            r#"
[resources.skills]
roots = ["./project-skills"]

[policy.shell]
allow = ["just"]
"#,
        )
        .await
        .expect("write project config");

        let loaded = load_layered(LayeredConfigPaths {
            user_config: Some(user),
            project_config: Some(project),
            ..LayeredConfigPaths::default()
        })
        .await
        .expect("load layered config");

        assert_eq!(
            loaded.resources.skills.roots,
            vec![
                PathBuf::from("./user-skills"),
                PathBuf::from("./project-skills")
            ]
        );
        assert_eq!(loaded.policy.shell.allow, vec!["just"]);
    }
}
