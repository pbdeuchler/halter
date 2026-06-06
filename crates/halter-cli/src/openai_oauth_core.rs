// pattern: Functional Core

use anyhow::Context;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use reqwest::Url;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const OPENAI_AUTH_CLAIM: &str = "https://api.openai.com/auth";

pub(crate) const DEFAULT_ISSUER: &str = "https://auth.openai.com";
pub(crate) const DEFAULT_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub(crate) const DEFAULT_SCOPE: &str =
    "openid profile email offline_access api.connectors.read api.connectors.invoke";
pub(crate) const DEFAULT_ORIGINATOR: &str = "codex_cli_rs";
pub(crate) const DEFAULT_PORT: u16 = 1455;
pub(crate) const DEFAULT_FALLBACK_PORT: u16 = 1457;
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 600;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PkceCodes {
    pub code_verifier: String,
    pub code_challenge: String,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct AuthorizeUrlParams<'a> {
    pub issuer: &'a str,
    pub client_id: &'a str,
    pub redirect_uri: &'a str,
    pub scope: &'a str,
    pub pkce: &'a PkceCodes,
    pub state: &'a str,
    pub originator: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParsedCallback {
    Code(String),
    IgnoredPath,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CallbackError {
    InvalidRequestTarget,
    StateMismatch,
    OAuth {
        code: String,
        description: Option<String>,
    },
    MissingAuthorizationCode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenEndpointErrorDetail {
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub display_message: String,
}

impl std::fmt::Display for TokenEndpointErrorDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.display_message.fmt(f)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum JwtPayloadError {
    Malformed,
    Base64(String),
    Json(String),
}

impl std::fmt::Display for JwtPayloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Malformed => f.write_str("malformed JWT"),
            Self::Base64(error) => write!(f, "invalid JWT payload base64: {error}"),
            Self::Json(error) => write!(f, "invalid JWT payload JSON: {error}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ApiKeyExchangeReadiness {
    Ready,
    MissingOrganizationId {
        default_organization_id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct OpenAiOAuthOutput {
    pub issuer: String,
    pub client_id: String,
    pub token_type: String,
    pub access_token: String,
    pub refresh_token: String,
    pub id_token: String,
    pub openai_api_key: Option<String>,
    pub api_key_exchange_error: Option<String>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub(crate) enum OpenAiOAuthOutputFormat {
    Json,
    Env,
}

pub(crate) fn pkce_from_verifier(code_verifier: impl Into<String>) -> PkceCodes {
    let code_verifier = code_verifier.into();
    let digest = Sha256::digest(code_verifier.as_bytes());
    let code_challenge = URL_SAFE_NO_PAD.encode(digest);
    PkceCodes {
        code_verifier,
        code_challenge,
    }
}

pub(crate) fn base64_url_no_pad(bytes: impl AsRef<[u8]>) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

pub(crate) fn decode_jwt_payload(token: &str) -> Result<Value, JwtPayloadError> {
    let mut parts = token.split('.');
    let Some(_header) = parts.next() else {
        return Err(JwtPayloadError::Malformed);
    };
    let Some(payload) = parts.next() else {
        return Err(JwtPayloadError::Malformed);
    };
    let Some(_signature) = parts.next() else {
        return Err(JwtPayloadError::Malformed);
    };
    if parts.next().is_some() {
        return Err(JwtPayloadError::Malformed);
    }

    let decoded = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|error| JwtPayloadError::Base64(error.to_string()))?;
    serde_json::from_slice(&decoded).map_err(|error| JwtPayloadError::Json(error.to_string()))
}

pub(crate) fn api_key_exchange_readiness(
    id_token: &str,
) -> Result<ApiKeyExchangeReadiness, JwtPayloadError> {
    let payload = decode_jwt_payload(id_token)?;
    if has_non_empty_string(payload.get("organization_id")) {
        return Ok(ApiKeyExchangeReadiness::Ready);
    }

    let auth_claim = payload.get(OPENAI_AUTH_CLAIM);
    if let Some(auth_claim) = auth_claim {
        if has_non_empty_string(auth_claim.get("organization_id")) {
            return Ok(ApiKeyExchangeReadiness::Ready);
        }
    }

    Ok(ApiKeyExchangeReadiness::MissingOrganizationId {
        default_organization_id: default_organization_id(auth_claim),
    })
}

fn has_non_empty_string(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
}

fn default_organization_id(auth_claim: Option<&Value>) -> Option<String> {
    let organizations = auth_claim?.get("organizations")?.as_array()?;
    organizations
        .iter()
        .find(|organization| {
            organization
                .get("is_default")
                .and_then(Value::as_bool)
                .unwrap_or(false)
        })
        .or_else(|| organizations.first())
        .and_then(|organization| organization.get("id"))
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .map(ToOwned::to_owned)
}

pub(crate) fn build_authorize_url(params: AuthorizeUrlParams<'_>) -> anyhow::Result<String> {
    let mut url = Url::parse(&format!(
        "{}/oauth/authorize",
        params.issuer.trim_end_matches('/')
    ))?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", params.client_id)
        .append_pair("redirect_uri", params.redirect_uri)
        .append_pair("scope", params.scope)
        .append_pair("code_challenge", &params.pkce.code_challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("id_token_add_organizations", "true")
        .append_pair("codex_cli_simplified_flow", "true")
        .append_pair("state", params.state)
        .append_pair("originator", params.originator);
    Ok(url.to_string())
}

pub(crate) fn parse_callback_target(
    target: &str,
    expected_state: &str,
) -> Result<ParsedCallback, CallbackError> {
    let url = Url::parse(&format!("http://localhost{target}"))
        .map_err(|_| CallbackError::InvalidRequestTarget)?;
    if url.path() != "/auth/callback" {
        return Ok(ParsedCallback::IgnoredPath);
    }

    let params: std::collections::HashMap<String, String> =
        url.query_pairs().into_owned().collect();
    let state_matches = params.get("state").map(String::as_str) == Some(expected_state);
    if !state_matches {
        return Err(CallbackError::StateMismatch);
    }

    if let Some(code) = params.get("error").filter(|code| !code.trim().is_empty()) {
        return Err(CallbackError::OAuth {
            code: code.to_owned(),
            description: params
                .get("error_description")
                .filter(|description| !description.trim().is_empty())
                .cloned(),
        });
    }

    match params.get("code").filter(|code| !code.trim().is_empty()) {
        Some(code) => Ok(ParsedCallback::Code(code.to_owned())),
        None => Err(CallbackError::MissingAuthorizationCode),
    }
}

impl CallbackError {
    pub(crate) fn http_status(&self) -> (u16, &'static str) {
        match self {
            Self::InvalidRequestTarget => (400, "Bad Request"),
            Self::StateMismatch => (400, "Bad Request"),
            Self::OAuth { .. } => (400, "Bad Request"),
            Self::MissingAuthorizationCode => (400, "Bad Request"),
        }
    }

    pub(crate) fn user_message(&self) -> String {
        match self {
            Self::InvalidRequestTarget => "invalid OAuth callback request".to_owned(),
            Self::StateMismatch => "OAuth callback state mismatch".to_owned(),
            Self::OAuth { code, description } => match description {
                Some(description) => format!("OAuth callback error {code}: {description}"),
                None => format!("OAuth callback error: {code}"),
            },
            Self::MissingAuthorizationCode => {
                "OAuth callback did not include an authorization code".to_owned()
            }
        }
    }
}

pub(crate) fn parse_token_endpoint_error(body: &str) -> TokenEndpointErrorDetail {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return TokenEndpointErrorDetail {
            error_code: None,
            error_message: None,
            display_message: "unknown error".to_owned(),
        };
    }

    let parsed = serde_json::from_str::<Value>(trimmed).ok();
    if let Some(json) = parsed {
        let error_code = json
            .get("error")
            .and_then(Value::as_str)
            .filter(|code| !code.trim().is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| {
                json.get("error")
                    .and_then(Value::as_object)
                    .and_then(|error| error.get("code"))
                    .and_then(Value::as_str)
                    .filter(|code| !code.trim().is_empty())
                    .map(ToOwned::to_owned)
            });

        if let Some(description) = json
            .get("error_description")
            .and_then(Value::as_str)
            .filter(|description| !description.trim().is_empty())
        {
            return TokenEndpointErrorDetail {
                error_code,
                error_message: Some(description.to_owned()),
                display_message: description.to_owned(),
            };
        }

        if let Some(message) = json
            .get("error")
            .and_then(Value::as_object)
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .filter(|message| !message.trim().is_empty())
        {
            return TokenEndpointErrorDetail {
                error_code,
                error_message: Some(message.to_owned()),
                display_message: message.to_owned(),
            };
        }

        if let Some(error_code) = error_code {
            return TokenEndpointErrorDetail {
                display_message: error_code.clone(),
                error_code: Some(error_code),
                error_message: None,
            };
        }
    }

    TokenEndpointErrorDetail {
        error_code: None,
        error_message: None,
        display_message: trimmed.to_owned(),
    }
}

pub(crate) fn render_oauth_output(
    output: &OpenAiOAuthOutput,
    format: OpenAiOAuthOutputFormat,
) -> anyhow::Result<String> {
    match format {
        OpenAiOAuthOutputFormat::Json => {
            let mut json =
                serde_json::to_string_pretty(output).context("failed to serialize OAuth tokens")?;
            json.push('\n');
            Ok(json)
        }
        OpenAiOAuthOutputFormat::Env => Ok(render_env_output(output)),
    }
}

fn render_env_output(output: &OpenAiOAuthOutput) -> String {
    let mut rendered = String::new();
    push_export(
        &mut rendered,
        "OPENAI_OAUTH_ACCESS_TOKEN",
        &output.access_token,
    );
    push_export(
        &mut rendered,
        "OPENAI_OAUTH_REFRESH_TOKEN",
        &output.refresh_token,
    );
    push_export(&mut rendered, "OPENAI_OAUTH_ID_TOKEN", &output.id_token);
    if let Some(openai_api_key) = &output.openai_api_key {
        push_export(&mut rendered, "OPENAI_API_KEY", openai_api_key);
    }
    rendered
}

fn push_export(rendered: &mut String, name: &str, value: &str) {
    rendered.push_str("export ");
    rendered.push_str(name);
    rendered.push('=');
    rendered.push_str(&shell_quote(value));
    rendered.push('\n');
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_matches_rfc7636_example() {
        let codes = pkce_from_verifier("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk");

        assert_eq!(
            codes.code_challenge,
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn decode_jwt_payload_covers_success_and_error_cases() {
        struct Case {
            name: &'static str,
            token: String,
            want_ok: bool,
        }
        let cases = [
            Case {
                name: "valid",
                token: test_jwt(serde_json::json!({"sub": "user"})),
                want_ok: true,
            },
            Case {
                name: "too_few_parts",
                token: "header.payload".to_owned(),
                want_ok: false,
            },
            Case {
                name: "too_many_parts",
                token: "a.b.c.d".to_owned(),
                want_ok: false,
            },
            Case {
                name: "invalid_base64",
                token: "header.*.signature".to_owned(),
                want_ok: false,
            },
            Case {
                name: "invalid_json",
                token: format!("header.{}.signature", base64_url_no_pad(b"not json")),
                want_ok: false,
            },
        ];

        for case in cases {
            assert_eq!(
                decode_jwt_payload(&case.token).is_ok(),
                case.want_ok,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn api_key_exchange_readiness_detects_platform_org_claims() {
        struct Case {
            name: &'static str,
            payload: Value,
            want: ApiKeyExchangeReadiness,
        }
        let cases = [
            Case {
                name: "top_level_organization_id",
                payload: serde_json::json!({"organization_id": "org-1"}),
                want: ApiKeyExchangeReadiness::Ready,
            },
            Case {
                name: "openai_auth_organization_id",
                payload: serde_json::json!({
                    "https://api.openai.com/auth": {
                        "organization_id": "org-1",
                        "project_id": "proj-1"
                    }
                }),
                want: ApiKeyExchangeReadiness::Ready,
            },
            Case {
                name: "chatgpt_organizations_default",
                payload: serde_json::json!({
                    "https://api.openai.com/auth": {
                        "organizations": [
                            {"id": "org-1", "is_default": false},
                            {"id": "org-2", "is_default": true}
                        ]
                    }
                }),
                want: ApiKeyExchangeReadiness::MissingOrganizationId {
                    default_organization_id: Some("org-2".to_owned()),
                },
            },
            Case {
                name: "chatgpt_organizations_first_fallback",
                payload: serde_json::json!({
                    "https://api.openai.com/auth": {
                        "organizations": [
                            {"id": "org-1"},
                            {"id": "org-2"}
                        ]
                    }
                }),
                want: ApiKeyExchangeReadiness::MissingOrganizationId {
                    default_organization_id: Some("org-1".to_owned()),
                },
            },
            Case {
                name: "no_organization_data",
                payload: serde_json::json!({"sub": "user"}),
                want: ApiKeyExchangeReadiness::MissingOrganizationId {
                    default_organization_id: None,
                },
            },
        ];

        for case in cases {
            let got = api_key_exchange_readiness(&test_jwt(case.payload)).expect("readiness");
            assert_eq!(got, case.want, "{}", case.name);
        }
    }

    #[test]
    fn api_key_exchange_readiness_rejects_malformed_id_token() {
        let error =
            api_key_exchange_readiness("not-a-jwt").expect_err("malformed id token should fail");

        assert_eq!(error, JwtPayloadError::Malformed);
    }

    #[test]
    fn build_authorize_url_includes_codex_oauth_parameters() {
        let pkce = PkceCodes {
            code_verifier: "verifier".to_owned(),
            code_challenge: "challenge".to_owned(),
        };
        let url = build_authorize_url(AuthorizeUrlParams {
            issuer: "https://auth.openai.com/",
            client_id: "client",
            redirect_uri: "http://localhost:1455/auth/callback",
            scope: "openid profile",
            pkce: &pkce,
            state: "state",
            originator: "codex_cli_rs",
        })
        .expect("authorize url");
        let parsed = Url::parse(&url).expect("valid url");
        let pairs: std::collections::HashMap<String, String> =
            parsed.query_pairs().into_owned().collect();

        assert_eq!(
            parsed.as_str().split('?').next(),
            Some("https://auth.openai.com/oauth/authorize")
        );
        assert_eq!(pairs.get("response_type").map(String::as_str), Some("code"));
        assert_eq!(pairs.get("client_id").map(String::as_str), Some("client"));
        assert_eq!(
            pairs.get("redirect_uri").map(String::as_str),
            Some("http://localhost:1455/auth/callback")
        );
        assert_eq!(
            pairs.get("scope").map(String::as_str),
            Some("openid profile")
        );
        assert_eq!(
            pairs.get("code_challenge").map(String::as_str),
            Some("challenge")
        );
        assert_eq!(
            pairs.get("code_challenge_method").map(String::as_str),
            Some("S256")
        );
        assert_eq!(
            pairs.get("id_token_add_organizations").map(String::as_str),
            Some("true")
        );
        assert_eq!(
            pairs.get("codex_cli_simplified_flow").map(String::as_str),
            Some("true")
        );
        assert_eq!(pairs.get("state").map(String::as_str), Some("state"));
        assert_eq!(
            pairs.get("originator").map(String::as_str),
            Some("codex_cli_rs")
        );
    }

    #[test]
    fn build_authorize_url_rejects_invalid_issuer() {
        let pkce = PkceCodes {
            code_verifier: "verifier".to_owned(),
            code_challenge: "challenge".to_owned(),
        };

        let error = build_authorize_url(AuthorizeUrlParams {
            issuer: "not a url",
            client_id: "client",
            redirect_uri: "http://localhost:1455/auth/callback",
            scope: "openid profile",
            pkce: &pkce,
            state: "state",
            originator: "codex_cli_rs",
        })
        .expect_err("invalid issuer should fail");

        assert!(error.to_string().contains("relative URL without a base"));
    }

    #[test]
    fn parse_callback_target_covers_success_and_error_cases() {
        struct Case {
            name: &'static str,
            target: &'static str,
            want: Result<ParsedCallback, CallbackError>,
        }
        let cases = [
            Case {
                name: "authorization_code",
                target: "/auth/callback?code=abc&state=expected",
                want: Ok(ParsedCallback::Code("abc".to_owned())),
            },
            Case {
                name: "ignored_path",
                target: "/favicon.ico",
                want: Ok(ParsedCallback::IgnoredPath),
            },
            Case {
                name: "state_mismatch",
                target: "/auth/callback?code=abc&state=wrong",
                want: Err(CallbackError::StateMismatch),
            },
            Case {
                name: "oauth_error_with_description",
                target: "/auth/callback?error=access_denied&error_description=nope&state=expected",
                want: Err(CallbackError::OAuth {
                    code: "access_denied".to_owned(),
                    description: Some("nope".to_owned()),
                }),
            },
            Case {
                name: "oauth_error_without_description",
                target: "/auth/callback?error=access_denied&state=expected",
                want: Err(CallbackError::OAuth {
                    code: "access_denied".to_owned(),
                    description: None,
                }),
            },
            Case {
                name: "missing_code",
                target: "/auth/callback?state=expected",
                want: Err(CallbackError::MissingAuthorizationCode),
            },
            Case {
                name: "invalid_target",
                target: "not a request target",
                want: Err(CallbackError::InvalidRequestTarget),
            },
        ];

        for case in cases {
            let got = parse_callback_target(case.target, "expected");
            assert_eq!(got, case.want, "{}", case.name);
        }
    }

    #[test]
    fn callback_error_user_message_covers_variants() {
        struct Case {
            error: CallbackError,
            want: &'static str,
        }
        let cases = [
            Case {
                error: CallbackError::InvalidRequestTarget,
                want: "invalid OAuth callback request",
            },
            Case {
                error: CallbackError::StateMismatch,
                want: "OAuth callback state mismatch",
            },
            Case {
                error: CallbackError::OAuth {
                    code: "access_denied".to_owned(),
                    description: Some("no access".to_owned()),
                },
                want: "OAuth callback error access_denied: no access",
            },
            Case {
                error: CallbackError::OAuth {
                    code: "access_denied".to_owned(),
                    description: None,
                },
                want: "OAuth callback error: access_denied",
            },
            Case {
                error: CallbackError::MissingAuthorizationCode,
                want: "OAuth callback did not include an authorization code",
            },
        ];

        for case in cases {
            assert_eq!(case.error.user_message(), case.want);
            assert_eq!(case.error.http_status(), (400, "Bad Request"));
        }
    }

    #[test]
    fn parse_token_endpoint_error_covers_response_shapes() {
        struct Case {
            name: &'static str,
            body: &'static str,
            want: TokenEndpointErrorDetail,
        }
        let cases = [
            Case {
                name: "empty",
                body: "",
                want: TokenEndpointErrorDetail {
                    error_code: None,
                    error_message: None,
                    display_message: "unknown error".to_owned(),
                },
            },
            Case {
                name: "description",
                body: r#"{"error":"invalid_grant","error_description":"expired"}"#,
                want: TokenEndpointErrorDetail {
                    error_code: Some("invalid_grant".to_owned()),
                    error_message: Some("expired".to_owned()),
                    display_message: "expired".to_owned(),
                },
            },
            Case {
                name: "nested_message",
                body: r#"{"error":{"code":"proxy_auth_required","message":"proxy required"}}"#,
                want: TokenEndpointErrorDetail {
                    error_code: Some("proxy_auth_required".to_owned()),
                    error_message: Some("proxy required".to_owned()),
                    display_message: "proxy required".to_owned(),
                },
            },
            Case {
                name: "code_only",
                body: r#"{"error":"temporarily_unavailable"}"#,
                want: TokenEndpointErrorDetail {
                    error_code: Some("temporarily_unavailable".to_owned()),
                    error_message: None,
                    display_message: "temporarily_unavailable".to_owned(),
                },
            },
            Case {
                name: "plain_text",
                body: "service unavailable",
                want: TokenEndpointErrorDetail {
                    error_code: None,
                    error_message: None,
                    display_message: "service unavailable".to_owned(),
                },
            },
        ];

        for case in cases {
            assert_eq!(
                parse_token_endpoint_error(case.body),
                case.want,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn render_oauth_output_supports_json_and_env() {
        let output = OpenAiOAuthOutput {
            issuer: DEFAULT_ISSUER.to_owned(),
            client_id: DEFAULT_CLIENT_ID.to_owned(),
            token_type: "Bearer".to_owned(),
            access_token: "access-token".to_owned(),
            refresh_token: "refresh-token".to_owned(),
            id_token: "id-token".to_owned(),
            openai_api_key: Some("api'key".to_owned()),
            api_key_exchange_error: None,
        };

        let json = render_oauth_output(&output, OpenAiOAuthOutputFormat::Json)
            .expect("json output should render");
        assert!(json.contains("\"access_token\": \"access-token\""));

        let env =
            render_oauth_output(&output, OpenAiOAuthOutputFormat::Env).expect("env should render");
        assert!(env.contains("export OPENAI_OAUTH_ACCESS_TOKEN='access-token'\n"));
        assert!(env.contains("export OPENAI_API_KEY='api'\"'\"'key'\n"));
    }

    #[test]
    fn render_env_output_omits_openai_api_key_when_exchange_was_skipped_or_failed() {
        let output = OpenAiOAuthOutput {
            issuer: DEFAULT_ISSUER.to_owned(),
            client_id: DEFAULT_CLIENT_ID.to_owned(),
            token_type: "Bearer".to_owned(),
            access_token: "access-token".to_owned(),
            refresh_token: "refresh-token".to_owned(),
            id_token: "id-token".to_owned(),
            openai_api_key: None,
            api_key_exchange_error: Some("api key exchange failed".to_owned()),
        };

        let env =
            render_oauth_output(&output, OpenAiOAuthOutputFormat::Env).expect("env should render");

        assert!(!env.contains("OPENAI_API_KEY"));
    }

    fn test_jwt(payload: Value) -> String {
        format!(
            "{}.{}.{}",
            base64_url_no_pad(r#"{"alg":"none"}"#),
            base64_url_no_pad(payload.to_string()),
            "signature"
        )
    }
}
