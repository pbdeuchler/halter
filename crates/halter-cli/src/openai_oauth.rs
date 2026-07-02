// pattern: Imperative Shell

use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use anyhow::Context;
use clap::Args;
use reqwest::Client;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::Command;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::openai_oauth_core::{
    ApiKeyExchangeReadiness, AuthorizeUrlParams, DEFAULT_CLIENT_ID, DEFAULT_FALLBACK_PORT,
    DEFAULT_ISSUER, DEFAULT_ORIGINATOR, DEFAULT_PORT, DEFAULT_SCOPE, DEFAULT_TIMEOUT_SECS,
    OpenAiOAuthOutput, OpenAiOAuthOutputFormat, ParsedCallback, PkceCodes,
    api_key_exchange_readiness, base64_url_no_pad, build_authorize_url, parse_callback_target,
    parse_token_endpoint_error, pkce_from_verifier, render_oauth_output,
};

#[derive(Debug, Clone, Args)]
pub(crate) struct OpenAiOAuthCommand {
    #[arg(long, default_value = DEFAULT_ISSUER, help = "OpenAI OAuth issuer base URL")]
    issuer: String,
    #[arg(
        long,
        default_value = DEFAULT_CLIENT_ID,
        help = "OAuth public client id"
    )]
    client_id: String,
    #[arg(
        long,
        default_value = DEFAULT_SCOPE,
        help = "Space-separated OAuth scopes"
    )]
    scope: String,
    #[arg(
        long,
        default_value = DEFAULT_ORIGINATOR,
        help = "Originator query parameter sent to the OAuth authorize endpoint"
    )]
    originator: String,
    #[arg(
        long,
        default_value_t = DEFAULT_PORT,
        help = "Local callback port. The default matches OpenAI Codex's registered redirect URI"
    )]
    port: u16,
    #[arg(
        long,
        default_value_t = DEFAULT_FALLBACK_PORT,
        help = "Fallback local callback port if --port is unavailable"
    )]
    fallback_port: u16,
    #[arg(
        long,
        default_value_t = DEFAULT_TIMEOUT_SECS,
        help = "Seconds to wait for the OAuth browser callback"
    )]
    timeout_secs: u64,
    #[arg(long, help = "Print the login URL without trying to open a browser")]
    no_open_browser: bool,
    #[arg(
        long,
        help = "Skip the best-effort id_token to OpenAI API-key token exchange"
    )]
    skip_api_key_exchange: bool,
    #[arg(
        long,
        conflicts_with = "skip_api_key_exchange",
        help = "Require the OpenAI API-key token exchange to succeed"
    )]
    require_api_key_exchange: bool,
    #[arg(long, value_enum, default_value_t = OpenAiOAuthOutputFormat::Json)]
    format: OpenAiOAuthOutputFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExchangedTokens {
    id_token: String,
    access_token: String,
    refresh_token: String,
}

#[derive(Debug, Deserialize)]
struct OAuthTokenResponse {
    id_token: String,
    access_token: String,
    refresh_token: String,
}

#[derive(Debug, Deserialize)]
struct ApiKeyExchangeResponse {
    access_token: String,
}

const MAX_CALLBACK_REQUEST_BYTES: usize = 16 * 1024;
const MAX_STRAY_CALLBACK_REQUESTS: usize = 16;
const CALLBACK_READ_DEADLINE: Duration = Duration::from_secs(5);

pub(crate) async fn run(command: OpenAiOAuthCommand, output: &mut dyn Write) -> anyhow::Result<()> {
    validate_command(&command)?;
    let pkce = generate_pkce();
    let state = generate_url_token();
    let listener = bind_callback_listener(command.port, command.fallback_port).await?;
    let actual_port = listener
        .local_addr()
        .context("failed to inspect OAuth callback listener address")?
        .port();
    let redirect_uri = format!("http://localhost:{actual_port}/auth/callback");
    let auth_url = build_authorize_url(AuthorizeUrlParams {
        issuer: &command.issuer,
        client_id: &command.client_id,
        redirect_uri: &redirect_uri,
        scope: &command.scope,
        pkce: &pkce,
        state: &state,
        originator: &command.originator,
    })?;

    if !command.no_open_browser
        && let Err(error) = open_browser(&auth_url).await
    {
        warn!(error = %error, "failed to open OAuth URL in browser");
        eprintln!("failed to open browser automatically: {error}");
    }

    eprintln!("Open this URL to authenticate with OpenAI:");
    eprintln!("{auth_url}");
    eprintln!("Waiting for OAuth callback on {redirect_uri}");

    let code = await_callback(listener, &state, Duration::from_secs(command.timeout_secs)).await?;

    let client = oauth_http_client()?;
    let tokens = exchange_code_for_tokens(
        &client,
        &command.issuer,
        &command.client_id,
        &redirect_uri,
        &pkce,
        &code,
    )
    .await
    .context("failed to exchange OAuth authorization code")?;

    let (openai_api_key, api_key_exchange_error) = maybe_exchange_id_token_for_api_key(
        &client,
        &command.issuer,
        &command.client_id,
        &tokens.id_token,
        command.skip_api_key_exchange,
        command.require_api_key_exchange,
    )
    .await?;

    let rendered = render_oauth_output(
        &OpenAiOAuthOutput {
            issuer: command.issuer,
            client_id: command.client_id,
            token_type: "Bearer".to_owned(),
            access_token: tokens.access_token,
            refresh_token: tokens.refresh_token,
            id_token: tokens.id_token,
            openai_api_key,
            api_key_exchange_error,
        },
        command.format,
    )?;
    output
        .write_all(rendered.as_bytes())
        .context("failed to write OAuth token output")
}

fn validate_command(command: &OpenAiOAuthCommand) -> anyhow::Result<()> {
    if command.timeout_secs == 0 {
        anyhow::bail!("invalid OAuth timeout: --timeout-secs must be greater than zero");
    }
    if command.client_id.trim().is_empty() {
        anyhow::bail!("invalid OAuth client id: --client-id must not be empty");
    }
    if command.scope.trim().is_empty() {
        anyhow::bail!("invalid OAuth scope: --scope must not be empty");
    }
    if command.originator.trim().is_empty() {
        anyhow::bail!("invalid OAuth originator: --originator must not be empty");
    }
    if command.skip_api_key_exchange && command.require_api_key_exchange {
        anyhow::bail!(
            "invalid OAuth options: --skip-api-key-exchange conflicts with --require-api-key-exchange"
        );
    }
    Ok(())
}

fn generate_pkce() -> PkceCodes {
    pkce_from_verifier(generate_url_token())
}

fn generate_url_token() -> String {
    let mut bytes = [0u8; 32];
    let (left, right) = bytes.split_at_mut(16);
    left.copy_from_slice(Uuid::new_v4().as_bytes());
    right.copy_from_slice(Uuid::new_v4().as_bytes());
    base64_url_no_pad(bytes)
}

fn oauth_http_client() -> anyhow::Result<Client> {
    Client::builder()
        .user_agent(concat!("halter/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("failed to build OAuth HTTP client")
}

async fn bind_callback_listener(port: u16, fallback_port: u16) -> anyhow::Result<TcpListener> {
    let primary = callback_addr(port);
    match TcpListener::bind(primary).await {
        Ok(listener) => {
            info!(port, "bound OAuth callback listener");
            return Ok(listener);
        }
        Err(error) if port != fallback_port => {
            warn!(
                port,
                fallback_port,
                error = %error,
                "OAuth callback port unavailable; trying fallback port"
            );
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to bind OAuth callback listener on {primary}"));
        }
    }

    let fallback = callback_addr(fallback_port);
    TcpListener::bind(fallback)
        .await
        .with_context(|| format!("failed to bind OAuth callback listener on {fallback}"))
}

fn callback_addr(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

async fn open_browser(url: &str) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    let mut command = {
        let mut command = Command::new("open");
        command.arg(url);
        command
    };

    #[cfg(target_os = "windows")]
    let mut command = {
        let mut command = Command::new("cmd");
        command.args(["/C", "start", "", url]);
        command
    };

    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    let mut command = {
        let mut command = Command::new("xdg-open");
        command.arg(url);
        command
    };

    let status = command
        .status()
        .await
        .context("failed to spawn browser opener")?;
    if !status.success() {
        anyhow::bail!("browser opener exited with status {status}");
    }
    Ok(())
}

async fn await_callback(
    listener: TcpListener,
    expected_state: &str,
    timeout: Duration,
) -> anyhow::Result<String> {
    tokio::time::timeout(timeout, async {
        let mut stray_count: usize = 0;
        loop {
            let (stream, peer_addr) = listener
                .accept()
                .await
                .context("failed to accept OAuth callback connection")?;
            debug!(%peer_addr, "accepted OAuth callback connection");
            match handle_callback_connection(stream, expected_state).await? {
                Some(code) => return Ok(code),
                None => {
                    stray_count += 1;
                    if stray_count >= MAX_STRAY_CALLBACK_REQUESTS {
                        anyhow::bail!(
                            "OAuth callback listener closed after {MAX_STRAY_CALLBACK_REQUESTS} stray requests without receiving the expected callback"
                        );
                    }
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| {
        Err(anyhow::anyhow!(
            "failed to receive OAuth callback within {} seconds",
            timeout.as_secs()
        ))
    })
}

async fn handle_callback_connection(
    mut stream: TcpStream,
    expected_state: &str,
) -> anyhow::Result<Option<String>> {
    let request = read_http_request(&mut stream).await?;
    let target = match request_target(&request) {
        Some(target) => target,
        None => {
            write_http_response(
                &mut stream,
                400,
                "Bad Request",
                "invalid OAuth callback request",
            )
            .await?;
            anyhow::bail!("invalid OAuth callback request line");
        }
    };

    match parse_callback_target(target, expected_state) {
        Ok(ParsedCallback::Code(code)) => {
            write_http_response(
                &mut stream,
                200,
                "OK",
                "Authentication complete. Return to your terminal.",
            )
            .await?;
            Ok(Some(code))
        }
        Ok(ParsedCallback::IgnoredPath) => {
            write_http_response(&mut stream, 404, "Not Found", "not found").await?;
            Ok(None)
        }
        Err(error) => {
            let (status, reason) = error.http_status();
            write_http_response(&mut stream, status, reason, &error.user_message()).await?;
            Err(anyhow::anyhow!(error.user_message()))
        }
    }
}

async fn read_http_request(stream: &mut TcpStream) -> anyhow::Result<Vec<u8>> {
    let mut request = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let n = tokio::time::timeout(CALLBACK_READ_DEADLINE, stream.read(&mut chunk))
            .await
            .context("OAuth callback read timed out")?
            .context("failed to read OAuth callback request")?;
        if n == 0 {
            anyhow::bail!("OAuth callback connection closed before headers were complete");
        }
        request.extend_from_slice(&chunk[..n]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            return Ok(request);
        }
        if request.len() > MAX_CALLBACK_REQUEST_BYTES {
            anyhow::bail!("OAuth callback request exceeded {MAX_CALLBACK_REQUEST_BYTES} bytes");
        }
    }
}

fn request_target(request: &[u8]) -> Option<&str> {
    let line_end = request.windows(2).position(|window| window == b"\r\n")?;
    let request_line = std::str::from_utf8(&request[..line_end]).ok()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?;
    let target = parts.next()?;
    let version = parts.next()?;
    (method == "GET" && version.starts_with("HTTP/")).then_some(target)
}

async fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    body: &str,
) -> anyhow::Result<()> {
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(response.as_bytes())
        .await
        .context("failed to write OAuth callback response")?;
    stream
        .shutdown()
        .await
        .context("failed to close OAuth callback response")
}

async fn exchange_code_for_tokens(
    client: &Client,
    issuer: &str,
    client_id: &str,
    redirect_uri: &str,
    pkce: &PkceCodes,
    code: &str,
) -> anyhow::Result<ExchangedTokens> {
    let token_endpoint = format!("{}/oauth/token", issuer.trim_end_matches('/'));
    let response = client
        .post(&token_endpoint)
        .form(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", client_id),
            ("code_verifier", &pkce.code_verifier),
        ])
        .send()
        .await
        .with_context(|| format!("failed to send OAuth token request to {token_endpoint}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .context("failed to read OAuth token error response")?;
        let detail = parse_token_endpoint_error(&body);
        anyhow::bail!("token endpoint returned status {status}: {detail}",);
    }

    let tokens = response
        .json::<OAuthTokenResponse>()
        .await
        .context("failed to decode OAuth token response")?;
    Ok(ExchangedTokens {
        id_token: tokens.id_token,
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token,
    })
}

async fn exchange_id_token_for_api_key(
    client: &Client,
    issuer: &str,
    client_id: &str,
    id_token: &str,
) -> anyhow::Result<String> {
    let token_endpoint = format!("{}/oauth/token", issuer.trim_end_matches('/'));
    let response = client
        .post(&token_endpoint)
        .form(&[
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:token-exchange",
            ),
            ("client_id", client_id),
            ("requested_token", "openai-api-key"),
            ("subject_token", id_token),
            (
                "subject_token_type",
                "urn:ietf:params:oauth:token-type:id_token",
            ),
        ])
        .send()
        .await
        .with_context(|| {
            format!("failed to send OpenAI API-key token request to {token_endpoint}")
        })?;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .context("failed to read OpenAI API-key token error response")?;
        let detail = parse_token_endpoint_error(&body);
        anyhow::bail!("API-key token endpoint returned status {status}: {detail}");
    }

    let token = response
        .json::<ApiKeyExchangeResponse>()
        .await
        .context("failed to decode OpenAI API-key token response")?;
    Ok(token.access_token)
}

async fn maybe_exchange_id_token_for_api_key(
    client: &Client,
    issuer: &str,
    client_id: &str,
    id_token: &str,
    skip_api_key_exchange: bool,
    require_api_key_exchange: bool,
) -> anyhow::Result<(Option<String>, Option<String>)> {
    if skip_api_key_exchange {
        return Ok((None, None));
    }

    match api_key_exchange_readiness(id_token) {
        Ok(ApiKeyExchangeReadiness::Ready) => {}
        Ok(ApiKeyExchangeReadiness::MissingOrganizationId { .. }) if require_api_key_exchange => {
            anyhow::bail!(
                "failed to exchange id_token for OpenAI API-key token: id_token does not include organization_id"
            );
        }
        Ok(ApiKeyExchangeReadiness::MissingOrganizationId { .. }) => {
            return Ok((
                None,
                Some(
                    "skipped: id_token does not include organization_id; API-key token exchange requires Platform organization claims"
                        .to_owned(),
                ),
            ));
        }
        Err(error) if require_api_key_exchange => {
            return Err(anyhow::anyhow!(error))
                .context("failed to inspect id_token before OpenAI API-key token exchange");
        }
        Err(error) => {
            return Ok((
                None,
                Some(format!(
                    "skipped: failed to inspect id_token before API-key token exchange: {error}"
                )),
            ));
        }
    }

    match exchange_id_token_for_api_key(client, issuer, client_id, id_token).await {
        Ok(api_key) => Ok((Some(api_key), None)),
        Err(error) if require_api_key_exchange => {
            Err(error).context("failed to exchange id_token for OpenAI API-key token")
        }
        Err(error) => {
            warn!(error = %error, "OpenAI API-key token exchange failed");
            Ok((None, Some(error.to_string())))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_command_accepts_defaults_and_rejects_invalid_options() {
        let valid = OpenAiOAuthCommand {
            issuer: DEFAULT_ISSUER.to_owned(),
            client_id: DEFAULT_CLIENT_ID.to_owned(),
            scope: DEFAULT_SCOPE.to_owned(),
            originator: DEFAULT_ORIGINATOR.to_owned(),
            port: DEFAULT_PORT,
            fallback_port: DEFAULT_FALLBACK_PORT,
            timeout_secs: DEFAULT_TIMEOUT_SECS,
            no_open_browser: true,
            skip_api_key_exchange: false,
            require_api_key_exchange: false,
            format: OpenAiOAuthOutputFormat::Json,
        };
        assert!(validate_command(&valid).is_ok());

        let cases = [
            (
                "timeout",
                OpenAiOAuthCommand {
                    timeout_secs: 0,
                    ..valid.clone()
                },
            ),
            (
                "client_id",
                OpenAiOAuthCommand {
                    client_id: " ".to_owned(),
                    ..valid.clone()
                },
            ),
            (
                "scope",
                OpenAiOAuthCommand {
                    scope: " ".to_owned(),
                    ..valid.clone()
                },
            ),
            (
                "originator",
                OpenAiOAuthCommand {
                    originator: " ".to_owned(),
                    ..valid.clone()
                },
            ),
            (
                "conflicting_api_key_exchange",
                OpenAiOAuthCommand {
                    skip_api_key_exchange: true,
                    require_api_key_exchange: true,
                    ..valid.clone()
                },
            ),
        ];

        for (name, command) in cases {
            assert!(validate_command(&command).is_err(), "{name}");
        }
    }

    #[test]
    fn generated_url_token_is_pkce_compatible() {
        let token = generate_url_token();

        assert_eq!(token.len(), 43);
        assert!(
            token
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
        );
        assert_ne!(generate_url_token(), token, "tokens should be unique");
    }

    #[tokio::test]
    async fn bind_callback_listener_uses_requested_port_when_available() {
        let listener = bind_callback_listener(0, DEFAULT_FALLBACK_PORT)
            .await
            .expect("listener");

        assert_ne!(
            listener.local_addr().expect("addr").port(),
            DEFAULT_FALLBACK_PORT
        );
    }

    #[tokio::test]
    async fn bind_callback_listener_uses_fallback_port_when_requested_port_is_busy() {
        let occupied = TcpListener::bind(callback_addr(0))
            .await
            .expect("occupied listener");
        let occupied_port = occupied.local_addr().expect("occupied addr").port();
        let fallback = TcpListener::bind(callback_addr(0))
            .await
            .expect("fallback holder");
        let fallback_port = fallback.local_addr().expect("fallback addr").port();
        drop(fallback);

        let listener = bind_callback_listener(occupied_port, fallback_port)
            .await
            .expect("fallback listener");

        assert_eq!(listener.local_addr().expect("addr").port(), fallback_port);
    }

    #[tokio::test]
    async fn handle_callback_connection_returns_code_and_success_response() {
        let listener = TcpListener::bind(callback_addr(0)).await.expect("listener");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            handle_callback_connection(stream, "state")
                .await
                .expect("callback")
        });

        let response = send_callback_request(
            addr,
            "GET /auth/callback?code=abc&state=state HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        let code = server.await.expect("server task");

        assert_eq!(code, Some("abc".to_owned()));
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("Authentication complete"));
    }

    #[tokio::test]
    async fn handle_callback_connection_rejects_state_mismatch() {
        let listener = TcpListener::bind(callback_addr(0)).await.expect("listener");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            handle_callback_connection(stream, "state").await
        });

        let response = send_callback_request(
            addr,
            "GET /auth/callback?code=abc&state=wrong HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        let result = server.await.expect("server task");

        assert!(result.is_err());
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(response.contains("OAuth callback state mismatch"));
    }

    #[tokio::test]
    async fn handle_callback_connection_ignores_unrelated_paths() {
        let listener = TcpListener::bind(callback_addr(0)).await.expect("listener");
        let addr = listener.local_addr().expect("addr");
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            handle_callback_connection(stream, "state")
                .await
                .expect("callback")
        });

        let response =
            send_callback_request(addr, "GET /favicon.ico HTTP/1.1\r\nHost: localhost\r\n\r\n")
                .await;
        let code = server.await.expect("server task");

        assert_eq!(code, None);
        assert!(response.starts_with("HTTP/1.1 404 Not Found"));
    }

    #[tokio::test]
    async fn await_callback_returns_code_after_strays_and_then_shuts_down_on_cap() {
        let listener = TcpListener::bind(callback_addr(0)).await.expect("listener");
        let addr = listener.local_addr().expect("addr");
        let state = "expected-state";
        let expected_code = "expected-code";

        let server = tokio::spawn(await_callback(listener, state, Duration::from_secs(60)));

        // Max allowed strays should be tolerated before the real callback.
        for _ in 0..MAX_STRAY_CALLBACK_REQUESTS - 1 {
            let response =
                send_callback_request(addr, "GET /favicon.ico HTTP/1.1\r\nHost: localhost\r\n\r\n")
                    .await;
            assert!(response.starts_with("HTTP/1.1 404 Not Found"));
        }

        let callback_response = send_callback_request(
            addr,
            &format!(
                "GET /auth/callback?code={expected_code}&state={state} HTTP/1.1\r\nHost: localhost\r\n\r\n"
            ),
        )
        .await;
        let code = server.await.expect("server task").expect("callback result");

        assert_eq!(code, expected_code);
        assert!(callback_response.starts_with("HTTP/1.1 200 OK"));
    }

    #[tokio::test]
    async fn await_callback_errors_after_stray_request_cap_exceeded() {
        let listener = TcpListener::bind(callback_addr(0)).await.expect("listener");
        let addr = listener.local_addr().expect("addr");
        let state = "expected-state";

        let server = tokio::spawn(await_callback(listener, state, Duration::from_secs(60)));

        // Send ignored paths until the budget is exhausted. The last connection attempt may be
        // refused because the listener is dropped as soon as the cap is exceeded.
        let mut accepted_count = 0;
        let mut refused_count = 0;
        for i in 0..=MAX_STRAY_CALLBACK_REQUESTS {
            match TcpStream::connect(addr).await {
                Ok(mut stream) => {
                    stream
                        .write_all(b"GET /favicon.ico HTTP/1.1\r\nHost: localhost\r\n\r\n")
                        .await
                        .expect("write request");
                    stream.shutdown().await.expect("shutdown write");
                    let mut response = Vec::new();
                    let _ = tokio::time::timeout(Duration::from_secs(3), async {
                        stream.read_to_end(&mut response).await.ok();
                    })
                    .await;
                    let response = String::from_utf8(response).expect("utf8 response");
                    if i < MAX_STRAY_CALLBACK_REQUESTS {
                        assert!(
                            response.starts_with("HTTP/1.1 404 Not Found"),
                            "stray {i} should be ignored"
                        );
                    }
                    accepted_count += 1;
                }
                Err(_) => {
                    refused_count += 1;
                }
            }
        }
        assert!(
            accepted_count >= MAX_STRAY_CALLBACK_REQUESTS,
            "at least {MAX_STRAY_CALLBACK_REQUESTS} strays should be accepted before the listener closes (accepted {accepted_count}, refused {refused_count})"
        );

        let result = tokio::time::timeout(Duration::from_secs(30), server)
            .await
            .expect("server task should finish promptly after cap exceeded")
            .expect("server join");
        let error = result.expect_err("exceeded stray cap should error");
        assert!(
            error
                .to_string()
                .contains("OAuth callback listener closed after")
        );
        assert!(
            error
                .to_string()
                .contains(&MAX_STRAY_CALLBACK_REQUESTS.to_string())
        );

        // The listener should no longer accept new connections once the server task finished.
        let start = tokio::time::Instant::now();
        let mut listener_closed = false;
        while start.elapsed() < Duration::from_millis(500) {
            if TcpStream::connect(addr).await.is_err() {
                listener_closed = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(
            listener_closed,
            "listener should be closed after stray cap exceeded"
        );
    }

    #[tokio::test]
    async fn read_http_request_times_out_on_slow_loris_client() {
        let listener = TcpListener::bind(callback_addr(0)).await.expect("listener");
        let addr = listener.local_addr().expect("addr");
        let state = "state";

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            handle_callback_connection(stream, state).await
        });

        // Connect and send only a partial request; do not close the write side.
        let connect = TcpStream::connect(addr).await.expect("connect");
        let (mut read, mut write) = connect.into_split();
        write.write_all(b"G").await.expect("write one byte");

        let result = tokio::time::timeout(CALLBACK_READ_DEADLINE + Duration::from_secs(2), server)
            .await
            .expect("server should hit read deadline")
            .expect("server task");
        let error = result.expect_err("slow client should time out");
        assert!(error.to_string().contains("OAuth callback read timed out"));

        // Cleanup the idle client connection so the test file descriptor is released promptly.
        drop(write);
        let _ = read.read_to_end(&mut Vec::new()).await;
    }

    #[tokio::test]
    async fn read_http_request_times_out_on_empty_connection() {
        let listener = TcpListener::bind(callback_addr(0)).await.expect("listener");
        let addr = listener.local_addr().expect("addr");
        let state = "state";

        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            handle_callback_connection(stream, state).await
        });

        // Connect and send nothing while keeping the socket open, so the server must enforce the
        // read deadline instead of observing EOF.
        let stream = TcpStream::connect(addr).await.expect("connect");

        let result = tokio::time::timeout(CALLBACK_READ_DEADLINE + Duration::from_secs(2), server)
            .await
            .expect("server should hit read deadline")
            .expect("server task");
        let error = result.expect_err("idle client should time out");
        assert!(error.to_string().contains("OAuth callback read timed out"));
        drop(stream);
    }

    #[tokio::test]
    async fn exchange_code_for_tokens_posts_authorization_code_form() {
        let (issuer, request) = spawn_one_response_server(
            200,
            r#"{"id_token":"id","access_token":"access","refresh_token":"refresh"}"#,
        )
        .await;
        let client = oauth_http_client().expect("client");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_owned(),
            code_challenge: "challenge".to_owned(),
        };

        let tokens = exchange_code_for_tokens(
            &client,
            &issuer,
            "client",
            "http://localhost:1455/auth/callback",
            &pkce,
            "code",
        )
        .await
        .expect("tokens");
        let request = request.await.expect("request task");

        assert_eq!(
            tokens,
            ExchangedTokens {
                id_token: "id".to_owned(),
                access_token: "access".to_owned(),
                refresh_token: "refresh".to_owned(),
            }
        );
        assert!(request.starts_with("POST /oauth/token HTTP/1.1"));
        assert!(request.contains("grant_type=authorization_code"));
        assert!(request.contains("code=code"));
        assert!(request.contains("client_id=client"));
        assert!(request.contains("code_verifier=verifier"));
    }

    #[tokio::test]
    async fn exchange_code_for_tokens_reports_endpoint_error() {
        let (issuer, _request) = spawn_one_response_server(
            400,
            r#"{"error":"invalid_grant","error_description":"expired"}"#,
        )
        .await;
        let client = oauth_http_client().expect("client");
        let pkce = PkceCodes {
            code_verifier: "verifier".to_owned(),
            code_challenge: "challenge".to_owned(),
        };

        let error = exchange_code_for_tokens(
            &client,
            &issuer,
            "client",
            "http://localhost:1455/auth/callback",
            &pkce,
            "code",
        )
        .await
        .expect_err("endpoint error should fail");

        assert!(error.to_string().contains("expired"));
    }

    #[tokio::test]
    async fn exchange_id_token_for_api_key_posts_token_exchange_form() {
        let (issuer, request) =
            spawn_one_response_server(200, r#"{"access_token":"api-key-token"}"#).await;
        let client = oauth_http_client().expect("client");

        let api_key = exchange_id_token_for_api_key(&client, &issuer, "client", "id-token")
            .await
            .expect("api key");
        let request = request.await.expect("request task");

        assert_eq!(api_key, "api-key-token");
        assert!(request.starts_with("POST /oauth/token HTTP/1.1"));
        assert!(
            request
                .contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Atoken-exchange")
        );
        assert!(request.contains("requested_token=openai-api-key"));
        assert!(request.contains("subject_token=id-token"));
    }

    #[tokio::test]
    async fn exchange_id_token_for_api_key_reports_endpoint_error() {
        let (issuer, _request) = spawn_one_response_server(
            403,
            r#"{"error":{"code":"forbidden","message":"no access"}}"#,
        )
        .await;
        let client = oauth_http_client().expect("client");

        let error = exchange_id_token_for_api_key(&client, &issuer, "client", "id-token")
            .await
            .expect_err("endpoint error should fail");

        assert!(error.to_string().contains("no access"));
    }

    #[tokio::test]
    async fn maybe_exchange_id_token_for_api_key_skips_chatgpt_tokens_without_org_claim() {
        let client = oauth_http_client().expect("client");
        let id_token = test_jwt(serde_json::json!({
            "https://api.openai.com/auth": {
                "organizations": [
                    {"id": "org-1", "is_default": true}
                ]
            }
        }));

        let (api_key, error) = maybe_exchange_id_token_for_api_key(
            &client,
            "http://127.0.0.1:9",
            "client",
            &id_token,
            false,
            false,
        )
        .await
        .expect("skip result");

        assert_eq!(api_key, None);
        assert!(
            error.as_deref().is_some_and(
                |message| message.contains("id_token does not include organization_id")
            )
        );
    }

    #[tokio::test]
    async fn maybe_exchange_id_token_for_api_key_requires_org_claim_when_required() {
        let client = oauth_http_client().expect("client");
        let id_token = test_jwt(serde_json::json!({
            "https://api.openai.com/auth": {
                "organizations": [
                    {"id": "org-1", "is_default": true}
                ]
            }
        }));

        let error = maybe_exchange_id_token_for_api_key(
            &client,
            "http://127.0.0.1:9",
            "client",
            &id_token,
            false,
            true,
        )
        .await
        .expect_err("missing organization_id should fail when required");

        assert!(error.to_string().contains("organization_id"));
    }

    #[tokio::test]
    async fn maybe_exchange_id_token_for_api_key_exchanges_platform_tokens() {
        let (issuer, request) =
            spawn_one_response_server(200, r#"{"access_token":"api-key-token"}"#).await;
        let client = oauth_http_client().expect("client");
        let id_token = test_jwt(serde_json::json!({
            "https://api.openai.com/auth": {
                "organization_id": "org-1",
                "project_id": "proj-1"
            }
        }));

        let (api_key, error) = maybe_exchange_id_token_for_api_key(
            &client, &issuer, "client", &id_token, false, false,
        )
        .await
        .expect("api-key exchange");
        let request = request.await.expect("request task");

        assert_eq!(api_key, Some("api-key-token".to_owned()));
        assert_eq!(error, None);
        assert!(request.contains("requested_token=openai-api-key"));
    }

    async fn send_callback_request(addr: SocketAddr, request: &str) -> String {
        let connect = TcpStream::connect(addr).await;
        let mut stream = match connect {
            Ok(stream) => stream,
            // Caller may intentionally probe after the listener has been dropped; surface the error.
            Err(error) => panic!("connect: {error}"),
        };
        stream
            .write_all(request.as_bytes())
            .await
            .expect("write request");
        stream.shutdown().await.expect("shutdown write");
        let mut response = Vec::new();
        // Guard against a server that never closes the connection with a small read timeout.
        let read_result = tokio::time::timeout(Duration::from_secs(3), async {
            stream
                .read_to_end(&mut response)
                .await
                .expect("read response");
        })
        .await;
        if read_result.is_err() {
            // The server may have dropped the listener; return what we have (possibly empty).
        }
        String::from_utf8(response).expect("utf8 response")
    }

    async fn spawn_one_response_server(
        status: u16,
        body: &'static str,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind(callback_addr(0))
            .await
            .expect("test server listener");
        let addr = listener.local_addr().expect("addr");
        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let request = read_full_http_request(&mut stream).await;
            let reason = if status == 200 { "OK" } else { "Bad Request" };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write response");
            request
        });
        (format!("http://{addr}"), handle)
    }

    async fn read_full_http_request(stream: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut chunk = [0u8; 1024];
        let mut content_length = None;
        loop {
            let n = stream.read(&mut chunk).await.expect("read");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..n]);
            if content_length.is_none()
                && let Some(headers_end) = find_headers_end(&request)
            {
                let headers = String::from_utf8_lossy(&request[..headers_end]);
                content_length = headers.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                });
            }
            if let (Some(headers_end), Some(content_length)) =
                (find_headers_end(&request), content_length)
                && request.len() >= headers_end + 4 + content_length
            {
                break;
            }
        }
        String::from_utf8(request).expect("utf8 request")
    }

    fn find_headers_end(request: &[u8]) -> Option<usize> {
        request.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn test_jwt(payload: serde_json::Value) -> String {
        format!(
            "{}.{}.{}",
            base64_url_no_pad(r#"{"alg":"none"}"#),
            base64_url_no_pad(payload.to_string()),
            "signature"
        )
    }
}
