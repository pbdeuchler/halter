//! Opt-in `tracing` subscriber setup (enabled with the `telemetry` feature).
//!
//! # Boundary
//!
//! Halter's library crates only *emit* `tracing` events and spans. Nothing in
//! halter installs a subscriber on its own: a global subscriber exists only if
//! you call [`TelemetryConfig::try_init`] or [`TelemetryConfig::try_init_with`]
//! (or install one yourself). The `halter` CLI is a thin caller of this module,
//! so embedders get the same defaults the CLI uses without copying code:
//!
//! - an [`EnvFilter`] read from `RUST_LOG`, defaulting to `warn`;
//! - [`NOISY_TARGET_SUPPRESSIONS`], which quiets per-token shell parser and
//!   HTTP connection-pool targets unless you name them explicitly;
//! - a compact or JSON [`fmt`] layer, selected with [`LogFormat`].
//!
//! ```rust,no_run
//! use halter::telemetry::{LogFormat, TelemetryConfig};
//!
//! fn main() -> anyhow::Result<()> {
//!     TelemetryConfig::new().with_format(LogFormat::Json).try_init()?;
//!     Ok(())
//! }
//! ```
//!
//! # Bringing your own subscriber
//!
//! You do not need this module. Halter's spans and events work with any
//! `tracing` subscriber; install yours with `tracing::subscriber::set_global_default`
//! (or a scoped default) and leave the `telemetry` feature off. You can still
//! reuse [`compose_directives`] to get halter's noisy-target suppression.
//!
//! # Adding layers
//!
//! [`TelemetryConfig::try_init_with`] installs one extra layer next to the
//! formatter. The composed [`EnvFilter`] filters the whole stack, so the extra
//! layer sees exactly what the formatter sees:
//!
//! ```rust,no_run
//! use halter::telemetry::{TelemetryConfig, tracing_subscriber};
//!
//! fn main() -> anyhow::Result<()> {
//!     // Any `Layer<Registry>` works here, e.g. an exporter layer.
//!     let extra = tracing_subscriber::fmt::layer().with_writer(std::io::stdout);
//!     TelemetryConfig::new().try_init_with(extra)?;
//!     Ok(())
//! }
//! ```
//!
//! When the extra layer needs a different filter (for example, an exporter
//! that wants `info` spans while the console stays at `warn`), compose the
//! pieces yourself with per-layer filters:
//!
//! ```rust,no_run
//! use halter::telemetry::TelemetryConfig;
//! use halter::telemetry::tracing_subscriber::{
//!     self, EnvFilter, Layer, layer::SubscriberExt,
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let config = TelemetryConfig::new();
//!     let extra = tracing_subscriber::fmt::layer()
//!         .with_writer(std::io::stdout)
//!         .with_filter(EnvFilter::try_new("halter_runtime=info,halter_providers=info")?);
//!     let subscriber = tracing_subscriber::registry()
//!         .with(extra)
//!         .with(config.fmt_layer().with_filter(config.env_filter()?));
//!     tracing::subscriber::set_global_default(subscriber)?;
//!     Ok(())
//! }
//! ```
//!
//! `tracing` diagnostics are unrelated to session transcript traces
//! (`traces_dir`, [`crate::session::export_session_trace`]); this module never
//! reads or writes transcripts.

use anyhow::Context;
use tracing_subscriber::{
    EnvFilter, Layer, Registry,
    fmt::{self, MakeWriter},
    layer::SubscriberExt,
    registry::LookupSpan,
};

/// Re-exported so embedders and the CLI name the same `tracing-subscriber`
/// version that this module's signatures use.
pub use tracing_subscriber;

/// Directives used when `RUST_LOG` is unset or blank (and no explicit
/// directives were configured).
pub const DEFAULT_DIRECTIVES: &str = "warn";

/// Third-party `tracing` targets that emit one DEBUG line per shell token,
/// HTTP connection, or pool event. Promoting them to WARN keeps the harness's
/// own DEBUG output readable when users opt into `RUST_LOG=debug`. Listed in
/// the directive string *before* the user's filter so an explicit per-target
/// user directive (e.g. `RUST_LOG=hyper=trace`) wins over the suppression
/// (per `EnvFilter` last-match-wins precedence), while the suppression still
/// overrides a global level like `debug` or `trace` for these noisy targets.
pub const NOISY_TARGET_SUPPRESSIONS: &str = "tokenize=warn,parse=warn,expansion=warn,commands=warn,pattern=warn,\
     completion=warn,jobs=warn,unimplemented=warn,\
     hyper_util=warn,hyper=warn,reqwest=warn,h2=warn,rustls=warn";

/// Output format of the formatting layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum LogFormat {
    /// Human-readable single-line output.
    #[default]
    Compact,
    /// One JSON object per line, including the current span context.
    Json,
}

/// Compose the `EnvFilter` directive string so that user directives come last.
///
/// Per `EnvFilter` precedence, the last matching directive wins; putting user
/// directives after [`NOISY_TARGET_SUPPRESSIONS`] lets explicit
/// `RUST_LOG=hyper=trace` overrides take effect while still quieting noisy
/// targets for `RUST_LOG=debug`. `None` or blank input falls back to
/// [`DEFAULT_DIRECTIVES`].
pub fn compose_directives(user_directives: Option<&str>) -> String {
    let user = user_directives.map(str::trim).unwrap_or("");
    if user.is_empty() {
        format!("{NOISY_TARGET_SUPPRESSIONS},{DEFAULT_DIRECTIVES}")
    } else {
        format!("{NOISY_TARGET_SUPPRESSIONS},{user}")
    }
}

/// Builder for halter's default subscriber stack.
///
/// Building filters and layers never installs anything; only
/// [`TelemetryConfig::try_init`] and [`TelemetryConfig::try_init_with`] set
/// the global default subscriber.
pub struct TelemetryConfig<W = fn() -> std::io::Stderr> {
    format: LogFormat,
    directives: Option<String>,
    writer: W,
}

impl<W> std::fmt::Debug for TelemetryConfig<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TelemetryConfig")
            .field("format", &self.format)
            .field("directives", &self.directives)
            .finish_non_exhaustive()
    }
}

impl TelemetryConfig {
    /// Compact output to stderr, with directives read from `RUST_LOG`.
    pub fn new() -> Self {
        Self {
            format: LogFormat::default(),
            directives: None,
            writer: std::io::stderr as fn() -> std::io::Stderr,
        }
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl<W> TelemetryConfig<W> {
    /// Select the output format.
    pub fn with_format(mut self, format: LogFormat) -> Self {
        self.format = format;
        self
    }

    /// Use these directives instead of reading `RUST_LOG`. They are still
    /// composed after [`NOISY_TARGET_SUPPRESSIONS`].
    pub fn with_directives(mut self, directives: impl Into<String>) -> Self {
        self.directives = Some(directives.into());
        self
    }

    /// Write formatted output to `writer` instead of stderr.
    pub fn with_writer<W2>(self, writer: W2) -> TelemetryConfig<W2> {
        TelemetryConfig {
            format: self.format,
            directives: self.directives,
            writer,
        }
    }

    /// The composed [`EnvFilter`]: suppressions followed by the explicit
    /// directives or `RUST_LOG`. Does not install anything.
    pub fn env_filter(&self) -> anyhow::Result<EnvFilter> {
        let user = match &self.directives {
            Some(directives) => Some(directives.clone()),
            None => user_directives_from_env(std::env::var(EnvFilter::DEFAULT_ENV))?,
        };
        let composed = compose_directives(user.as_deref());
        EnvFilter::try_new(&composed).context("invalid RUST_LOG filter")
    }
}

impl<W> TelemetryConfig<W>
where
    W: for<'a> MakeWriter<'a> + Clone + Send + Sync + 'static,
{
    /// The unfiltered formatting layer. Combine it with [`Self::env_filter`]
    /// (as a global layer or a per-layer filter). Does not install anything.
    pub fn fmt_layer<S>(&self) -> Box<dyn Layer<S> + Send + Sync + 'static>
    where
        S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    {
        let base = fmt::layer()
            .with_writer(self.writer.clone())
            .with_target(true);
        match self.format {
            LogFormat::Json => base.json().boxed(),
            LogFormat::Compact => base.compact().boxed(),
        }
    }

    /// Install the default stack (filter + formatter) as the global default
    /// subscriber. Fails if a global subscriber is already set.
    pub fn try_init(self) -> anyhow::Result<()> {
        self.try_init_with(tracing_subscriber::layer::Identity::new())
    }

    /// Install the default stack plus `extra` as the global default
    /// subscriber. The composed [`EnvFilter`] acts as a global filter for the
    /// whole stack, including `extra`; compose manually with per-layer
    /// filters (see the module docs) if `extra` needs its own filter.
    pub fn try_init_with<L>(self, extra: L) -> anyhow::Result<()>
    where
        L: Layer<Registry> + Send + Sync + 'static,
    {
        let filter = self.env_filter()?;
        let stack = tracing_subscriber::registry().with(extra).with(filter);
        let subscriber = stack.with(self.fmt_layer());
        // Deliberately not `SubscriberInitExt::try_init`: with
        // tracing-subscriber's default `tracing-log` feature it would also
        // install a `log` -> `tracing` bridge (`LogTracer`), which changes
        // which third-party `log` records reach the output.
        tracing::subscriber::set_global_default(subscriber).context("failed to initialize logging")
    }
}

fn user_directives_from_env(
    value: Result<String, std::env::VarError>,
) -> anyhow::Result<Option<String>> {
    match value {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!("invalid utf-8 in RUST_LOG"),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::io;
    use std::sync::{Arc, Mutex};

    use super::*;

    macro_rules! assert_target_level {
        ($composed:expr, $target:literal, $level:path, $expected:expr) => {{
            let filter = EnvFilter::try_new(&$composed).expect("valid filter");
            let subscriber = tracing_subscriber::registry().with(filter);
            let mut enabled = false;
            tracing::subscriber::with_default(subscriber, || {
                enabled = tracing::enabled!(target: $target, $level);
            });
            assert_eq!(
                enabled, $expected,
                "expected target={} level={:?} to be {}",
                $target, $level, $expected
            );
        }};
    }

    #[test]
    fn compose_filter_uses_warn_fallback_when_unset() {
        let composed = compose_directives(None);
        assert!(composed.starts_with(NOISY_TARGET_SUPPRESSIONS));
        assert!(composed.ends_with(",warn"));
        EnvFilter::try_new(&composed).expect("valid filter");
    }

    #[test]
    fn compose_filter_parses_whitespace_only_rust_log() {
        let composed = compose_directives(Some("   "));
        assert_eq!(composed, compose_directives(None));
        EnvFilter::try_new(&composed).expect("valid filter");
    }

    #[test]
    fn compose_filter_honors_global_level() {
        let composed = compose_directives(Some("info"));
        EnvFilter::try_new(&composed).expect("valid filter");
        assert_target_level!(composed, "halter", tracing::Level::INFO, true);
        assert_target_level!(composed, "halter", tracing::Level::DEBUG, false);
        assert_target_level!(composed, "tokenize", tracing::Level::DEBUG, false);
        assert_target_level!(composed, "hyper", tracing::Level::DEBUG, false);
    }

    #[test]
    fn compose_filter_honors_multi_directive_user_filter() {
        let composed = compose_directives(Some("debug,halter=trace"));
        EnvFilter::try_new(&composed).expect("valid filter");
        assert_target_level!(composed, "halter", tracing::Level::TRACE, true);
        assert_target_level!(composed, "some_crate", tracing::Level::DEBUG, true);
        assert_target_level!(composed, "tokenize", tracing::Level::DEBUG, false);
    }

    #[test]
    fn regression_99_explicit_per_target_wins_over_suppression() {
        let composed = compose_directives(Some("hyper=trace"));
        assert_target_level!(composed, "hyper", tracing::Level::TRACE, true);
    }

    #[test]
    fn regression_99_explicit_target_overrides_global_warn_and_suppression() {
        let composed = compose_directives(Some("warn,hyper=trace"));
        assert_target_level!(composed, "some_other_crate", tracing::Level::INFO, false);
        assert_target_level!(composed, "hyper", tracing::Level::TRACE, true);
    }

    #[test]
    fn regression_99_multiple_suppressed_targets_can_be_overridden() {
        let composed = compose_directives(Some("reqwest=debug,h2=info"));
        assert_target_level!(composed, "reqwest", tracing::Level::DEBUG, true);
        assert_target_level!(composed, "h2", tracing::Level::INFO, true);
        assert_target_level!(composed, "hyper", tracing::Level::DEBUG, false);
    }

    #[test]
    fn compose_filter_global_debug_still_suppresses_noisy_targets() {
        let composed = compose_directives(Some("debug"));
        assert_target_level!(composed, "halter", tracing::Level::DEBUG, true);
        assert_target_level!(composed, "tokenize", tracing::Level::DEBUG, false);
        assert_target_level!(composed, "hyper", tracing::Level::DEBUG, false);
        assert_target_level!(composed, "reqwest", tracing::Level::DEBUG, false);
    }

    #[test]
    fn compose_filter_rejects_invalid_user_directive() {
        let composed = compose_directives(Some("hyper=invalid_level"));
        EnvFilter::try_new(&composed).expect_err("invalid user directive should fail");
    }

    #[test]
    fn user_directives_from_env_maps_var_results() {
        assert_eq!(
            user_directives_from_env(Err(std::env::VarError::NotPresent)).expect("not present"),
            None
        );
        let error = user_directives_from_env(Err(std::env::VarError::NotUnicode(OsString::new())))
            .expect_err("non-unicode must fail");
        assert!(error.to_string().contains("invalid utf-8"), "{error}");
        assert_eq!(
            user_directives_from_env(Ok(" debug ".to_owned())).expect("present"),
            Some(" debug ".to_owned())
        );
    }

    #[test]
    fn explicit_invalid_directives_fail_with_context() {
        let error = TelemetryConfig::new()
            .with_directives("hyper=invalid_level")
            .env_filter()
            .expect_err("invalid directive");
        assert!(error.to_string().contains("invalid RUST_LOG filter"));
    }

    #[derive(Clone, Default)]
    struct BufferWriter(Arc<Mutex<Vec<u8>>>);

    impl BufferWriter {
        fn contents(&self) -> String {
            String::from_utf8(self.0.lock().expect("lock").clone()).expect("utf-8")
        }
    }

    impl io::Write for BufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().expect("lock").extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for BufferWriter {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    fn emit_with(format: LogFormat) -> String {
        let buffer = BufferWriter::default();
        let config = TelemetryConfig::new()
            .with_format(format)
            .with_directives("info")
            .with_writer(buffer.clone());
        let subscriber = tracing_subscriber::registry()
            .with(config.env_filter().expect("filter"))
            .with(config.fmt_layer());
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(answer = 42, "hello");
            tracing::debug!("filtered out");
        });
        buffer.contents()
    }

    #[test]
    fn compact_layer_writes_structured_fields() {
        let output = emit_with(LogFormat::Compact);
        assert!(output.contains("hello"), "{output}");
        assert!(output.contains("answer"), "{output}");
        assert!(!output.contains("filtered out"), "{output}");
    }

    #[test]
    fn json_layer_writes_one_object_per_line() {
        let output = emit_with(LogFormat::Json);
        assert!(output.contains("hello"), "{output}");
        assert!(output.contains("answer"), "{output}");
        let lines: Vec<_> = output.lines().collect();
        assert_eq!(lines.len(), 1, "{output}");
        for line in lines {
            let value: serde_json::Value = serde_json::from_str(line).expect("json line");
            assert_eq!(value["fields"]["answer"], 42);
        }
    }
}
