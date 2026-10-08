//! OTLP trace and metric export, behind the `otel` feature.
//!
//! [`OtelConfig::build`] builds a [`tracing_opentelemetry`] span layer and a
//! custom metrics layer ([`crate::telemetry::otel::metrics`], not public)
//! that derives OTel instruments from the span/event contract documented in
//! `README.md`. Both are plain `tracing_subscriber::Layer<Registry>`s,
//! already wrapped in their own per-layer [`EnvFilter`] (see
//! [`DEFAULT_FILTER_DIRECTIVES`] / [`OtelConfig::with_filter_directives`]) so
//! they see `info`-level halter spans *independently* of whatever filter
//! governs the rest of the subscriber stack.
//!
//! # The shared-filter pitfall
//!
//! [`crate::telemetry::TelemetryConfig::try_init_with`] composes its
//! formatter and the `extra` layer under *one shared* top-level
//! [`EnvFilter`] (defaulting to `warn`). Because `tracing-subscriber`'s
//! layer composition is a logical AND — the whole stack's `enabled()` must
//! agree before a span is even created — that shared filter gates the OTel
//! layers too, *regardless* of their own per-layer filter: a per-layer
//! filter can only make a layer *more* permissive than its siblings when
//! every level-filtering layer in the stack uses the per-layer filtering
//! API (`.with_filter(...)`), not when one of them is a bare top-level
//! layer like `try_init_with`'s shared filter. Concretely: calling
//! `TelemetryConfig::new().try_init_with(layers.combined())` with
//! `RUST_LOG` unset (so the shared filter defaults to `warn`) silently
//! exports **nothing**, because the `info_span!`-level `turn`/`tool_call`/
//! `provider_request`/`subagent` spans are never created in the first
//! place.
//!
//! Use [`crate::telemetry::TelemetryConfig::try_init_with_otel`] instead:
//! it composes the formatter and the OTel layers as siblings, each with its
//! *own* per-layer filter, so OTel sees `info`-level halter spans
//! regardless of the console's `RUST_LOG`. If you still want to use
//! `try_init_with` (for example because you are composing more than one
//! extra layer yourself), raise the shared filter's directives to include
//! at least [`DEFAULT_FILTER_DIRECTIVES`] (e.g.
//! `TelemetryConfig::new().with_directives(halter::telemetry::otel::DEFAULT_FILTER_DIRECTIVES)`),
//! or build your own manual per-layer-filter composition (see the
//! `telemetry` module docs' "Adding layers" section).
//!
//! # No implicit global state
//!
//! `OtelConfig::build` never calls
//! `opentelemetry::global::set_tracer_provider`,
//! `set_meter_provider`, or `set_text_map_propagator`. The `Tracer`/`Meter`
//! built from the two SDK providers are threaded directly into the returned
//! layers; nothing OTel-related becomes globally ambient. A global `tracing`
//! subscriber is only installed if the embedder goes on to call
//! [`crate::telemetry::TelemetryConfig::try_init_with_otel`] (or
//! `tracing::subscriber::set_global_default` themselves).
//!
//! ```rust,no_run
//! use halter::telemetry::TelemetryConfig;
//! use halter::telemetry::otel::OtelConfig;
//!
//! fn main() -> anyhow::Result<()> {
//!     let (layers, _guard) = OtelConfig::new().build()?;
//!     // Each of `layers.trace`/`layers.metrics` already carries its own
//!     // per-layer filter, independent of the console's `RUST_LOG` — this
//!     // exports `info`-level halter spans even when the console (and
//!     // `RUST_LOG`) stays at the default `warn`.
//!     TelemetryConfig::new().try_init_with_otel(layers)?;
//!     // ... run the application ...
//!     Ok(())
//! }
//! ```
//!
//! # Transport
//!
//! Exports OTLP over HTTP/protobuf using the workspace's existing `reqwest`
//! 0.12 client (see [`http_client`]), not `opentelemetry-otlp`'s bundled
//! `reqwest-client`/`grpc-tonic` features: those pull in `reqwest ^0.13` (a
//! second major version alongside the workspace's pinned `0.12`) or `tonic`
//! respectively. See the crate's PR description for the dependency-tree
//! evidence.
//!
//! # Env vars
//!
//! Honored automatically by the OTel SDK (not re-implemented here):
//! `OTEL_SERVICE_NAME`, `OTEL_RESOURCE_ATTRIBUTES`, `OTEL_TRACES_SAMPLER`,
//! `OTEL_TRACES_SAMPLER_ARG`, `OTEL_EXPORTER_OTLP_ENDPOINT` (and the
//! `_TRACES_`/`_METRICS_` signal-specific variants). Builder overrides
//! ([`OtelConfig::with_endpoint`], [`OtelConfig::with_service_name`],
//! [`OtelConfig::with_resource_attributes`], [`OtelConfig::with_sampler`])
//! take priority over all of these. The *span-visibility* filter
//! ([`DEFAULT_FILTER_DIRECTIVES`]) is separate from `RUST_LOG`; override it
//! with [`OtelConfig::with_filter_directives`] if you need a different
//! target/level set exported.

mod http_client;
mod metrics;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use opentelemetry::metrics::MeterProvider as _;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{BatchSpanProcessor, Sampler, SdkTracerProvider};
use tracing_subscriber::layer::Layer;
use tracing_subscriber::{EnvFilter, Registry};

use self::http_client::ReqwestOtlpClient;
use self::metrics::OtelMetricsLayer;

/// Default `service.name` when neither an explicit override nor
/// `OTEL_SERVICE_NAME`/`OTEL_RESOURCE_ATTRIBUTES` set one. The SDK's own
/// fallback (`unknown_service:<exe>`) is less useful than naming the crate.
const DEFAULT_SERVICE_NAME: &str = "halter";

/// Default bound on [`OtelGuard::shutdown`] / its `Drop` impl, per provider.
///
/// Honored by the tracer provider's shutdown. **Not** currently honored by
/// `opentelemetry_sdk` 0.33's `SdkMeterProvider::shutdown_with_timeout` for
/// the metrics side, which ignores its `timeout` argument and always uses
/// an internal hardcoded ~5s bound (`PeriodicReader`'s shutdown message has
/// a literal `// TODO: Make this timeout configurable.`). Shutdown still
/// cannot hang forever either way; a configured timeout shorter than 5s
/// just won't shorten the metrics-provider wait. See [`OtelGuard::shutdown`].
const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

/// Directives applied to both OTel layers' own per-layer [`EnvFilter`] by
/// default: `info`-level for the two targets the exported span contract
/// actually uses (`halter_runtime`: `turn`/`tool_call`/`subagent`;
/// `halter_providers`: `provider_request`/`provider_compaction`). This is
/// deliberately independent of `RUST_LOG`/[`crate::telemetry::compose_directives`]:
/// it exists so OTel keeps exporting even when the console stays at the
/// default `warn`. Override with [`OtelConfig::with_filter_directives`].
pub const DEFAULT_FILTER_DIRECTIVES: &str = "halter_runtime=info,halter_providers=info";

/// A boxed `Layer<Registry>`, used so [`OtelLayers`] can hand back
/// trait objects without naming `tracing_opentelemetry`'s or this crate's
/// internal layer types.
type BoxedLayer = Box<dyn Layer<Registry> + Send + Sync + 'static>;

/// Builder for the OTLP trace and metric pipelines.
///
/// Building does not install anything; call [`OtelConfig::build`] to get the
/// layers and a shutdown [`OtelGuard`], then install them with
/// [`crate::telemetry::TelemetryConfig::try_init_with_otel`] or manually.
#[derive(Debug, Default)]
pub struct OtelConfig {
    endpoint: Option<String>,
    service_name: Option<String>,
    resource_attributes: Vec<(String, String)>,
    sampler: Option<Sampler>,
    shutdown_timeout: Duration,
    filter_directives: Option<String>,
}

impl OtelConfig {
    /// A config with no overrides: endpoint, service name, resource
    /// attributes, and sampler are all resolved from `OTEL_*` env vars (or
    /// their documented defaults) in [`Self::build`].
    pub fn new() -> Self {
        Self {
            endpoint: None,
            service_name: None,
            resource_attributes: Vec::new(),
            sampler: None,
            shutdown_timeout: DEFAULT_SHUTDOWN_TIMEOUT,
            filter_directives: None,
        }
    }

    /// Override the OTLP base endpoint, e.g. `http://localhost:4318`.
    ///
    /// Takes priority over `OTEL_EXPORTER_OTLP_ENDPOINT` and the
    /// signal-specific `OTEL_EXPORTER_OTLP_{TRACES,METRICS}_ENDPOINT`.
    /// Treated as a *base* URL exactly like `OTEL_EXPORTER_OTLP_ENDPOINT`:
    /// `/v1/traces` and `/v1/metrics` are appended for you (unlike calling
    /// the underlying `opentelemetry-otlp` builder's own `.with_endpoint()`
    /// directly, which uses the value verbatim).
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Override `service.name`. Takes priority over `OTEL_SERVICE_NAME` and
    /// over a `service.name` entry in `OTEL_RESOURCE_ATTRIBUTES`.
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = Some(name.into());
        self
    }

    /// Add resource attributes in addition to `OTEL_RESOURCE_ATTRIBUTES`.
    /// These take priority over same-named keys from the environment.
    pub fn with_resource_attributes(
        mut self,
        attrs: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        self.resource_attributes.extend(attrs);
        self
    }

    /// Override the trace sampler. Takes priority over
    /// `OTEL_TRACES_SAMPLER`/`OTEL_TRACES_SAMPLER_ARG`. Re-exports the SDK's
    /// own [`Sampler`] type rather than wrapping it.
    pub fn with_sampler(mut self, sampler: Sampler) -> Self {
        self.sampler = Some(sampler);
        self
    }

    /// Bound how long [`OtelGuard::shutdown`] (and its `Drop` impl) may block
    /// flushing each provider. Default: 3 seconds. See [`DEFAULT_SHUTDOWN_TIMEOUT`]
    /// for a caveat: the SDK's metrics provider does not currently honor this.
    pub fn with_shutdown_timeout(mut self, timeout: Duration) -> Self {
        self.shutdown_timeout = timeout;
        self
    }

    /// Override the per-layer filter directives applied to both OTel
    /// layers (default: [`DEFAULT_FILTER_DIRECTIVES`]). Independent of
    /// `RUST_LOG` and of whatever filter governs the console formatter.
    pub fn with_filter_directives(mut self, directives: impl Into<String>) -> Self {
        self.filter_directives = Some(directives.into());
        self
    }

    /// Build the trace and metric pipelines.
    ///
    /// Returns the composable layers plus a guard that must be held (and
    /// ideally [`OtelGuard::shutdown`] called explicitly) for as long as
    /// telemetry should keep exporting; dropping it flushes and shuts down
    /// both providers, bounded by the configured shutdown timeout. Both
    /// returned layers already carry their own per-layer filter (see the
    /// module docs' "shared-filter pitfall" section) — compose them with
    /// [`crate::telemetry::TelemetryConfig::try_init_with_otel`].
    pub fn build(self) -> anyhow::Result<(OtelLayers, OtelGuard)> {
        let resource = self.build_resource();
        let http_client = Arc::new(ReqwestOtlpClient::default());
        let directives = self
            .filter_directives
            .as_deref()
            .unwrap_or(DEFAULT_FILTER_DIRECTIVES);
        let trace_filter =
            EnvFilter::try_new(directives).context("invalid otel filter directives")?;
        let metrics_filter = trace_filter.clone();

        let span_exporter = {
            let mut builder = SpanExporter::builder()
                .with_http()
                .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
                .with_shared_http_client(http_client.clone());
            if let Some(endpoint) = &self.endpoint {
                builder = builder.with_endpoint(signal_endpoint(endpoint, "/v1/traces"));
            }
            builder
                .build()
                .context("failed to build OTLP span exporter")?
        };
        let span_processor = BatchSpanProcessor::builder(span_exporter).build();
        let mut tracer_provider_builder = SdkTracerProvider::builder()
            .with_span_processor(span_processor)
            .with_resource(resource.clone());
        if let Some(sampler) = self.sampler.clone() {
            tracer_provider_builder = tracer_provider_builder.with_sampler(sampler);
        }
        let tracer_provider = tracer_provider_builder.build();
        let tracer = tracer_provider.tracer("halter");
        // `.with_filter(...)` makes this a per-layer-filtered layer: its
        // effective level is independent of whatever filter the rest of
        // the subscriber stack uses, which is exactly what avoids the
        // shared-filter pitfall described in the module docs.
        let trace_layer: BoxedLayer = tracing_opentelemetry::layer()
            .with_tracer(tracer)
            .with_filter(trace_filter)
            .boxed();

        let metric_exporter = {
            let mut builder = MetricExporter::builder()
                .with_http()
                .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
                .with_shared_http_client(http_client);
            if let Some(endpoint) = &self.endpoint {
                builder = builder.with_endpoint(signal_endpoint(endpoint, "/v1/metrics"));
            }
            builder
                .build()
                .context("failed to build OTLP metric exporter")?
        };
        let reader = PeriodicReader::builder(metric_exporter).build();
        let meter_provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_resource(resource)
            .build();
        let meter = meter_provider.meter("halter");
        let metrics_layer: BoxedLayer = OtelMetricsLayer::new(&meter)
            .with_filter(metrics_filter)
            .boxed();

        let layers = OtelLayers {
            trace: trace_layer,
            metrics: metrics_layer,
        };
        let guard = OtelGuard {
            tracer_provider,
            meter_provider,
            shutdown_timeout: self.shutdown_timeout,
        };
        Ok((layers, guard))
    }

    /// Resolve the `Resource` (service name + attributes). `Resource::builder()`
    /// already wires `SdkProvidedResourceDetector` (reads `OTEL_SERVICE_NAME`,
    /// defaults to `unknown_service:<exe>`) and `EnvResourceDetector` (reads
    /// `OTEL_RESOURCE_ATTRIBUTES`). An explicit `.with_service_name()` call
    /// always overrides both, because `ResourceBuilder::with_attribute`
    /// merges with the new value taking priority. With no explicit override
    /// and no env var, this falls back to [`DEFAULT_SERVICE_NAME`] instead of
    /// the SDK's `unknown_service:...` default.
    fn build_resource(&self) -> Resource {
        let mut builder = Resource::builder();
        let env_has_service_name = std::env::var_os("OTEL_SERVICE_NAME").is_some()
            || std::env::var("OTEL_RESOURCE_ATTRIBUTES")
                .map(|value| resource_attributes_contain_service_name(&value))
                .unwrap_or(false);
        match &self.service_name {
            Some(name) => builder = builder.with_service_name(name.clone()),
            None if !env_has_service_name => {
                builder = builder.with_service_name(DEFAULT_SERVICE_NAME);
            }
            None => {}
        }
        if !self.resource_attributes.is_empty() {
            builder = builder.with_attributes(
                self.resource_attributes
                    .iter()
                    .map(|(key, value)| opentelemetry::KeyValue::new(key.clone(), value.clone())),
            );
        }
        builder.build()
    }
}

fn resource_attributes_contain_service_name(value: &str) -> bool {
    value
        .split_terminator(',')
        .filter_map(|entry| entry.split_once('='))
        .any(|(key, _)| key.trim() == "service.name")
}

/// Append `path` to `base`, matching `OTEL_EXPORTER_OTLP_ENDPOINT`'s own
/// path-joining behavior (`opentelemetry-otlp`'s `build_endpoint_uri`).
fn signal_endpoint(base: &str, path: &str) -> String {
    if base.ends_with('/') {
        format!("{base}{}", path.trim_start_matches('/'))
    } else {
        format!("{base}{path}")
    }
}

/// The two layers built by [`OtelConfig::build`]: traces
/// ([`tracing_opentelemetry`]) and metrics (this module's custom layer).
/// Each already carries its own per-layer filter (see the module docs'
/// "shared-filter pitfall" section), independent of whatever filter the
/// rest of the subscriber stack uses. Install both with
/// [`crate::telemetry::TelemetryConfig::try_init_with_otel`], or use
/// [`Self::combined`] (or the individual fields) in your own manual
/// composition.
pub struct OtelLayers {
    /// `tracing-opentelemetry`'s span layer, wired to an OTLP span exporter
    /// and already filtered to [`DEFAULT_FILTER_DIRECTIVES`] (or the
    /// override passed to [`OtelConfig::with_filter_directives`]).
    pub trace: BoxedLayer,
    /// The custom metrics-deriving layer, wired to an OTLP metric exporter,
    /// with the same per-layer filter as [`Self::trace`].
    pub metrics: BoxedLayer,
}

impl OtelLayers {
    /// Compose both layers into one. Each retains its own per-layer filter,
    /// so the result behaves correctly whether it's passed to
    /// [`crate::telemetry::TelemetryConfig::try_init_with_otel`] or added
    /// to your own `tracing_subscriber::registry()` stack directly.
    pub fn combined(self) -> BoxedLayer {
        self.trace.and_then(self.metrics).boxed()
    }
}

/// Owns the OTel tracer and meter providers and bounds their shutdown.
///
/// Dropping the guard (or calling [`Self::shutdown`] explicitly, which is
/// recommended for deterministic flush ordering relative to process exit)
/// flushes and shuts down both providers. Shutdown errors are logged (via
/// `tracing::warn!`, a no-op if nothing is listening) rather than panicking.
/// Shutdown can never hang forever on either provider, but the configured
/// timeout is only honored precisely for the *tracer*: `opentelemetry_sdk`
/// 0.33's meter provider ignores its `timeout` argument and always uses an
/// internal hardcoded ~5s bound instead (see [`DEFAULT_SHUTDOWN_TIMEOUT`]).
pub struct OtelGuard {
    tracer_provider: SdkTracerProvider,
    meter_provider: SdkMeterProvider,
    shutdown_timeout: Duration,
}

impl OtelGuard {
    /// Flush and shut down both providers, bounded by the configured
    /// shutdown timeout. Prefer calling this explicitly over relying on
    /// `Drop` so the flush completes before dependent resources (e.g. output
    /// handles) are torn down.
    pub fn shutdown(self) -> anyhow::Result<()> {
        let trace_result = self
            .tracer_provider
            .shutdown_with_timeout(self.shutdown_timeout);
        let metrics_result = self
            .meter_provider
            .shutdown_with_timeout(self.shutdown_timeout);
        trace_result.context("failed to shut down OTel tracer provider")?;
        metrics_result.context("failed to shut down OTel meter provider")?;
        Ok(())
    }
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        if let Err(error) = self
            .tracer_provider
            .shutdown_with_timeout(self.shutdown_timeout)
        {
            tracing::warn!(%error, "failed to shut down OTel tracer provider");
        }
        if let Err(error) = self
            .meter_provider
            .shutdown_with_timeout(self.shutdown_timeout)
        {
            tracing::warn!(%error, "failed to shut down OTel meter provider");
        }
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn default_service_name_is_halter_without_override_or_env() {
        // `cargo test` runs tests in one process (unlike `cargo nextest`,
        // which gives each test its own process), so env-mutating tests here
        // must serialize on a mutex to avoid interfering with each other.
        let _guard = env_lock();
        clear_resource_env();
        let resource = OtelConfig::new().build_resource();
        assert_eq!(
            resource.get(&opentelemetry::Key::new("service.name")),
            Some(opentelemetry::Value::from(DEFAULT_SERVICE_NAME))
        );
    }

    #[test]
    fn explicit_service_name_overrides_env() {
        let _guard = env_lock();
        clear_resource_env();
        unsafe {
            std::env::set_var("OTEL_SERVICE_NAME", "from-env");
        }
        let resource = OtelConfig::new()
            .with_service_name("from-override")
            .build_resource();
        assert_eq!(
            resource.get(&opentelemetry::Key::new("service.name")),
            Some(opentelemetry::Value::from("from-override"))
        );
        clear_resource_env();
    }

    #[test]
    fn env_service_name_wins_over_halter_default() {
        let _guard = env_lock();
        clear_resource_env();
        unsafe {
            std::env::set_var("OTEL_SERVICE_NAME", "from-env");
        }
        let resource = OtelConfig::new().build_resource();
        assert_eq!(
            resource.get(&opentelemetry::Key::new("service.name")),
            Some(opentelemetry::Value::from("from-env"))
        );
        clear_resource_env();
    }

    #[test]
    fn resource_attributes_env_var_detects_service_name_key() {
        assert!(resource_attributes_contain_service_name(
            "deployment.environment=prod,service.name=svc"
        ));
        assert!(!resource_attributes_contain_service_name(
            "deployment.environment=prod"
        ));
    }

    #[test]
    fn signal_endpoint_joins_paths_like_otel_exporter_otlp_endpoint() {
        assert_eq!(
            signal_endpoint("http://localhost:4318", "/v1/traces"),
            "http://localhost:4318/v1/traces"
        );
        assert_eq!(
            signal_endpoint("http://localhost:4318/", "/v1/traces"),
            "http://localhost:4318/v1/traces"
        );
    }

    #[test]
    fn guard_shutdown_against_unroutable_endpoint_returns_within_bound() {
        let _guard = env_lock();
        clear_resource_env();
        let (layers, guard) = OtelConfig::new()
            .with_endpoint("http://127.0.0.1:1")
            .with_shutdown_timeout(Duration::from_millis(200))
            .build()
            .expect("build should succeed even against an unroutable endpoint");
        drop(layers);

        let start = std::time::Instant::now();
        let result = guard.shutdown();
        let elapsed = start.elapsed();
        // The point of this test is the bound below: shutdown against an
        // unroutable endpoint may legitimately report an export error, but
        // must never hang indefinitely trying to flush.
        let _ = result;
        assert!(
            elapsed < Duration::from_secs(2),
            "shutdown must be bounded by the configured timeout, took {elapsed:?}"
        );
    }

    /// Serializes tests that mutate process-wide `OTEL_*` env vars.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn clear_resource_env() {
        unsafe {
            std::env::remove_var("OTEL_SERVICE_NAME");
            std::env::remove_var("OTEL_RESOURCE_ATTRIBUTES");
        }
    }
}
