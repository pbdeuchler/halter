//! OTLP trace and metric export, behind the `otel` feature.
//!
//! [`OtelConfig::build`] builds a [`tracing_opentelemetry`] span layer and a
//! custom metrics layer ([`crate::telemetry::otel::metrics`], not public)
//! that derives OTel instruments from the span/event contract documented in
//! `README.md`. Both are plain `tracing_subscriber::Layer<Registry>`s, so
//! they compose with [`crate::telemetry::TelemetryConfig::try_init_with`] or
//! manual composition exactly like any other extra layer.
//!
//! # No implicit global state
//!
//! `OtelConfig::build` never calls
//! `opentelemetry::global::set_tracer_provider`,
//! `set_meter_provider`, or `set_text_map_propagator`. The `Tracer`/`Meter`
//! built from the two SDK providers are threaded directly into the returned
//! layers; nothing OTel-related becomes globally ambient. A global `tracing`
//! subscriber is only installed if the embedder goes on to call
//! [`crate::telemetry::TelemetryConfig::try_init_with`] (or
//! `tracing::subscriber::set_global_default` themselves).
//!
//! ```rust,no_run
//! use halter::telemetry::TelemetryConfig;
//! use halter::telemetry::otel::OtelConfig;
//!
//! fn main() -> anyhow::Result<()> {
//!     let (layers, _guard) = OtelConfig::new().build()?;
//!     TelemetryConfig::new().try_init_with(layers.combined())?;
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
//! take priority over all of these.

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
use tracing_subscriber::Registry;
use tracing_subscriber::layer::Layer;

use self::http_client::ReqwestOtlpClient;
use self::metrics::OtelMetricsLayer;

/// Default `service.name` when neither an explicit override nor
/// `OTEL_SERVICE_NAME`/`OTEL_RESOURCE_ATTRIBUTES` set one. The SDK's own
/// fallback (`unknown_service:<exe>`) is less useful than naming the crate.
const DEFAULT_SERVICE_NAME: &str = "halter";

/// Default bound on [`OtelGuard::shutdown`] / its `Drop` impl, per provider.
const DEFAULT_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);

/// A boxed `Layer<Registry>`, used so [`OtelLayers`] can hand back
/// trait objects without naming `tracing_opentelemetry`'s or this crate's
/// internal layer types.
type BoxedLayer = Box<dyn Layer<Registry> + Send + Sync + 'static>;

/// Builder for the OTLP trace and metric pipelines.
///
/// Building does not install anything; call [`OtelConfig::build`] to get the
/// layers and a shutdown [`OtelGuard`], then compose the layers with
/// [`crate::telemetry::TelemetryConfig::try_init_with`] or manually.
#[derive(Debug, Default)]
pub struct OtelConfig {
    endpoint: Option<String>,
    service_name: Option<String>,
    resource_attributes: Vec<(String, String)>,
    sampler: Option<Sampler>,
    shutdown_timeout: Duration,
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
    /// flushing each provider. Default: 3 seconds.
    pub fn with_shutdown_timeout(mut self, timeout: Duration) -> Self {
        self.shutdown_timeout = timeout;
        self
    }

    /// Build the trace and metric pipelines.
    ///
    /// Returns the composable layers plus a guard that must be held (and
    /// ideally [`OtelGuard::shutdown`] called explicitly) for as long as
    /// telemetry should keep exporting; dropping it flushes and shuts down
    /// both providers, bounded by the configured shutdown timeout.
    pub fn build(self) -> anyhow::Result<(OtelLayers, OtelGuard)> {
        let resource = self.build_resource();
        let http_client = Arc::new(ReqwestOtlpClient::default());

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
        let trace_layer: BoxedLayer = tracing_opentelemetry::layer().with_tracer(tracer).boxed();

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
        let metrics_layer: BoxedLayer = OtelMetricsLayer::new(&meter).boxed();

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
/// Both are plain `Layer<Registry>`s; use [`Self::combined`] to pass both to
/// a single [`crate::telemetry::TelemetryConfig::try_init_with`] call, or
/// `.with_filter(...)` each individually before composing them yourself.
pub struct OtelLayers {
    /// `tracing-opentelemetry`'s span layer, wired to an OTLP span exporter.
    pub trace: BoxedLayer,
    /// The custom metrics-deriving layer, wired to an OTLP metric exporter.
    pub metrics: BoxedLayer,
}

impl OtelLayers {
    /// Compose both layers into one, for a single
    /// [`crate::telemetry::TelemetryConfig::try_init_with`] call. Both
    /// layers still see every event; use the individual fields with their
    /// own `.with_filter(...)` if they need different filters.
    pub fn combined(self) -> BoxedLayer {
        self.trace.and_then(self.metrics).boxed()
    }
}

/// Owns the OTel tracer and meter providers and bounds their shutdown.
///
/// Dropping the guard (or calling [`Self::shutdown`] explicitly, which is
/// recommended for deterministic flush ordering relative to process exit)
/// flushes and shuts down both providers. Shutdown errors are logged (via
/// `tracing::warn!`, a no-op if nothing is listening) rather than panicking,
/// and are bounded by the configured shutdown timeout so a stuck exporter
/// can never hang the process.
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
