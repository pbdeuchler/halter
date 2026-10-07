//! End-to-end tests against the OTel SDK's in-memory exporters: real
//! `tracing_opentelemetry` span layer + the real [`super::metrics::OtelMetricsLayer`],
//! wired to `InMemorySpanExporter`/`InMemoryMetricExporter` instead of a real
//! OTLP/HTTP exporter. No global subscriber is installed anywhere here:
//! every test uses `tracing::subscriber::with_default`.

use std::collections::BTreeSet;

use opentelemetry::metrics::MeterProvider as _;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::metrics::data::AggregatedMetrics;
use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};
use opentelemetry_sdk::trace::{
    BatchSpanProcessor, InMemorySpanExporter, Sampler, SdkTracerProvider, SpanData,
};
use tracing::field::Empty;
use tracing_subscriber::layer::SubscriberExt;

use super::metrics::OtelMetricsLayer;

/// Attribute keys that must never appear on an exported *metric*. They are
/// expected (and fine) on spans.
const FORBIDDEN_METRIC_ATTRIBUTES: &[&str] = &["session_id", "turn_id", "tool_call_id", "agent_id"];

/// `ProviderErrorKind`'s real variants, mirrored here so the test exercises
/// the same `Debug` output (`"RateLimited"`, `"Transient"`, ...) that
/// `halter-providers` actually produces, without a cross-crate test
/// dependency.
#[derive(Debug, Clone, Copy)]
enum FakeErrorKind {
    Transient,
    RateLimited,
}

struct Harness {
    tracer_provider: SdkTracerProvider,
    meter_provider: SdkMeterProvider,
    span_exporter: InMemorySpanExporter,
    metric_exporter: InMemoryMetricExporter,
}

impl Harness {
    fn new() -> Self {
        let span_exporter = InMemorySpanExporter::default();
        let span_processor = BatchSpanProcessor::builder(span_exporter.clone()).build();
        let tracer_provider = SdkTracerProvider::builder()
            .with_span_processor(span_processor)
            // Deterministic: every span is sampled regardless of parent.
            .with_sampler(Sampler::AlwaysOn)
            .build();

        let metric_exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(metric_exporter.clone()).build();
        let meter_provider = SdkMeterProvider::builder().with_reader(reader).build();

        Self {
            tracer_provider,
            meter_provider,
            span_exporter,
            metric_exporter,
        }
    }

    fn run<R>(&self, f: impl FnOnce() -> R) -> R {
        let tracer = self.tracer_provider.tracer("halter-test");
        let meter = self.meter_provider.meter("halter-test");
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(tracer))
            .with(OtelMetricsLayer::new(&meter));
        tracing::subscriber::with_default(subscriber, f)
    }

    fn flush(&self) {
        self.tracer_provider.force_flush().expect("flush traces");
        self.meter_provider.force_flush().expect("flush metrics");
    }

    fn finished_spans(&self) -> Vec<SpanData> {
        self.span_exporter.get_finished_spans().expect("spans")
    }

    fn finished_metrics(&self) -> Vec<opentelemetry_sdk::metrics::data::ResourceMetrics> {
        self.metric_exporter
            .get_finished_metrics()
            .expect("metrics")
    }
}

fn span_named<'a>(spans: &'a [SpanData], name: &str) -> &'a SpanData {
    spans
        .iter()
        .find(|span| span.name == name)
        .unwrap_or_else(|| panic!("no exported span named {name}"))
}

fn attr<'a>(span: &'a SpanData, key: &str) -> Option<&'a opentelemetry::Value> {
    span.attributes
        .iter()
        .find(|kv| kv.key.as_str() == key)
        .map(|kv| &kv.value)
}

fn metric_named<'a>(
    metrics: &'a [opentelemetry_sdk::metrics::data::ResourceMetrics],
    name: &str,
) -> &'a opentelemetry_sdk::metrics::data::Metric {
    metrics
        .iter()
        .flat_map(|rm| rm.scope_metrics())
        .flat_map(|sm| sm.metrics())
        .find(|metric| metric.name() == name)
        .unwrap_or_else(|| panic!("no exported metric named {name}"))
}

/// Emits a `turn` span with a nested `tool_call`, matching the field shapes
/// `halter-runtime` actually uses (see `session.rs`).
fn emit_turn_and_tool_call() {
    let turn_span = tracing::info_span!(
        target: "halter_runtime",
        "turn",
        session_id = "sess-1",
        turn_id = "turn-1",
        provider = Empty,
        model = Empty,
        model_id = Empty,
        provider_iterations = Empty,
        input_tokens = Empty,
        output_tokens = Empty,
        cache_read_input_tokens = Empty,
        cache_creation_input_tokens = Empty,
        outcome = Empty,
    );
    let _turn_enter = turn_span.enter();
    turn_span.record("provider", "anthropic");
    turn_span.record("model", "claude-sonnet");
    turn_span.record("model_id", "claude-sonnet-x");
    turn_span.record("input_tokens", 100u64);
    turn_span.record("output_tokens", 40u64);
    turn_span.record("cache_read_input_tokens", 0u64);

    {
        let tool_span = tracing::info_span!(
            target: "halter_runtime",
            "tool_call",
            session_id = "sess-1",
            turn_id = "turn-1",
            tool_name = "grep",
            tool_call_id = "call-1",
            outcome = Empty,
            is_error = Empty,
        );
        let _tool_enter = tool_span.enter();
        tool_span.record("outcome", "ok");
        tool_span.record("is_error", false);
    }

    turn_span.record("outcome", "completed");
}

#[test]
fn turn_and_tool_call_spans_export_with_expected_attributes_and_nesting() {
    let harness = Harness::new();
    harness.run(emit_turn_and_tool_call);
    harness.flush();

    let spans = harness.finished_spans();
    let turn = span_named(&spans, "turn");
    let tool_call = span_named(&spans, "tool_call");

    assert_eq!(
        attr(turn, "session_id"),
        Some(&opentelemetry::Value::from("sess-1"))
    );
    assert_eq!(
        attr(turn, "outcome"),
        Some(&opentelemetry::Value::from("completed"))
    );
    // `tracing-opentelemetry`'s span attribute visitor has no `record_u64`
    // override (OTel span attribute `Value` has no native u64 variant), so
    // `u64` fields fall back to `record_debug` and are exported as strings.
    // This only affects the span attribute, not the metrics layer's own
    // field visitor (see `token_usage_counter_...` below), which parses the
    // numeric fields itself.
    assert_eq!(
        attr(turn, "input_tokens"),
        Some(&opentelemetry::Value::from("100"))
    );

    assert_eq!(
        attr(tool_call, "tool_name"),
        Some(&opentelemetry::Value::from("grep"))
    );
    assert_eq!(
        attr(tool_call, "outcome"),
        Some(&opentelemetry::Value::from("ok"))
    );

    assert_eq!(
        tool_call.parent_span_id,
        turn.span_context.span_id(),
        "tool_call must nest under turn"
    );
}

#[test]
fn subagent_span_is_root_and_links_to_its_causal_span() {
    let harness = Harness::new();
    harness.run(|| {
        let turn_span = tracing::info_span!(
            target: "halter_runtime",
            "turn",
            session_id = "sess-1",
            turn_id = "turn-1",
            provider = Empty,
            model = Empty,
            model_id = Empty,
            provider_iterations = Empty,
            input_tokens = Empty,
            output_tokens = Empty,
            cache_read_input_tokens = Empty,
            cache_creation_input_tokens = Empty,
            outcome = Empty,
        );
        let _turn_enter = turn_span.enter();

        // Mirrors `subagents.rs`: a root span, `follows_from` the still-open
        // causal span.
        let subagent_span = tracing::info_span!(
            target: "halter_runtime",
            parent: None,
            "subagent",
            agent_id = "agent-1",
            session_id = "sub-sess-1",
            parent_session_id = "sess-1",
            agent_type = "worker",
            generation = 1u64,
            outcome = Empty,
        );
        subagent_span.follows_from(tracing::Span::current());
        subagent_span.record("outcome", "completed");
        drop(subagent_span);

        turn_span.record("outcome", "completed");
    });
    harness.flush();

    let spans = harness.finished_spans();
    let turn = span_named(&spans, "turn");
    let subagent = span_named(&spans, "subagent");

    assert_eq!(
        subagent.parent_span_id,
        opentelemetry::trace::SpanId::INVALID,
        "subagent must be a root span, not a child of turn"
    );
    assert_eq!(
        subagent.links.len(),
        1,
        "subagent must link to exactly its causal span, got {:?}",
        subagent.links
    );
    assert_eq!(
        subagent.links[0].span_context.span_id(),
        turn.span_context.span_id(),
        "the link must point at the turn span that started the subagent"
    );
}

#[test]
fn provider_request_metrics_and_retry_counters_use_expected_attributes() {
    let harness = Harness::new();
    harness.run(|| {
        let span = tracing::info_span!(
            target: "halter_providers",
            "provider_request",
            provider = "anthropic",
            provider_kind = "anthropic",
            model = "claude-sonnet",
            model_id = "claude-sonnet-x",
            session_id = "sess-1",
            turn_id = "turn-1",
            attempt = Empty,
            outcome = Empty,
            error_kind = Empty,
        );
        let _enter = span.enter();
        tracing::info!(
            target: "halter_providers",
            provider_kind = "anthropic",
            attempt = 1u32,
            error_kind = ?FakeErrorKind::RateLimited,
            retry_in_ms = 250u64,
            "retrying provider request"
        );
        tracing::info!(
            target: "halter_providers",
            provider_kind = "anthropic",
            attempt = 2u32,
            error_kind = ?FakeErrorKind::Transient,
            retry_in_ms = 500u64,
            "retrying provider request"
        );
        // The exhausted-budget warning: no `retry_in_ms`, field named `kind`
        // (not `error_kind`) -- must NOT be counted as a retry.
        tracing::warn!(
            target: "halter_providers",
            provider_kind = "anthropic",
            attempt = 3u32,
            kind = ?FakeErrorKind::Transient,
            "provider retry budget exhausted"
        );
        span.record("attempt", 3u32);
        span.record("outcome", "completed");
    });
    harness.flush();

    let metrics = harness.finished_metrics();

    let retries = metric_named(&metrics, "halter.provider.retries");
    let AggregatedMetrics::U64(opentelemetry_sdk::metrics::data::MetricData::Sum(sum)) =
        retries.data()
    else {
        panic!("expected a u64 sum for halter.provider.retries");
    };
    assert_eq!(sum.data_points().count(), 2, "one per distinct error_kind");
    let total_retries: u64 = sum.data_points().map(|dp| dp.value()).sum();
    assert_eq!(
        total_retries, 2,
        "exactly 2 retry events, 0 for the exhausted warning"
    );

    let rate_limited = metric_named(&metrics, "halter.provider.rate_limited");
    let AggregatedMetrics::U64(opentelemetry_sdk::metrics::data::MetricData::Sum(sum)) =
        rate_limited.data()
    else {
        panic!("expected a u64 sum for halter.provider.rate_limited");
    };
    let total_rate_limited: u64 = sum.data_points().map(|dp| dp.value()).sum();
    assert_eq!(
        total_rate_limited, 1,
        "only the RateLimited retry increments the rate-limit counter"
    );

    let duration = metric_named(&metrics, "gen_ai.client.operation.duration");
    let AggregatedMetrics::F64(opentelemetry_sdk::metrics::data::MetricData::Histogram(hist)) =
        duration.data()
    else {
        panic!("expected an f64 histogram for gen_ai.client.operation.duration");
    };
    let point = hist.data_points().next().expect("one data point");
    let attrs: std::collections::BTreeMap<_, _> = point
        .attributes()
        .map(|kv| (kv.key.as_str().to_owned(), kv.value.as_str().to_string()))
        .collect();
    assert_eq!(attrs.get("outcome").map(String::as_str), Some("completed"));
    assert_eq!(
        attrs.get("gen_ai.system").map(String::as_str),
        Some("anthropic")
    );
    assert_eq!(
        attrs.get("halter.operation").map(String::as_str),
        Some("request")
    );
}

#[test]
fn token_usage_counter_records_per_type_gen_ai_attributes() {
    let harness = Harness::new();
    harness.run(emit_turn_and_tool_call);
    harness.flush();

    let metrics = harness.finished_metrics();
    let usage = metric_named(&metrics, "gen_ai.client.token.usage");
    let AggregatedMetrics::U64(opentelemetry_sdk::metrics::data::MetricData::Sum(sum)) =
        usage.data()
    else {
        panic!("expected a u64 sum for gen_ai.client.token.usage");
    };

    let mut by_type = std::collections::BTreeMap::new();
    for point in sum.data_points() {
        let token_type = point
            .attributes()
            .find(|kv| kv.key.as_str() == "gen_ai.token.type")
            .map(|kv| kv.value.as_str().to_string())
            .expect("gen_ai.token.type attribute");
        by_type.insert(token_type, point.value());
    }

    assert_eq!(by_type.get("input").copied(), Some(100));
    assert_eq!(by_type.get("output").copied(), Some(40));
    // cache_read_input_tokens was recorded as 0, and the layer skips
    // zero-valued token fields, so it must not appear at all.
    assert!(!by_type.contains_key("cache_read"));
}

#[test]
fn tool_call_duration_and_error_counter_use_tool_name_and_outcome() {
    let harness = Harness::new();
    harness.run(|| {
        let tool_span = tracing::info_span!(
            target: "halter_runtime",
            "tool_call",
            session_id = "sess-1",
            turn_id = "turn-1",
            tool_name = "shell",
            tool_call_id = "call-2",
            outcome = Empty,
            is_error = Empty,
        );
        let _enter = tool_span.enter();
        tool_span.record("outcome", "error");
        tool_span.record("is_error", true);
    });
    harness.flush();

    let metrics = harness.finished_metrics();
    let errors = metric_named(&metrics, "halter.tool_call.errors");
    let AggregatedMetrics::U64(opentelemetry_sdk::metrics::data::MetricData::Sum(sum)) =
        errors.data()
    else {
        panic!("expected a u64 sum for halter.tool_call.errors");
    };
    let point = sum.data_points().next().expect("one data point");
    assert_eq!(point.value(), 1);
    let attrs: std::collections::BTreeMap<_, _> = point
        .attributes()
        .map(|kv| (kv.key.as_str().to_owned(), kv.value.as_str().to_string()))
        .collect();
    assert_eq!(attrs.get("tool_name").map(String::as_str), Some("shell"));
    assert_eq!(attrs.get("outcome").map(String::as_str), Some("error"));
}

#[test]
fn subagents_in_flight_counter_tracks_overlapping_spans() {
    let harness = Harness::new();

    let gauge_value = |harness: &Harness| -> i64 {
        harness.flush();
        let metrics = harness.finished_metrics();
        // `get_finished_metrics` accumulates every export since the last
        // `reset`; take the most recent snapshot of this metric, then reset
        // so the next call doesn't re-read a stale one.
        let value = metrics
            .iter()
            .flat_map(|rm| rm.scope_metrics())
            .flat_map(|sm| sm.metrics())
            .filter(|m| m.name() == "halter.subagents.in_flight")
            .last()
            .map(|metric| {
                let AggregatedMetrics::I64(opentelemetry_sdk::metrics::data::MetricData::Sum(sum)) =
                    metric.data()
                else {
                    panic!("expected an i64 sum for halter.subagents.in_flight");
                };
                sum.data_points().map(|dp| dp.value()).sum()
            })
            .unwrap_or(0);
        harness.metric_exporter.reset();
        value
    };

    let new_subagent = |agent_id: &'static str| {
        tracing::info_span!(
            target: "halter_runtime",
            parent: None,
            "subagent",
            agent_id = agent_id,
            session_id = "sub-sess",
            parent_session_id = "sess-1",
            agent_type = "worker",
            generation = 1u64,
            outcome = Empty,
        )
    };

    let span_a = harness.run(|| new_subagent("agent-a"));
    assert_eq!(gauge_value(&harness), 1);

    let span_b = harness.run(|| new_subagent("agent-b"));
    assert_eq!(gauge_value(&harness), 2);

    harness.run(|| {
        span_a.record("outcome", "completed");
        drop(span_a);
    });
    assert_eq!(gauge_value(&harness), 1);

    harness.run(|| {
        span_b.record("outcome", "completed");
        drop(span_b);
    });
    assert_eq!(gauge_value(&harness), 0);
}

#[test]
fn metrics_never_carry_high_cardinality_identifiers() {
    let harness = Harness::new();
    harness.run(|| {
        emit_turn_and_tool_call();
        let span = tracing::info_span!(
            target: "halter_providers",
            "provider_request",
            provider = "anthropic",
            provider_kind = "anthropic",
            model = "claude-sonnet",
            model_id = "claude-sonnet-x",
            session_id = "sess-1",
            turn_id = "turn-1",
            attempt = Empty,
            outcome = Empty,
            error_kind = Empty,
        );
        let _enter = span.enter();
        span.record("outcome", "completed");
    });
    harness.flush();

    let metrics = harness.finished_metrics();
    let mut seen_keys = BTreeSet::new();
    for resource_metrics in &metrics {
        for scope in resource_metrics.scope_metrics() {
            for metric in scope.metrics() {
                collect_attribute_keys(metric.data(), &mut seen_keys);
            }
        }
    }

    for forbidden in FORBIDDEN_METRIC_ATTRIBUTES {
        assert!(
            !seen_keys.contains(*forbidden),
            "metric attributes must never include {forbidden}, saw: {seen_keys:?}"
        );
    }

    // Spans, in contrast, are expected to carry these identifiers.
    let spans = harness.finished_spans();
    let turn = span_named(&spans, "turn");
    assert!(attr(turn, "session_id").is_some());
}

fn collect_attribute_keys(data: &AggregatedMetrics, keys: &mut BTreeSet<String>) {
    use opentelemetry_sdk::metrics::data::MetricData;

    fn from_sum<T>(sum: &opentelemetry_sdk::metrics::data::Sum<T>, keys: &mut BTreeSet<String>) {
        for point in sum.data_points() {
            for kv in point.attributes() {
                keys.insert(kv.key.as_str().to_owned());
            }
        }
    }
    fn from_hist<T>(
        hist: &opentelemetry_sdk::metrics::data::Histogram<T>,
        keys: &mut BTreeSet<String>,
    ) {
        for point in hist.data_points() {
            for kv in point.attributes() {
                keys.insert(kv.key.as_str().to_owned());
            }
        }
    }

    match data {
        AggregatedMetrics::U64(MetricData::Sum(sum)) => from_sum(sum, keys),
        AggregatedMetrics::I64(MetricData::Sum(sum)) => from_sum(sum, keys),
        AggregatedMetrics::F64(MetricData::Sum(sum)) => from_sum(sum, keys),
        AggregatedMetrics::U64(MetricData::Histogram(hist)) => from_hist(hist, keys),
        AggregatedMetrics::I64(MetricData::Histogram(hist)) => from_hist(hist, keys),
        AggregatedMetrics::F64(MetricData::Histogram(hist)) => from_hist(hist, keys),
        _ => {}
    }
}
