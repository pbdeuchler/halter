//! Derives OTel metrics from the span/event contract documented in
//! `README.md`'s "Observability" section, instead of requiring library
//! crates to carry an OTel dependency. [`OtelMetricsLayer`] watches the
//! same spans and events the `tracing` subscriber already sees and records
//! OTel instruments at span close (and at specific retry events).
//!
//! Modeled on `halter-runtime`'s test-only `telemetry_capture::Capture`
//! layer: a `register_callsite`/`enabled` filter restricted to `halter*`
//! targets, a `tracing::field::Visit` implementation that records every
//! field as a string, and span-local state stashed in the registry's span
//! extensions.
//!
//! Every attribute key used here is deliberately low-cardinality: never
//! `session_id`, `turn_id`, `tool_call_id`, or `agent_id`.

use std::fmt;
use std::time::Instant;

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Histogram, Meter, UpDownCounter};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::{Event, Metadata, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// `gen_ai.*` and `halter.*` instruments derived from the documented span
/// contract. See README "Observability" -> "OpenTelemetry" for the full
/// table (name, unit, attributes, source).
pub(crate) struct Instruments {
    /// `gen_ai.client.token.usage` (Counter<u64>, `{token}`): `gen_ai.system`,
    /// `gen_ai.request.model`, `gen_ai.token.type`. Recorded once per
    /// non-zero token field on `turn` close.
    token_usage: Counter<u64>,
    /// `halter.turn.duration` (Histogram<f64>, `s`): `outcome`.
    turn_duration: Histogram<f64>,
    /// `halter.tool_call.duration` (Histogram<f64>, `s`): `tool_name`, `outcome`.
    tool_call_duration: Histogram<f64>,
    /// `halter.tool_call.errors` (Counter<u64>, `{error}`): `tool_name`, `outcome`.
    tool_call_errors: Counter<u64>,
    /// `gen_ai.client.operation.duration` (Histogram<f64>, `s`): `gen_ai.system`,
    /// `gen_ai.request.model`, `outcome`, `halter.operation`.
    provider_operation_duration: Histogram<f64>,
    /// `halter.provider.retries` (Counter<u64>, `{retry}`): `provider_kind`, `error_kind`.
    provider_retries: Counter<u64>,
    /// `halter.provider.rate_limited` (Counter<u64>, `{retry}`): `provider_kind`.
    provider_rate_limited: Counter<u64>,
    /// `halter.subagents.in_flight` (UpDownCounter<i64>, `{subagent}`): `agent_type`.
    subagents_in_flight: UpDownCounter<i64>,
}

impl Instruments {
    pub(crate) fn build(meter: &Meter) -> Self {
        Self {
            token_usage: meter
                .u64_counter("gen_ai.client.token.usage")
                .with_unit("{token}")
                .with_description("Number of tokens used, by GenAI system, model, and token type.")
                .build(),
            turn_duration: meter
                .f64_histogram("halter.turn.duration")
                .with_unit("s")
                .with_description("Duration of a halter_runtime `turn` span.")
                .build(),
            tool_call_duration: meter
                .f64_histogram("halter.tool_call.duration")
                .with_unit("s")
                .with_description("Duration of a halter_runtime `tool_call` span.")
                .build(),
            tool_call_errors: meter
                .u64_counter("halter.tool_call.errors")
                .with_unit("{error}")
                .with_description("Tool calls whose outcome was not `ok`.")
                .build(),
            provider_operation_duration: meter
                .f64_histogram("gen_ai.client.operation.duration")
                .with_unit("s")
                .with_description("Duration of a `provider_request`/`provider_compaction` span.")
                .build(),
            provider_retries: meter
                .u64_counter("halter.provider.retries")
                .with_unit("{retry}")
                .with_description(
                    "Provider request/compaction retries, by provider and error kind.",
                )
                .build(),
            provider_rate_limited: meter
                .u64_counter("halter.provider.rate_limited")
                .with_unit("{retry}")
                .with_description("Provider retries caused by a `RateLimited` error.")
                .build(),
            subagents_in_flight: meter
                .i64_up_down_counter("halter.subagents.in_flight")
                .with_unit("{subagent}")
                .with_description("Subagents with an open `subagent` span, by agent type.")
                .build(),
        }
    }
}

fn is_halter(metadata: &Metadata<'_>) -> bool {
    metadata.target().starts_with("halter")
}

/// Every field recorded on a span, as strings (mirrors
/// `halter_runtime::telemetry_capture::FieldVisitor`).
#[derive(Default, Clone)]
struct FieldVisitor(std::collections::BTreeMap<String, String>);

impl Visit for FieldVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        self.0.insert(field.name().to_owned(), value.to_owned());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.0.insert(field.name().to_owned(), value.to_string());
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.0.insert(field.name().to_owned(), format!("{value:?}"));
    }
}

/// Per-span state stashed in the registry's extensions between
/// `on_new_span` and `on_close`.
struct SpanState {
    start: Instant,
    fields: std::collections::BTreeMap<String, String>,
}

/// `Layer<Registry>` that derives OTel metrics from the `turn`, `tool_call`,
/// `provider_request`, `provider_compaction`, and `subagent` spans, plus
/// `ResilientProvider`'s retry events. See the module docs.
pub(crate) struct OtelMetricsLayer {
    instruments: Instruments,
}

impl OtelMetricsLayer {
    pub(crate) fn new(meter: &Meter) -> Self {
        Self {
            instruments: Instruments::build(meter),
        }
    }
}

/// Span names this layer derives metrics from. Any other span (including
/// `hook_dispatch`) is ignored even though its target matches `halter*`.
fn tracked_span_name(name: &str) -> bool {
    matches!(
        name,
        "turn" | "tool_call" | "provider_request" | "provider_compaction" | "subagent"
    )
}

impl<S> Layer<S> for OtelMetricsLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if is_halter(metadata) {
            Interest::sometimes()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        is_halter(metadata)
    }

    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let name = attrs.metadata().name();
        if !tracked_span_name(name) {
            return;
        }
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);

        if name == "subagent" {
            let agent_type = visitor
                .0
                .get("agent_type")
                .map(String::as_str)
                .unwrap_or("");
            self.instruments
                .subagents_in_flight
                .add(1, &[KeyValue::new("agent_type", agent_type.to_owned())]);
        }

        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(SpanState {
                start: Instant::now(),
                fields: visitor.0,
            });
        }
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let mut extensions = span.extensions_mut();
        let Some(state) = extensions.get_mut::<SpanState>() else {
            return;
        };
        let mut visitor = FieldVisitor::default();
        values.record(&mut visitor);
        state.fields.extend(visitor.0);
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        // Restrict to events nested inside a `provider_request` /
        // `provider_compaction` span, matching the real retry call sites in
        // `ResilientProvider` (defense in depth; the field check below is
        // what actually identifies a retry).
        let in_provider_span = ctx
            .event_scope(event)
            .and_then(|mut scope| scope.next())
            .map(|span| matches!(span.name(), "provider_request" | "provider_compaction"))
            .unwrap_or(false);
        if !in_provider_span {
            return;
        }

        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        // A retry event always carries both `error_kind` and `retry_in_ms`;
        // the exhausted-budget warning carries neither (it uses `kind`
        // instead), so this field-shape check (not the message string)
        // robustly identifies retries regardless of future wording changes.
        let (Some(error_kind), Some(_retry_in_ms)) =
            (visitor.0.get("error_kind"), visitor.0.get("retry_in_ms"))
        else {
            return;
        };
        let provider_kind = visitor
            .0
            .get("provider_kind")
            .map(String::as_str)
            .unwrap_or("")
            .to_owned();

        self.instruments.provider_retries.add(
            1,
            &[
                KeyValue::new("provider_kind", provider_kind.clone()),
                KeyValue::new("error_kind", error_kind.clone()),
            ],
        );
        if error_kind == "RateLimited" {
            self.instruments
                .provider_rate_limited
                .add(1, &[KeyValue::new("provider_kind", provider_kind)]);
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else { return };
        let Some(state) = span.extensions_mut().remove::<SpanState>() else {
            return;
        };
        let elapsed = state.start.elapsed().as_secs_f64();
        let fields = &state.fields;
        let field = |name: &str| fields.get(name).map(String::as_str).unwrap_or("");

        match span.name() {
            "turn" => {
                let outcome = field("outcome");
                self.instruments
                    .turn_duration
                    .record(elapsed, &[KeyValue::new("outcome", outcome.to_owned())]);

                let provider = field("provider").to_owned();
                let model = field("model").to_owned();
                for (field_name, token_type) in [
                    ("input_tokens", "input"),
                    ("output_tokens", "output"),
                    ("cache_read_input_tokens", "cache_read"),
                    ("cache_creation_input_tokens", "cache_creation"),
                ] {
                    let Some(value) = fields.get(field_name).and_then(|v| v.parse::<u64>().ok())
                    else {
                        continue;
                    };
                    if value == 0 {
                        continue;
                    }
                    self.instruments.token_usage.add(
                        value,
                        &[
                            KeyValue::new("gen_ai.system", provider.clone()),
                            KeyValue::new("gen_ai.request.model", model.clone()),
                            KeyValue::new("gen_ai.token.type", token_type),
                        ],
                    );
                }
            }
            "tool_call" => {
                let tool_name = field("tool_name").to_owned();
                let outcome = field("outcome").to_owned();
                self.instruments.tool_call_duration.record(
                    elapsed,
                    &[
                        KeyValue::new("tool_name", tool_name.clone()),
                        KeyValue::new("outcome", outcome.clone()),
                    ],
                );
                if outcome != "ok" {
                    self.instruments.tool_call_errors.add(
                        1,
                        &[
                            KeyValue::new("tool_name", tool_name),
                            KeyValue::new("outcome", outcome),
                        ],
                    );
                }
            }
            "provider_request" | "provider_compaction" => {
                let operation = if span.name() == "provider_request" {
                    "request"
                } else {
                    "compaction"
                };
                self.instruments.provider_operation_duration.record(
                    elapsed,
                    &[
                        KeyValue::new("gen_ai.system", field("provider_kind").to_owned()),
                        KeyValue::new("gen_ai.request.model", field("model").to_owned()),
                        KeyValue::new("outcome", field("outcome").to_owned()),
                        KeyValue::new("halter.operation", operation),
                    ],
                );
            }
            "subagent" => {
                let agent_type = field("agent_type").to_owned();
                self.instruments
                    .subagents_in_flight
                    .add(-1, &[KeyValue::new("agent_type", agent_type)]);
            }
            _ => {}
        }
    }
}
