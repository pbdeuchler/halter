//! Test-only `tracing` layer that records spans and events so tests can
//! assert on span names, parents, and fields. Install it with a scoped
//! default (`tracing::subscriber::set_default`), never globally.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::subscriber::Interest;
use tracing::{Event, Level, Metadata, Subscriber};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;

#[derive(Clone, Default)]
pub(crate) struct Capture {
    spans: Arc<Mutex<Vec<CapturedSpan>>>,
    events: Arc<Mutex<Vec<CapturedEvent>>>,
    /// `(span name, field name)` for every `Span::record` call.
    records: Arc<Mutex<Vec<(&'static str, String)>>>,
    /// `(span name, followed span name)` for every `follows_from` link.
    follows: Arc<Mutex<Vec<(&'static str, &'static str)>>>,
}

#[derive(Clone, Debug)]
pub(crate) struct CapturedSpan {
    pub(crate) id: u64,
    pub(crate) name: &'static str,
    pub(crate) parent: Option<&'static str>,
    pub(crate) fields: BTreeMap<String, String>,
}

impl CapturedSpan {
    pub(crate) fn field(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CapturedEvent {
    pub(crate) level: Level,
    pub(crate) message: String,
    pub(crate) fields: BTreeMap<String, String>,
    /// Names of the enclosing spans, innermost first.
    pub(crate) scope: Vec<&'static str>,
}

impl CapturedEvent {
    pub(crate) fn field(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }
}

fn is_halter(metadata: &Metadata<'_>) -> bool {
    metadata.target().starts_with("halter")
}

#[derive(Default)]
struct FieldVisitor(BTreeMap<String, String>);

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

impl<S> Layer<S> for Capture
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        // Only halter's own callsites matter; ignoring dependencies keeps the
        // capture cheap for concurrently running, timing-sensitive tests.
        // `sometimes` (not `always`) because interest is shared with other
        // tests' threads, which have no subscriber.
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
        let mut visitor = FieldVisitor::default();
        attrs.record(&mut visitor);
        let parent = ctx
            .span(id)
            .and_then(|span| span.parent())
            .map(|parent| parent.name());
        self.spans.lock().expect("spans").push(CapturedSpan {
            id: id.into_u64(),
            name: attrs.metadata().name(),
            parent,
            fields: visitor.0,
        });
    }

    fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        values.record(&mut visitor);
        if let Some(span) = ctx.span(id) {
            let mut records = self.records.lock().expect("records");
            for field in visitor.0.keys() {
                records.push((span.name(), field.clone()));
            }
        }
        let mut spans = self.spans.lock().expect("spans");
        // Span ids are reused after close; the latest span with the id is
        // the live one.
        if let Some(span) = spans.iter_mut().rev().find(|span| span.id == id.into_u64()) {
            span.fields.extend(visitor.0);
        }
    }

    fn on_follows_from(&self, span: &Id, follows: &Id, ctx: Context<'_, S>) {
        if let (Some(span), Some(follows)) = (ctx.span(span), ctx.span(follows)) {
            self.follows
                .lock()
                .expect("follows")
                .push((span.name(), follows.name()));
        }
    }

    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let mut visitor = FieldVisitor::default();
        event.record(&mut visitor);
        let mut fields = visitor.0;
        let message = fields.remove("message").unwrap_or_default();
        let scope = ctx
            .event_scope(event)
            .map(|scope| scope.map(|span| span.name()).collect())
            .unwrap_or_default();
        self.events.lock().expect("events").push(CapturedEvent {
            level: *event.metadata().level(),
            message,
            fields,
            scope,
        });
    }
}

/// Keep a second dispatcher registered for the whole test process.
///
/// With exactly one live dispatcher, tracing-core computes the interest of a
/// newly registered callsite from the registering thread's default only. A
/// test running on another thread without a subscriber would then cache
/// `never` for a callsite this capture needs. A permanent second dispatcher
/// keeps interest computed across all live dispatchers.
pub(crate) fn keep_interest_shared() {
    static KEEPALIVE: std::sync::OnceLock<tracing::Dispatch> = std::sync::OnceLock::new();
    KEEPALIVE.get_or_init(|| tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default()));
}

impl Capture {
    pub(crate) fn subscriber(&self) -> impl Subscriber + Send + Sync + 'static {
        keep_interest_shared();
        tracing_subscriber::registry().with(self.clone())
    }

    pub(crate) fn spans_named(&self, name: &str) -> Vec<CapturedSpan> {
        self.spans
            .lock()
            .expect("spans")
            .iter()
            .filter(|span| span.name == name)
            .cloned()
            .collect()
    }

    /// Names of the spans that spans named `span` follow from.
    pub(crate) fn follows_from(&self, span: &str) -> Vec<&'static str> {
        self.follows
            .lock()
            .expect("follows")
            .iter()
            .filter(|(name, _)| *name == span)
            .map(|(_, follows)| *follows)
            .collect()
    }

    /// How many times `field` was recorded on spans named `span`.
    pub(crate) fn record_count(&self, span: &str, field: &str) -> usize {
        self.records
            .lock()
            .expect("records")
            .iter()
            .filter(|(name, recorded)| *name == span && recorded == field)
            .count()
    }

    pub(crate) fn events_with_message(&self, message: &str) -> Vec<CapturedEvent> {
        self.events
            .lock()
            .expect("events")
            .iter()
            .filter(|event| event.message == message)
            .cloned()
            .collect()
    }
}

/// In-memory `MakeWriter` for asserting on formatted log output.
#[derive(Clone, Default)]
pub(crate) struct BufferWriter(Arc<Mutex<Vec<u8>>>);

impl BufferWriter {
    pub(crate) fn contents(&self) -> String {
        String::from_utf8(self.0.lock().expect("buffer").clone()).expect("utf-8")
    }
}

impl std::io::Write for BufferWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("buffer").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for BufferWriter {
    type Writer = Self;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// A subscriber with the same compact formatter `halter::telemetry` installs
/// (`fmt::layer().with_target(true).compact()`; ANSI off for matching),
/// limited to halter targets at `info`. `halter-runtime` cannot depend on
/// `halter`, so the layer is rebuilt here.
pub(crate) fn compact_subscriber(writer: BufferWriter) -> impl Subscriber + Send + Sync + 'static {
    use tracing_subscriber::Layer as _;
    keep_interest_shared();
    let filter = tracing_subscriber::filter::Targets::new().with_target("halter", Level::INFO);
    tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .with_writer(writer)
            .with_ansi(false)
            .with_target(true)
            .compact()
            .with_filter(filter),
    )
}
