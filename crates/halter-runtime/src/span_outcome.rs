//! Exactly-once `outcome` recording for operation spans.

use tracing::Span;

/// Outcome recorded when an operation ends without reporting one: its task
/// was aborted, its future was dropped, or it panicked.
pub(crate) const ABORTED: &str = "aborted";

/// Records a span's `outcome` field exactly once.
///
/// The first [`SpanOutcome::record`] wins; later calls are ignored, because
/// the compact formatter repeats a re-recorded field on every line. If no
/// outcome was recorded when the guard drops, it records [`ABORTED`], so a
/// closed span never has an empty `outcome`.
pub(crate) struct SpanOutcome {
    span: Span,
    recorded: bool,
}

impl SpanOutcome {
    pub(crate) fn new(span: Span) -> Self {
        Self {
            span,
            recorded: false,
        }
    }

    pub(crate) fn record(&mut self, outcome: &'static str) {
        if !self.recorded {
            self.recorded = true;
            self.span.record("outcome", outcome);
        }
    }
}

impl Drop for SpanOutcome {
    fn drop(&mut self) {
        self.record(ABORTED);
    }
}

#[cfg(test)]
mod tests {
    use tracing::field::Empty;

    use super::*;
    use crate::telemetry_capture::Capture;

    #[test]
    fn records_first_outcome_once_and_aborted_when_dropped_unset() {
        let capture = Capture::default();
        tracing::subscriber::with_default(capture.subscriber(), || {
            let mut reported = SpanOutcome::new(tracing::info_span!("reported", outcome = Empty));
            reported.record("completed");
            reported.record("failed");
            drop(reported);
            drop(SpanOutcome::new(tracing::info_span!(
                "dropped",
                outcome = Empty
            )));
        });

        assert_eq!(
            capture.spans_named("reported")[0].field("outcome"),
            Some("completed")
        );
        assert_eq!(
            capture.spans_named("dropped")[0].field("outcome"),
            Some(ABORTED)
        );
        assert_eq!(capture.record_count("reported", "outcome"), 1);
    }
}
