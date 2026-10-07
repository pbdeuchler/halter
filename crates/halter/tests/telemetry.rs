#![cfg(feature = "telemetry")]
//! Verifies that `halter::telemetry` installs a global subscriber only when
//! `try_init` is called. This file holds a single test so its global-state
//! steps run in a deterministic order (each integration test file is its own
//! process).

use std::io;
use std::sync::{Arc, Mutex};

use halter::telemetry::TelemetryConfig;
use halter::telemetry::tracing_subscriber::{self, fmt::MakeWriter, layer::SubscriberExt};

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

fn global_default_is_unset() -> bool {
    tracing::dispatcher::get_default(|dispatch| dispatch.is::<tracing::subscriber::NoSubscriber>())
}

#[test]
fn helper_installs_global_subscriber_only_when_initialized() {
    assert!(global_default_is_unset());

    // (a) Building the filter and layer and using them in a scope installs
    // nothing globally.
    let scoped_buffer = BufferWriter::default();
    let config = TelemetryConfig::new()
        .with_directives("info")
        .with_writer(scoped_buffer.clone());
    let subscriber = tracing_subscriber::registry()
        .with(config.env_filter().expect("filter"))
        .with(config.fmt_layer());
    tracing::subscriber::with_default(subscriber, || tracing::info!("scoped event"));
    assert!(scoped_buffer.contents().contains("scoped event"));
    assert!(global_default_is_unset());

    // (b) `try_init` installs the global default.
    let buffer = BufferWriter::default();
    TelemetryConfig::new()
        .with_directives("info")
        .with_writer(buffer.clone())
        .try_init()
        .expect("first init succeeds");
    assert!(!global_default_is_unset());

    // (c) Events reach the configured writer.
    tracing::info!(answer = 42, "global event");
    tracing::debug!("filtered out");
    let output = buffer.contents();
    assert!(output.contains("global event"), "{output}");
    assert!(output.contains("answer"), "{output}");
    assert!(!output.contains("filtered out"), "{output}");

    // (d) A second init reports an error instead of panicking.
    let error = TelemetryConfig::new()
        .try_init()
        .expect_err("second init fails");
    assert!(error.to_string().contains("failed to initialize logging"));
}
