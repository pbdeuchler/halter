// pattern: Functional Core

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Instant;

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use parking_lot::Mutex;
use serde_json::json;
use smallvec::SmallVec;

use crate::{Tool, ToolContext};

const MAX_SAMPLES: usize = 10_000;

static PROCESS_START: LazyLock<Instant> = LazyLock::new(Instant::now);
static PROFILE_BUFFER: LazyLock<Mutex<CircularBuffer>> =
    LazyLock::new(|| Mutex::new(CircularBuffer::new(MAX_SAMPLES)));

#[derive(Clone)]
struct ProfileSample {
    session_id: String,
    stack: SmallVec<[&'static str; 4]>,
    duration_us: u64,
    timestamp_us: u64,
}

struct CircularBuffer {
    samples: Vec<ProfileSample>,
    capacity: usize,
    write_pos: usize,
}

impl CircularBuffer {
    const fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::new(),
            capacity,
            write_pos: 0,
        }
    }

    fn push(&mut self, sample: ProfileSample) {
        if self.samples.len() < self.capacity {
            self.samples.push(sample);
            return;
        }

        self.samples[self.write_pos] = sample;
        self.write_pos = (self.write_pos + 1) % self.capacity;
    }

    fn get_since(&self, cutoff_us: u64, session_id: Option<&str>) -> Vec<ProfileSample> {
        self.samples
            .iter()
            .filter(|sample| {
                sample.timestamp_us >= cutoff_us
                    && session_id.is_none_or(|session_id| sample.session_id == session_id)
            })
            .cloned()
            .collect()
    }
}

pub struct FlatProfileGuard {
    region: &'static str,
    session_id: String,
    start: Instant,
}

impl FlatProfileGuard {
    fn new(region: &'static str, session_id: impl Into<String>) -> Self {
        Self {
            region,
            session_id: session_id.into(),
            start: Instant::now(),
        }
    }
}

impl Drop for FlatProfileGuard {
    fn drop(&mut self) {
        let duration_us = self.start.elapsed().as_micros() as u64;
        let timestamp_us = PROCESS_START.elapsed().as_micros() as u64;
        let mut stack = SmallVec::new();
        stack.push(self.region);
        PROFILE_BUFFER.lock().push(ProfileSample {
            session_id: self.session_id.clone(),
            stack,
            duration_us,
            timestamp_us,
        });
    }
}

#[must_use]
pub(crate) fn profile_flat_region(region: &'static str, session_id: &str) -> FlatProfileGuard {
    FlatProfileGuard::new(region, session_id)
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkProfile {
    pub folded: String,
    pub summary: String,
    pub svg: Option<String>,
    pub total_ms: f64,
    pub sample_count: u32,
}

#[must_use]
#[cfg(any(test, feature = "profiling"))]
pub fn get_work_profile_for_session(last_seconds: f64, session_id: Option<&str>) -> WorkProfile {
    let window_us = (last_seconds * 1_000_000.0).max(0.0) as u64;
    let now_us = PROCESS_START.elapsed().as_micros() as u64;
    let cutoff_us = now_us.saturating_sub(window_us);
    let samples = PROFILE_BUFFER.lock().get_since(cutoff_us, session_id);
    let folded = generate_folded(&samples);
    let summary = generate_summary(&samples, last_seconds * 1000.0);
    let total_us: u64 = samples.iter().map(|sample| sample.duration_us).sum();

    WorkProfile {
        svg: generate_svg(&folded),
        folded,
        summary,
        total_ms: total_us as f64 / 1000.0,
        // Saturating cast: sample counts above u32::MAX are not actionable
        // downstream, but silent truncation would hide a runaway buffer.
        // Saturate and let the caller see a pinned ceiling. (finding L27)
        sample_count: u32::try_from(samples.len()).unwrap_or(u32::MAX),
    }
}

#[cfg(feature = "profiling")]
#[derive(Debug)]
pub struct ProfilingTool;

#[cfg(feature = "profiling")]
#[async_trait]
impl Tool for ProfilingTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("profile"),
            description: "Inspect the always-on tool execution profiler".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "last_seconds": { "type": "number", "minimum": 0 }
                }
            }),
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: ToolCapabilities::default(),
            provider_aliases: Default::default(),
        }
    }

    async fn execute(
        &self,
        context: ToolContext,
        input: serde_json::Value,
    ) -> anyhow::Result<ToolResult> {
        let last_seconds = input
            .get("last_seconds")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(30.0);
        let profile = get_work_profile_for_session(last_seconds, Some(&context.session_id.0));
        Ok(ToolResult::Json {
            value: json!({
                "folded": profile.folded,
                "summary": profile.summary,
                "svg": profile.svg,
                "total_ms": profile.total_ms,
                "sample_count": profile.sample_count,
            }),
        })
    }
}

#[cfg(any(test, feature = "profiling"))]
fn generate_folded(samples: &[ProfileSample]) -> String {
    let mut aggregated: HashMap<String, u64> = HashMap::new();
    for sample in samples {
        if sample.stack.is_empty() {
            continue;
        }
        let key = sample.stack.join(";");
        *aggregated.entry(key).or_default() += sample.duration_us;
    }

    let mut sorted: Vec<_> = aggregated.into_iter().collect();
    sorted.sort_by_key(|(_, duration_us)| Reverse(*duration_us));

    let mut output = String::new();
    for (stack, duration_us) in sorted {
        output.push_str(&stack);
        output.push(' ');
        output.push_str(&duration_us.to_string());
        output.push('\n');
    }
    output
}

#[cfg(any(test, feature = "profiling"))]
fn generate_summary(samples: &[ProfileSample], window_ms: f64) -> String {
    let mut by_region: HashMap<&'static str, (u64, usize)> = HashMap::new();
    for sample in samples {
        if let Some(region) = sample.stack.last() {
            let entry = by_region.entry(region).or_insert((0, 0));
            entry.0 += sample.duration_us;
            entry.1 += 1;
        }
    }

    let mut sorted: Vec<_> = by_region.into_iter().collect();
    sorted.sort_by_key(|(_, (duration_us, _))| Reverse(*duration_us));
    let total_us: u64 = sorted
        .iter()
        .map(|(_, (duration_us, _))| *duration_us)
        .sum();

    let mut lines = vec![
        "# Work Profile Summary".to_owned(),
        String::new(),
        format!("Window: {window_ms:.1}ms"),
        format!("Total work time: {:.1}ms", total_us as f64 / 1000.0),
        format!("Samples: {}", samples.len()),
        String::new(),
        "| Region | Time (ms) | % | Calls |".to_owned(),
        "|--------|-----------|---|-------|".to_owned(),
    ];

    for (region, (duration_us, count)) in sorted {
        let percent = if total_us == 0 {
            0.0
        } else {
            (duration_us as f64 / total_us as f64) * 100.0
        };
        lines.push(format!(
            "| {region} | {:.2} | {:.1}% | {count} |",
            duration_us as f64 / 1000.0,
            percent
        ));
    }
    lines.join("\n")
}

#[cfg(feature = "profiling")]
fn generate_svg(folded: &str) -> Option<String> {
    if folded.is_empty() {
        return None;
    }

    let mut options = inferno::flamegraph::Options::default();
    options.title = "Work Profile".to_owned();
    options.count_name = "μs".to_owned();
    options.min_width = 0.1;

    let mut output = Vec::new();
    inferno::flamegraph::from_reader(&mut options, std::io::Cursor::new(folded), &mut output)
        .ok()?;
    String::from_utf8(output).ok()
}

#[cfg(not(feature = "profiling"))]
fn generate_svg(_folded: &str) -> Option<String> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_profile_by_session() {
        drop(FlatProfileGuard::new("alpha", "session-a"));
        drop(FlatProfileGuard::new("beta", "session-b"));

        let profile = get_work_profile_for_session(10.0, Some("session-a"));
        assert!(profile.folded.contains("alpha"));
        assert!(!profile.folded.contains("beta"));
    }
}
