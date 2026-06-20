use std::env;

use anyhow::{Context, bail};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

pub(crate) const NOISY_TARGET_SUPPRESSIONS: &str = "tokenize=warn,parse=warn,expansion=warn,commands=warn,\
     pattern=warn,completion=warn,jobs=warn,unimplemented=warn,\
     hyper_util=warn,hyper=warn,reqwest=warn,h2=warn,rustls=warn";

pub(crate) fn logging_filter_spec(user_directives: Option<&str>) -> String {
    let directives = user_directives
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("info");
    format!("{directives},{NOISY_TARGET_SUPPRESSIONS}")
}

pub(crate) fn logging_filter_from_spec(spec: &str) -> anyhow::Result<EnvFilter> {
    EnvFilter::try_new(spec).context("invalid RUST_LOG filter")
}

pub(crate) fn init_logging() -> anyhow::Result<()> {
    let user_directives = match env::var(EnvFilter::DEFAULT_ENV) {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => bail!("invalid utf-8 in RUST_LOG"),
    };
    let filter = logging_filter_from_spec(&logging_filter_spec(user_directives.as_deref()))?;
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true).compact())
        .try_init()
        .context("failed to initialize logging")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub(crate) fn logging_filter_spec_defaults_or_uses_rust_log_and_appends_suppressions() {
        pub(crate) struct Case {
            pub(crate) name: &'static str,
            pub(crate) user_directives: Option<&'static str>,
            pub(crate) expected_prefix: &'static str,
        }

        let cases = [
            Case {
                name: "missing rust log defaults to info",
                user_directives: None,
                expected_prefix: "info,",
            },
            Case {
                name: "blank rust log defaults to info",
                user_directives: Some(" \n"),
                expected_prefix: "info,",
            },
            Case {
                name: "configured rust log is preserved",
                user_directives: Some("debug,halter=trace"),
                expected_prefix: "debug,halter=trace,",
            },
        ];

        for case in cases {
            let spec = logging_filter_spec(case.user_directives);

            assert!(
                spec.starts_with(case.expected_prefix),
                "{}: {spec}",
                case.name
            );
            assert!(spec.contains(NOISY_TARGET_SUPPRESSIONS), "{}", case.name);
            logging_filter_from_spec(&spec).expect(case.name);
        }
    }

    #[test]
    pub(crate) fn logging_filter_from_spec_covers_valid_and_invalid_specs() {
        logging_filter_from_spec("info,halter=debug").expect("valid filter");

        let error =
            logging_filter_from_spec("halter=not-a-level").expect_err("invalid level should fail");

        assert!(error.to_string().contains("invalid RUST_LOG filter"));
    }
}
