// pattern: Functional Core

#[derive(Debug, Clone, Copy)]
/// Thresholds that decide *when* the runtime compacts and where it caps the
/// context. What happens then is the installed
/// [`CompactionStrategy`](crate::CompactionStrategy)'s business.
pub struct ContextSettings {
    /// Compact once the ledger's effective count reaches this many tokens.
    pub compaction_threshold: u64,
    /// Hard cap on the effective count. Checked after compaction has had its
    /// chance, before every provider request; exceeding it fails the turn
    /// with [`ContextCapExceeded`] instead of blowing the provider's window.
    pub max_tokens: Option<u64>,
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            compaction_threshold: 80_000,
            max_tokens: None,
        }
    }
}

impl ContextSettings {
    /// Whether the effective ledger count reached the threshold.
    #[must_use]
    pub fn compaction_due(&self, effective_tokens: u64) -> bool {
        effective_tokens >= self.compaction_threshold
    }

    /// Enforce the hard cap on the effective ledger count.
    pub fn check_cap(&self, effective_tokens: u64) -> Result<(), ContextCapExceeded> {
        match self.max_tokens {
            Some(max_tokens) if effective_tokens > max_tokens => Err(ContextCapExceeded {
                effective_tokens,
                max_tokens,
            }),
            _ => Ok(()),
        }
    }
}

/// The session context grew past `context.max_tokens` and compaction could
/// not bring it back under. Raised before the provider request that would
/// have carried the oversized context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "failed to run turn: session context of {effective_tokens} tokens exceeds context.max_tokens ({max_tokens})"
)]
pub struct ContextCapExceeded {
    pub effective_tokens: u64,
    pub max_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_due_uses_the_exact_threshold() {
        let settings = ContextSettings {
            compaction_threshold: 1_000,
            max_tokens: None,
        };
        struct Case {
            effective_tokens: u64,
            due: bool,
        }
        let cases = [
            Case {
                effective_tokens: 0,
                due: false,
            },
            Case {
                effective_tokens: 999,
                due: false,
            },
            Case {
                effective_tokens: 1_000,
                due: true,
            },
            Case {
                effective_tokens: u64::MAX,
                due: true,
            },
        ];
        for case in cases {
            assert_eq!(
                settings.compaction_due(case.effective_tokens),
                case.due,
                "{} tokens",
                case.effective_tokens
            );
        }
    }

    #[test]
    fn check_cap_fails_only_past_a_configured_cap() {
        let uncapped = ContextSettings::default();
        assert_eq!(uncapped.check_cap(u64::MAX), Ok(()));

        let capped = ContextSettings {
            max_tokens: Some(500),
            ..ContextSettings::default()
        };
        assert_eq!(capped.check_cap(500), Ok(()));
        let error = capped.check_cap(501).expect_err("over cap");
        assert_eq!(
            error,
            ContextCapExceeded {
                effective_tokens: 501,
                max_tokens: 500,
            }
        );
        assert!(
            error
                .to_string()
                .contains("501 tokens exceeds context.max_tokens (500)")
        );
    }
}
