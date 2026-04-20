// pattern: Functional Core

use std::fmt;

/// Opaque wrapper around a credential whose `Debug`/`Display` implementations
/// refuse to reveal the wrapped value. The only way to access the underlying
/// string is [`SecretString::expose_secret`], making accidental logging
/// impossible through normal formatting.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns a reference to the wrapped secret. Callers are responsible for
    /// not logging or otherwise leaking the returned string.
    #[must_use]
    pub fn expose_secret(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString(<redacted>)")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_and_display_redact() {
        let secret = SecretString::new("hunter2");
        assert_eq!(format!("{secret:?}"), "SecretString(<redacted>)");
        assert_eq!(format!("{secret}"), "<redacted>");
    }

    #[test]
    fn expose_secret_returns_wrapped_value() {
        let secret = SecretString::from("hunter2".to_owned());
        assert_eq!(secret.expose_secret(), "hunter2");
    }

    #[test]
    fn eq_is_structural() {
        assert_eq!(SecretString::from("a"), SecretString::from("a"));
        assert_ne!(SecretString::from("a"), SecretString::from("b"));
    }
}
