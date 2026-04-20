// pattern: Functional Core

use std::collections::HashSet;

use anyhow::Context;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

/// User-supplied header overrides applied case-insensitively.
///
/// HTTP header names are case-insensitive, so callers can set
/// `authorization` or `Authorization` and it will still override a default
/// `AUTHORIZATION`. Values are preserved verbatim.
#[derive(Debug, Clone, Default)]
pub(crate) struct HeaderOverrides {
    entries: Vec<(HeaderName, HeaderValue)>,
    lowercased_names: HashSet<String>,
}

impl HeaderOverrides {
    pub(crate) fn new(raw: &[(String, String)]) -> anyhow::Result<Self> {
        let mut entries = Vec::with_capacity(raw.len());
        let mut lowercased_names = HashSet::with_capacity(raw.len());
        for (name, value) in raw {
            let header_name = HeaderName::from_bytes(name.as_bytes())
                .with_context(|| format!("invalid http header name '{name}'"))?;
            let header_value = HeaderValue::from_str(value)
                .with_context(|| format!("invalid http header value for '{name}'"))?;
            lowercased_names.insert(header_name.as_str().to_ascii_lowercase());
            entries.push((header_name, header_value));
        }
        Ok(Self {
            entries,
            lowercased_names,
        })
    }

    #[must_use]
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.lowercased_names.contains(&name.to_ascii_lowercase())
    }

    /// Insert overrides into an existing [`HeaderMap`], replacing any entries
    /// already present with the same (case-insensitive) name.
    pub(crate) fn apply_to_map(&self, map: &mut HeaderMap) {
        for (name, value) in &self.entries {
            map.insert(name.clone(), value.clone());
        }
    }

    /// Merge overrides into a `(name, value)` list. Defaults with a name that
    /// appears in `self` are dropped; overrides are appended in insertion
    /// order.
    #[must_use]
    pub(crate) fn merge_string_pairs(
        &self,
        defaults: Vec<(String, String)>,
    ) -> Vec<(String, String)> {
        let mut merged: Vec<(String, String)> = defaults
            .into_iter()
            .filter(|(name, _)| !self.contains(name))
            .collect();
        merged.extend(
            self.entries
                .iter()
                .map(|(name, value)| {
                    let value_str = value
                        .to_str()
                        .expect("HeaderOverrides values are constructed from &str")
                        .to_owned();
                    (name.as_str().to_owned(), value_str)
                }),
        );
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_replaces_default_case_insensitively() {
        let overrides =
            HeaderOverrides::new(&[("authorization".to_owned(), "Token abc".to_owned())])
                .expect("valid overrides");

        let merged = overrides.merge_string_pairs(vec![
            ("Authorization".to_owned(), "Bearer default".to_owned()),
            ("x-api-key".to_owned(), "secret".to_owned()),
        ]);

        assert_eq!(
            merged,
            vec![
                ("x-api-key".to_owned(), "secret".to_owned()),
                ("authorization".to_owned(), "Token abc".to_owned()),
            ]
        );
    }

    #[test]
    fn non_conflicting_override_is_appended() {
        let overrides = HeaderOverrides::new(&[(
            "x-trace-id".to_owned(),
            "trace-123".to_owned(),
        )])
        .expect("valid overrides");

        let merged = overrides.merge_string_pairs(vec![(
            "x-api-key".to_owned(),
            "secret".to_owned(),
        )]);

        assert_eq!(
            merged,
            vec![
                ("x-api-key".to_owned(), "secret".to_owned()),
                ("x-trace-id".to_owned(), "trace-123".to_owned()),
            ]
        );
    }

    #[test]
    fn apply_to_map_overwrites_existing_entries() {
        let overrides =
            HeaderOverrides::new(&[("Content-Type".to_owned(), "application/xml".to_owned())])
                .expect("valid overrides");

        let mut map = HeaderMap::new();
        map.insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        overrides.apply_to_map(&mut map);

        let values: Vec<&str> = map
            .get_all("content-type")
            .iter()
            .filter_map(|v| v.to_str().ok())
            .collect();
        assert_eq!(values, vec!["application/xml"]);
    }

    #[test]
    fn invalid_header_name_is_rejected() {
        let error = HeaderOverrides::new(&[("bad name".to_owned(), "value".to_owned())])
            .expect_err("invalid name should fail");
        assert!(error.to_string().contains("invalid http header name"));
    }

    #[test]
    fn invalid_header_value_is_rejected() {
        let error = HeaderOverrides::new(&[("x-trace".to_owned(), "bad\nvalue".to_owned())])
            .expect_err("invalid value should fail");
        assert!(error.to_string().contains("invalid http header value"));
    }
}
