// pattern: Functional Core

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

const ESC: u8 = 0x1b;

#[must_use]
pub fn visible_width(line: &str) -> usize {
    let mut width = 0usize;
    let mut index = 0usize;
    let bytes = line.as_bytes();
    while index < bytes.len() {
        if let Some(end) = ansi_sequence_end(bytes, index) {
            index = end;
            continue;
        }

        let next = next_grapheme_boundary(line, index);
        let grapheme = &line[index..next];
        width += grapheme_width(grapheme);
        index = next;
    }
    width
}

#[must_use]
pub fn truncate_to_width(line: &str, max_cols: usize) -> String {
    let mut output = String::new();
    let mut width = 0usize;
    let mut index = 0usize;
    let bytes = line.as_bytes();

    while index < bytes.len() {
        if let Some(end) = ansi_sequence_end(bytes, index) {
            output.push_str(&line[index..end]);
            index = end;
            continue;
        }

        let next = next_grapheme_boundary(line, index);
        let grapheme = &line[index..next];
        let grapheme_width = grapheme_width(grapheme);
        if width + grapheme_width > max_cols {
            return output;
        }
        output.push_str(grapheme);
        width += grapheme_width;
        index = next;
    }

    // Every byte fit, so `output` is a copy of `line`.
    output
}

fn grapheme_width(grapheme: &str) -> usize {
    if grapheme == "\t" {
        return 4;
    }
    UnicodeWidthStr::width(grapheme)
}

/// Returns the end of the grapheme that starts at `start`.
///
/// Segments only from `start`, so a full left-to-right walk stays linear in the
/// line length. `start` must be a char boundary; callers only pass 0, a previous
/// grapheme end, or the end of an ANSI sequence (whose final byte is ASCII).
fn next_grapheme_boundary(text: &str, start: usize) -> usize {
    text[start..]
        .graphemes(true)
        .next()
        .map_or(text.len(), |grapheme| start + grapheme.len())
}

fn ansi_sequence_end(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start).copied() != Some(ESC) || bytes.get(start + 1).copied() != Some(b'[') {
        return None;
    }

    let mut index = start + 2;
    while index < bytes.len() {
        if (0x40..=0x7e).contains(&bytes[index]) {
            return Some(index + 1);
        }
        index += 1;
    }
    Some(bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn visible_width_ignores_ansi_sequences() {
        assert_eq!(visible_width("\u{1b}[31mhello\u{1b}[0m"), 5);
    }

    #[test]
    fn truncate_to_width_preserves_ansi_sequences() {
        assert_eq!(
            truncate_to_width("\u{1b}[31mhello\u{1b}[0m", 3),
            "\u{1b}[31mhel"
        );
    }

    #[test]
    fn widths_follow_grapheme_clusters() {
        // Combining mark, ZWJ family emoji (width 2), CJK (width 2 each), tab (4).
        let line = "e\u{301}\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\t\u{65E5}\u{672C}";
        assert_eq!(visible_width(line), 1 + 2 + 4 + 2 + 2);
        assert_eq!(truncate_to_width(line, 1), "e\u{301}");
        assert_eq!(truncate_to_width(line, 2), "e\u{301}");
        assert_eq!(
            truncate_to_width(line, 3),
            "e\u{301}\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}"
        );
        assert_eq!(truncate_to_width(line, 11), line);
        assert_eq!(truncate_to_width("", 0), "");
    }

    #[test]
    fn unterminated_ansi_sequence_has_no_width() {
        assert_eq!(visible_width("ab\u{1b}[3"), 2);
        assert_eq!(truncate_to_width("ab\u{1b}[3", 2), "ab\u{1b}[3");
    }

    /// Regression: grapheme lookup used to re-segment from byte 0 on every call,
    /// making a single multi-megabyte line (minified JS, one-line JSON) cost hours
    /// of CPU inside `grep` with `max_columns` set.
    #[test]
    fn long_single_line_is_linear() {
        let line = "{\"name\":\"index-serving\",\"x\":1},".repeat(64 * 1024);
        assert!(line.len() > 2_000_000);
        assert_eq!(visible_width(&line), line.len());
        assert_eq!(truncate_to_width(&line, 300).len(), 300);
    }

    proptest! {
        #[test]
        fn visible_width_matches_full_string_segmentation(text in "[a-z \t\u{301}\u{200D}\u{1F468}\u{65E5}\u{1F1FA}\u{1F1F8}]{0,64}") {
            let expected: usize = text.graphemes(true).map(grapheme_width).sum();
            prop_assert_eq!(visible_width(&text), expected);
        }

        #[test]
        fn truncation_is_the_longest_prefix_that_fits(
            text in "[a-z \t\u{301}\u{200D}\u{1F468}\u{65E5}\u{1B}\\[0-9m]{0,64}",
            max_cols in 0usize..80,
        ) {
            let truncated = truncate_to_width(&text, max_cols);
            prop_assert!(text.starts_with(&truncated));
            prop_assert!(visible_width(&truncated) <= max_cols);
            if truncated.len() < text.len() {
                let next = next_grapheme_boundary(&text, truncated.len());
                prop_assert!(visible_width(&text[..next]) > max_cols);
            }
        }
    }
}
