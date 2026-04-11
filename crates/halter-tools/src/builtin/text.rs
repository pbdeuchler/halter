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
    if visible_width(line) <= max_cols {
        return line.to_owned();
    }

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
            break;
        }
        output.push_str(grapheme);
        width += grapheme_width;
        index = next;
    }

    output
}

fn grapheme_width(grapheme: &str) -> usize {
    if grapheme == "\t" {
        return 4;
    }
    UnicodeWidthStr::width(grapheme)
}

fn next_grapheme_boundary(text: &str, start: usize) -> usize {
    UnicodeSegmentation::grapheme_indices(text, true)
        .map(|(index, _)| index)
        .find(|index| *index > start)
        .unwrap_or(text.len())
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

    #[test]
    fn visible_width_ignores_ansi_sequences() {
        assert_eq!(visible_width("\u{1b}[31mhello\u{1b}[0m"), 5);
    }

    #[test]
    fn truncate_to_width_preserves_ansi_sequences() {
        assert_eq!(truncate_to_width("\u{1b}[31mhello\u{1b}[0m", 3), "\u{1b}[31mhel");
    }
}
