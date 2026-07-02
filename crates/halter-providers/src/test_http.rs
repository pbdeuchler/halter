//! Shared HTTP fixture helpers for provider tests: reads one HTTP request
//! (headers plus `Content-Length` body) from a raw socket so test servers can
//! assert on captured bytes. Replaces the five per-module copies that had
//! drifted apart.

use tokio::io::AsyncReadExt;
use tokio::net::TcpStream;

/// Read a full HTTP request (through its `Content-Length` body) from the
/// socket and return the raw bytes. Fails if the peer closes the connection
/// before a complete request arrives.
pub(crate) async fn read_http_request(socket: &mut TcpStream) -> anyhow::Result<Vec<u8>> {
    let mut request = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        let read = socket.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        request.extend_from_slice(&chunk[..read]);
        if let Some(headers_end) = find_headers_end(&request) {
            let content_length = content_length(&request[..headers_end]).unwrap_or_default();
            if request.len() >= headers_end + 4 + content_length {
                return Ok(request);
            }
        }
    }
    anyhow::bail!("incomplete http request")
}

/// Byte offset of the `\r\n\r\n` header terminator, if present.
pub(crate) fn find_headers_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

/// Parse the `Content-Length` header from a raw header block.
pub(crate) fn content_length(headers: &[u8]) -> Option<usize> {
    let text = String::from_utf8_lossy(headers);
    text.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.trim()
            .eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse().ok())
            .flatten()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_headers_end_locates_terminator_or_none() {
        assert_eq!(
            find_headers_end(b"POST / HTTP/1.1\r\nA: b\r\n\r\nbody"),
            Some(21)
        );
        assert_eq!(find_headers_end(b"POST / HTTP/1.1\r\nA: b\r\n"), None);
    }

    #[test]
    fn content_length_parses_case_insensitively_or_none() {
        assert_eq!(
            content_length(b"POST / HTTP/1.1\r\nContent-Length: 12\r\n"),
            Some(12)
        );
        assert_eq!(
            content_length(b"POST / HTTP/1.1\r\ncontent-length:7\r\n"),
            Some(7)
        );
        assert_eq!(content_length(b"POST / HTTP/1.1\r\nHost: x\r\n"), None);
        assert_eq!(
            content_length(b"POST / HTTP/1.1\r\nContent-Length: nope\r\n"),
            None
        );
    }
}
