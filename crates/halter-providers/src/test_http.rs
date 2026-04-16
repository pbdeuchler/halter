// pattern: Functional Core
//
// Shared test helpers for reading raw HTTP requests off a TcpStream inside
// provider tests. Not used by production code.

#![cfg(test)]

use tokio::io::AsyncReadExt;

pub(crate) async fn read_http_request(
    socket: &mut tokio::net::TcpStream,
) -> anyhow::Result<Vec<u8>> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 1024];

    loop {
        let read = socket.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(headers_end) = find_headers_end(&buffer) {
            let header_text = String::from_utf8_lossy(&buffer[..headers_end]);
            let content_length = header_text
                .lines()
                .find_map(|line| {
                    line.split_once(':').and_then(|(name, value)| {
                        name.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                })
                .unwrap_or(0);
            let body_bytes = buffer.len().saturating_sub(headers_end + 4);
            if body_bytes >= content_length {
                return Ok(buffer);
            }
        }
    }

    anyhow::bail!("incomplete http request")
}

pub(crate) fn find_headers_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}
