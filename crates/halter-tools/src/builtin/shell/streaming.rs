// pattern: Imperative Shell

use std::fs;
use std::io;
use std::str;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{ToolEventSink, ToolRuntimeEvent};

const REPLACEMENT: &str = "\u{FFFD}";
const BUFFER_SIZE: usize = 65_536;
const OUTPUT_CAP_BYTES: usize = 1024 * 1024;

fn truncation_notice() -> String {
    format!("\n[output truncated after {} bytes]\n", OUTPUT_CAP_BYTES)
}

pub async fn collect_output(
    reader: fs::File,
    emit: std::sync::Arc<dyn ToolEventSink>,
    tool_name: &'static str,
    cancel: CancellationToken,
    activity: mpsc::Sender<()>,
) -> anyhow::Result<String> {
    let mut collected = String::new();
    let mut buffer = vec![0u8; BUFFER_SIZE + 4];
    let mut pending = 0usize;
    let mut truncated = false;

    #[cfg(unix)]
    let reader = register_nonblocking_pipe(reader)?;
    #[cfg(not(unix))]
    let reader = tokio::fs::File::from_std(reader);
    #[cfg(not(unix))]
    tokio::pin!(reader);

    loop {
        debug_assert!(pending <= BUFFER_SIZE);

        #[cfg(unix)]
        let read = {
            let mut readiness = tokio::select! {
                ready = reader.readable() => ready,
                () = cancel.cancelled() => break,
            }?;
            match readiness.try_io(|inner| {
                read_nonblocking(inner.get_ref(), &mut buffer[pending..BUFFER_SIZE])
            }) {
                Ok(Ok(0)) => break,
                Ok(Ok(count)) => count,
                Ok(Err(error)) if error.kind() == io::ErrorKind::Interrupted => continue,
                Ok(Err(_)) => break,
                Err(_) => continue,
            }
        };

        #[cfg(not(unix))]
        let read = {
            use tokio::io::AsyncReadExt;

            let mut future = reader.read(&mut buffer[pending..BUFFER_SIZE]);
            tokio::pin!(future);
            match tokio::select! {
                value = &mut future => value,
                () = cancel.cancelled() => break,
            } {
                Ok(0) => break,
                Ok(count) => count,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        };

        if read > 0 {
            let _ = activity.try_send(());
        }
        pending += read;

        while pending > 0 {
            let available = &buffer[..pending];
            match str::from_utf8(available) {
                Ok(text) => {
                    emit_chunk(&mut collected, &emit, tool_name, text, &mut truncated);
                    pending = 0;
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0 {
                        let Some(valid_bytes) = available.get(..valid_up_to) else {
                            anyhow::bail!(
                                "shell streaming: invalid valid_up_to {valid_up_to} > available {}",
                                available.len()
                            );
                        };
                        let valid = str::from_utf8(valid_bytes).map_err(|err| {
                            anyhow::anyhow!("shell streaming: valid prefix failed to decode: {err}")
                        })?;
                        emit_chunk(&mut collected, &emit, tool_name, valid, &mut truncated);
                        buffer.copy_within(valid_up_to..pending, 0);
                        pending -= valid_up_to;
                    }

                    match error.error_len() {
                        Some(invalid_len) => {
                            emit_chunk(
                                &mut collected,
                                &emit,
                                tool_name,
                                REPLACEMENT,
                                &mut truncated,
                            );
                            buffer.copy_within(invalid_len..pending, 0);
                            pending -= invalid_len;
                        }
                        None => break,
                    }
                }
            }
        }
    }

    for chunk in buffer[..pending].utf8_chunks() {
        if !chunk.valid().is_empty() {
            emit_chunk(
                &mut collected,
                &emit,
                tool_name,
                chunk.valid(),
                &mut truncated,
            );
        }
        if !chunk.invalid().is_empty() {
            emit_chunk(
                &mut collected,
                &emit,
                tool_name,
                REPLACEMENT,
                &mut truncated,
            );
        }
    }

    Ok(collected)
}

fn emit_chunk(
    collected: &mut String,
    emit: &std::sync::Arc<dyn ToolEventSink>,
    tool_name: &'static str,
    chunk: &str,
    truncated: &mut bool,
) {
    if *truncated {
        return;
    }
    let original_len = chunk.len();
    let remaining = OUTPUT_CAP_BYTES.saturating_sub(collected.len());
    let emitted = truncate_to_char_boundary(chunk, remaining);
    if !emitted.is_empty() {
        collected.push_str(emitted);
        emit.emit(ToolRuntimeEvent::ToolOutput {
            tool_name: tool_name.to_owned(),
            chunk: emitted.to_owned(),
        });
    }

    if emitted.len() < original_len {
        let notice = truncation_notice();
        collected.push_str(&notice);
        emit.emit(ToolRuntimeEvent::ToolOutput {
            tool_name: tool_name.to_owned(),
            chunk: notice,
        });
        *truncated = true;
    }
}

fn truncate_to_char_boundary(value: &str, cap: usize) -> &str {
    if value.len() <= cap {
        return value;
    }
    let mut boundary = cap;
    while boundary > 0 && !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    &value[..boundary]
}

#[cfg(unix)]
fn register_nonblocking_pipe(reader: fs::File) -> io::Result<tokio::io::unix::AsyncFd<fs::File>> {
    set_nonblocking(&reader)?;
    tokio::io::unix::AsyncFd::new(reader)
}

#[cfg(unix)]
fn set_nonblocking<T: std::os::fd::AsRawFd>(file: &T) -> io::Result<()> {
    let fd = file.as_raw_fd();
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    if flags & libc::O_NONBLOCK != 0 {
        return Ok(());
    }
    let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn read_nonblocking<T: std::os::fd::AsRawFd>(file: &T, buffer: &mut [u8]) -> io::Result<usize> {
    let read = unsafe { libc::read(file.as_raw_fd(), buffer.as_mut_ptr().cast(), buffer.len()) };
    if read < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(read as usize)
    }
}

pub fn pipe_to_files(label: &str) -> anyhow::Result<(fs::File, fs::File)> {
    let (reader, writer) = os_pipe::pipe()
        .map_err(|error| anyhow::anyhow!("failed to create {label} pipe: {error}"))?;

    #[cfg(unix)]
    let (reader, writer): (fs::File, fs::File) = {
        use std::os::unix::io::{FromRawFd, IntoRawFd};
        let reader = reader.into_raw_fd();
        let writer = writer.into_raw_fd();
        unsafe {
            (
                FromRawFd::from_raw_fd(reader),
                FromRawFd::from_raw_fd(writer),
            )
        }
    };

    #[cfg(windows)]
    let (reader, writer): (fs::File, fs::File) = {
        use std::os::windows::io::{FromRawHandle, IntoRawHandle};
        let reader = reader.into_raw_handle();
        let writer = writer.into_raw_handle();
        unsafe {
            (
                FromRawHandle::from_raw_handle(reader),
                FromRawHandle::from_raw_handle(writer),
            )
        }
    };

    Ok((reader, writer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_notice_derived_from_output_cap() {
        let expected = format!("\n[output truncated after {} bytes]\n", OUTPUT_CAP_BYTES);
        assert_eq!(truncation_notice(), expected);
    }

    #[test]
    fn truncation_notice_contains_byte_count() {
        let notice = truncation_notice();
        assert!(!notice.is_empty());
        assert!(notice.contains(&OUTPUT_CAP_BYTES.to_string()));
    }
}
