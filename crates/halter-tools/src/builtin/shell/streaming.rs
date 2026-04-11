// pattern: Imperative Shell

use std::fs;
use std::io;
use std::str;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::{ToolEventSink, ToolRuntimeEvent};

const REPLACEMENT: &str = "\u{FFFD}";
const BUFFER_SIZE: usize = 65_536;

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

    #[cfg(unix)]
    let reader = register_nonblocking_pipe(reader)?;
    #[cfg(not(unix))]
    let reader = tokio::fs::File::from_std(reader);
    #[cfg(not(unix))]
    tokio::pin!(reader);

    loop {
        #[cfg(unix)]
        let read = {
            let mut readiness = tokio::select! {
                ready = reader.readable() => ready,
                () = cancel.cancelled() => break,
            }?;
            match readiness.try_io(|inner| read_nonblocking(inner.get_ref(), &mut buffer[pending..BUFFER_SIZE])) {
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
                    emit_chunk(&mut collected, &emit, tool_name, text);
                    pending = 0;
                    break;
                }
                Err(error) => {
                    let valid_up_to = error.valid_up_to();
                    if valid_up_to > 0 {
                        let valid = unsafe { str::from_utf8_unchecked(&available[..valid_up_to]) };
                        emit_chunk(&mut collected, &emit, tool_name, valid);
                        buffer.copy_within(valid_up_to..pending, 0);
                        pending -= valid_up_to;
                    }

                    match error.error_len() {
                        Some(invalid_len) => {
                            emit_chunk(&mut collected, &emit, tool_name, REPLACEMENT);
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
            emit_chunk(&mut collected, &emit, tool_name, chunk.valid());
        }
        if !chunk.invalid().is_empty() {
            emit_chunk(&mut collected, &emit, tool_name, REPLACEMENT);
        }
    }

    Ok(collected)
}

fn emit_chunk(
    collected: &mut String,
    emit: &std::sync::Arc<dyn ToolEventSink>,
    tool_name: &'static str,
    chunk: &str,
) {
    collected.push_str(chunk);
    emit.emit(ToolRuntimeEvent::ToolOutput {
        tool_name: tool_name.to_owned(),
        chunk: chunk.to_owned(),
    });
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
    let (reader, writer) =
        os_pipe::pipe().map_err(|error| anyhow::anyhow!("failed to create {label} pipe: {error}"))?;

    #[cfg(unix)]
    let (reader, writer): (fs::File, fs::File) = {
        use std::os::unix::io::{FromRawFd, IntoRawFd};
        let reader = reader.into_raw_fd();
        let writer = writer.into_raw_fd();
        unsafe { (FromRawFd::from_raw_fd(reader), FromRawFd::from_raw_fd(writer)) }
    };

    #[cfg(windows)]
    let (reader, writer): (fs::File, fs::File) = {
        use std::os::windows::io::{FromRawHandle, IntoRawHandle};
        let reader = reader.into_raw_handle();
        let writer = writer.into_raw_handle();
        unsafe { (FromRawHandle::from_raw_handle(reader), FromRawHandle::from_raw_handle(writer)) }
    };

    Ok((reader, writer))
}
