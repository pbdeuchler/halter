//! Process management utilities

pub(crate) type ProcessId = i32;
pub(crate) use tokio::process::Child;

pub(crate) fn spawn(command: std::process::Command) -> std::io::Result<Child> {
    let mut command = tokio::process::Command::from(command);
    // Ensure abandoned children (e.g., after cancellation) are killed rather
    // than leaked when their handle is dropped.
    command.kill_on_drop(true);
    command.spawn()
}
