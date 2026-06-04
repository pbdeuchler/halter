// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use serde_json::{Value, json};

use crate::{Tool, ToolContext};

use super::common::{ToolScope, ensure_not_cancelled, optional_u64, required_string};

#[cfg(target_os = "linux")]
mod platform {
    use std::collections::{HashMap, VecDeque};
    use std::fs;
    use std::path::Path;

    pub fn collect_descendants(pid: i32, pids: &mut Vec<i32>) {
        let tree = build_process_tree(Path::new("/proc"));
        collect_descendants_from_tree(pid, &tree, pids);
    }

    fn build_process_tree(root: &Path) -> HashMap<i32, Vec<i32>> {
        let mut tree = HashMap::<i32, Vec<i32>>::new();
        let Ok(entries) = fs::read_dir(root) else {
            return tree;
        };

        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let Some(pid) = file_name
                .to_str()
                .and_then(|value| value.parse::<i32>().ok())
            else {
                continue;
            };
            let Ok(status) = fs::read_to_string(entry.path().join("status")) else {
                continue;
            };
            if let Some(ppid) = parse_status_ppid(&status) {
                tree.entry(ppid).or_default().push(pid);
            }
        }
        tree
    }

    fn collect_descendants_from_tree(pid: i32, tree: &HashMap<i32, Vec<i32>>, pids: &mut Vec<i32>) {
        let mut queue = VecDeque::new();
        if let Some(children) = tree.get(&pid) {
            queue.extend(children.iter().copied());
        }

        while let Some(child_pid) = queue.pop_front() {
            pids.push(child_pid);
            if let Some(children) = tree.get(&child_pid) {
                queue.extend(children.iter().copied());
            }
        }
    }

    fn parse_status_ppid(status: &str) -> Option<i32> {
        status.lines().find_map(|line| {
            let value = line.strip_prefix("PPid:")?.trim();
            value.parse::<i32>().ok()
        })
    }

    pub fn kill_pid(pid: i32, signal: i32) -> bool {
        unsafe { libc::kill(pid, signal) == 0 }
    }

    pub fn process_group_id(pid: i32) -> Option<i32> {
        let pgid = unsafe { libc::getpgid(pid) };
        if pgid < 0 { None } else { Some(pgid) }
    }

    pub fn kill_process_group(pgid: i32, signal: i32) -> bool {
        unsafe { libc::kill(-pgid, signal) == 0 }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn parses_status_parent_pid() {
            assert_eq!(
                parse_status_ppid("Name:\tbash\nState:\tS\nPPid:\t42\n"),
                Some(42)
            );
            assert_eq!(parse_status_ppid("Name:\tbash\n"), None);
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::mem::size_of;
    use std::ptr;

    #[link(name = "proc", kind = "dylib")]
    unsafe extern "C" {
        fn proc_listchildpids(ppid: i32, buffer: *mut i32, buffersize: i32) -> i32;
    }

    pub fn collect_descendants(pid: i32, pids: &mut Vec<i32>) {
        let count = unsafe { proc_listchildpids(pid, ptr::null_mut(), 0) };
        if count <= 0 {
            return;
        }

        let mut buffer = vec![0i32; count as usize];
        let actual = unsafe {
            proc_listchildpids(
                pid,
                buffer.as_mut_ptr(),
                (buffer.len() * size_of::<i32>()) as i32,
            )
        };
        if actual <= 0 {
            return;
        }

        let child_count = actual as usize / size_of::<i32>();
        for &child_pid in &buffer[..child_count] {
            if child_pid > 0 {
                pids.push(child_pid);
                collect_descendants(child_pid, pids);
            }
        }
    }

    pub fn kill_pid(pid: i32, signal: i32) -> bool {
        unsafe { libc::kill(pid, signal) == 0 }
    }

    pub fn process_group_id(pid: i32) -> Option<i32> {
        let pgid = unsafe { libc::getpgid(pid) };
        if pgid < 0 { None } else { Some(pgid) }
    }

    pub fn kill_process_group(pgid: i32, signal: i32) -> bool {
        unsafe { libc::kill(-pgid, signal) == 0 }
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use std::collections::HashMap;
    use std::mem;

    use smallvec::SmallVec;

    #[repr(C)]
    #[allow(non_snake_case)]
    struct PROCESSENTRY32W {
        dwSize: u32,
        cntUsage: u32,
        th32ProcessID: u32,
        th32DefaultHeapID: usize,
        th32ModuleID: u32,
        cntThreads: u32,
        th32ParentProcessID: u32,
        pcPriClassBase: i32,
        dwFlags: u32,
        szExeFile: [u16; 260],
    }

    type Handle = *mut std::ffi::c_void;
    const INVALID_HANDLE_VALUE: Handle = -1isize as Handle;
    const TH32CS_SNAPPROCESS: u32 = 0x00000002;
    const PROCESS_TERMINATE: u32 = 0x0001;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateToolhelp32Snapshot(dwFlags: u32, th32ProcessID: u32) -> Handle;
        fn Process32FirstW(hSnapshot: Handle, lppe: *mut PROCESSENTRY32W) -> i32;
        fn Process32NextW(hSnapshot: Handle, lppe: *mut PROCESSENTRY32W) -> i32;
        fn CloseHandle(hObject: Handle) -> i32;
        fn OpenProcess(dwDesiredAccess: u32, bInheritHandle: i32, dwProcessId: u32) -> Handle;
        fn TerminateProcess(hProcess: Handle, uExitCode: u32) -> i32;
    }

    fn build_process_tree() -> HashMap<u32, SmallVec<[u32; 4]>> {
        let mut tree = HashMap::<u32, SmallVec<[u32; 4]>>::new();
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                return tree;
            }

            let mut entry: PROCESSENTRY32W = mem::zeroed();
            entry.dwSize = mem::size_of::<PROCESSENTRY32W>() as u32;

            if Process32FirstW(snapshot, &raw mut entry) != 0 {
                loop {
                    tree.entry(entry.th32ParentProcessID)
                        .or_default()
                        .push(entry.th32ProcessID);
                    if Process32NextW(snapshot, &raw mut entry) == 0 {
                        break;
                    }
                }
            }

            CloseHandle(snapshot);
        }
        tree
    }

    pub fn collect_descendants(pid: i32, pids: &mut Vec<i32>) {
        collect_descendants_from_tree(pid as u32, &build_process_tree(), pids);
    }

    fn collect_descendants_from_tree(
        pid: u32,
        tree: &HashMap<u32, SmallVec<[u32; 4]>>,
        pids: &mut Vec<i32>,
    ) {
        if let Some(children) = tree.get(&pid) {
            for &child_pid in children {
                pids.push(child_pid as i32);
                collect_descendants_from_tree(child_pid, tree, pids);
            }
        }
    }

    pub fn kill_pid(pid: i32, _signal: i32) -> bool {
        unsafe {
            let handle = OpenProcess(PROCESS_TERMINATE, 0, pid as u32);
            if handle.is_null() || handle == INVALID_HANDLE_VALUE {
                return false;
            }
            let result = TerminateProcess(handle, 1);
            CloseHandle(handle);
            result != 0
        }
    }

    pub const fn process_group_id(_pid: i32) -> Option<i32> {
        None
    }

    pub const fn kill_process_group(_pgid: i32, _signal: i32) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Per-process result returned by [`kill_tree`].
pub struct KillTreeEntry {
    pub pid: i32,
    pub killed: bool,
}

/// Kills `pid` and all descendants with `signal`. Returns per-PID results
/// (descendants first, target last). Useful for debugging hung processes
/// where a single count hides which PID refused the signal (finding M41).
pub fn kill_tree(pid: i32, signal: i32) -> Vec<KillTreeEntry> {
    let mut descendants = Vec::new();
    platform::collect_descendants(pid, &mut descendants);

    let mut report = Vec::with_capacity(descendants.len() + 1);
    for &child_pid in descendants.iter().rev() {
        report.push(KillTreeEntry {
            pid: child_pid,
            killed: platform::kill_pid(child_pid, signal),
        });
    }
    report.push(KillTreeEntry {
        pid,
        killed: platform::kill_pid(pid, signal),
    });
    report
}

#[cfg_attr(not(feature = "pty"), allow(dead_code))]
/// Return the process group id for a pid when the platform supports it.
pub fn process_group_id(pid: i32) -> Option<i32> {
    platform::process_group_id(pid)
}

#[cfg_attr(not(feature = "pty"), allow(dead_code))]
/// Send a signal to a process group when the platform supports it.
pub fn kill_process_group(pgid: i32, signal: i32) -> bool {
    platform::kill_process_group(pgid, signal)
}

/// List descendants of a process id.
pub fn list_descendants(pid: i32) -> Vec<i32> {
    let mut descendants = Vec::new();
    platform::collect_descendants(pid, &mut descendants);
    descendants
}

#[derive(Debug)]
/// Built-in tool for inspecting or terminating process trees.
pub struct ProcessTool;

#[async_trait]
impl Tool for ProcessTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("process"),
            description: "Inspect or terminate a process tree".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["kill_tree", "list_descendants"] },
                    "pid": { "type": "integer", "minimum": 1 },
                    "signal": { "type": "integer", "minimum": 1, "default": 9 }
                },
                "required": ["action", "pid"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: true,
                requires_approval: true,
                cancellable: false,
                long_running: false,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "process");
        ensure_not_cancelled(&context.cancel)?;
        let action = required_string(&input, "action")?;
        let pid = optional_u64(&input, "pid")?
            .ok_or_else(|| anyhow::anyhow!("invalid tool input: missing u64 field 'pid'"))?;
        let pid = i32::try_from(pid)
            .map_err(|_| anyhow::anyhow!("failed to execute process tool: pid is out of range"))?;

        let value = match action {
            "kill_tree" => {
                context.policy.check_process_signal(pid).await?;
                let signal = optional_u64(&input, "signal")?.unwrap_or(9);
                let signal = i32::try_from(signal).map_err(|_| {
                    anyhow::anyhow!("failed to execute process tool: signal is out of range")
                })?;
                let report = kill_tree(pid, signal);
                let killed_count = report.iter().filter(|entry| entry.killed).count() as u64;
                let per_pid: Vec<Value> = report
                    .iter()
                    .map(|entry| {
                        json!({
                            "pid": entry.pid,
                            "killed": entry.killed,
                        })
                    })
                    .collect();
                json!({
                    "pid": pid,
                    "signal": signal,
                    "killed": killed_count,
                    "per_pid": per_pid,
                })
            }
            "list_descendants" => json!({
                "pid": pid,
                "descendants": list_descendants(pid),
            }),
            _ => anyhow::bail!("failed to execute process tool: unknown action '{action}'"),
        };

        Ok(ToolResult::Json { value })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{
        DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolPolicy,
        ToolSessionStore,
    };

    use super::*;

    fn tool_context(root: &std::path::Path, allowed_shell_commands: Vec<String>) -> ToolContext {
        ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: root.to_path_buf(),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
                allowed_write_roots: vec![root.to_path_buf()],
                allowed_shell_commands,
                ..PolicySettings::default()
            })) as Arc<dyn ToolPolicy>,
            shell_timeout_secs: 30,
            subagent_parent: None,
        }
    }

    #[tokio::test]
    async fn kill_tree_rejects_pids_outside_session_tree() {
        // Targeting init / kernel pids must always be denied. The session
        // tree boundary lives in the policy and shows up as the typed
        // `ProcessOutsideTree` error.
        let temp = tempfile::tempdir().expect("tempdir");
        let error = ProcessTool
            .execute(
                tool_context(temp.path(), vec!["rg".to_owned()]),
                json!({
                    "action": "kill_tree",
                    "pid": 1
                }),
            )
            .await
            .expect_err("kill_tree of init must be denied");

        assert!(
            error.to_string().contains("outside the session"),
            "expected ProcessOutsideTree, got: {error}"
        );
    }
}
