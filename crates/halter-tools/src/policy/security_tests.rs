// Adversarial policy tests. Each case encodes a concrete bypass attempt so
// that a regression fails loudly. See
// `docs/design-plans/2026-04-17-review-remediation-core.md` §Acceptance
// Criteria for the AC mapping.
//
// AC coverage:
//   Phase 1: AC1.1, AC1.2, AC1.3, AC1.4, AC1.5, AC1.8, AC1.12, AC1.14.
//   Phase 2: AC1.6, AC1.7, AC1.9 (policy half), AC1.13.
//   AC1.10 and AC1.11 (shell command parsing) were retired with the shell
//   allowlist.

use std::path::PathBuf;

use tempfile::TempDir;

use super::{DefaultToolPolicy, LoopbackAllow, PolicyError, PolicySettings, ToolPolicy};

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
    let _ = p.check_network("https://example.com").await;
    let _ = p.check_subagent_spawn_typed(0, 0).await;
}

// Denied by the default `**/.ssh/**` glob on every platform. `/etc/shadow`
// is not: Windows resolves it under the current drive (`D:/etc/shadow`).
const SENSITIVE_PATH: &str = "/home/user/.ssh/id_rsa";

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

// Unix-only: `/etc/shadow` is a Unix path.
#[tokio::test]
#[cfg(unix)]
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

// Unix-only: creating a symlink on Windows needs elevated privileges.
#[tokio::test]
#[cfg(unix)]
async fn ac1_5_symlink_suffix_escape_is_rejected() {
    let base = TempDir::new().expect("base tempdir");
    let outside = TempDir::new().expect("outside tempdir");

    // Write is only allowed under `base`. Attacker places a symlink under
    // `base` that points at `outside`; the canonicalize step should see
    // through the link and reject.
    let inside = base.path().join("allowed");
    std::fs::create_dir(&inside).unwrap();
    let link = inside.join("escape");
    std::os::unix::fs::symlink(outside.path(), &link).unwrap();

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

// ------------- AC1.8 / AC1.9 (policy half): the shell capability gate

#[tokio::test]
async fn ac1_8_shell_enabled_follows_the_policy_switch() {
    let policy = |shell_enabled| {
        DefaultToolPolicy::new(PolicySettings {
            shell_enabled,
            ..PolicySettings::default()
        })
    };
    policy(true)
        .check_shell_enabled()
        .await
        .expect("enabled shell is allowed");
    let err = policy(false)
        .check_shell_enabled()
        .await
        .expect_err("must deny when shell_enabled=false");
    assert!(matches!(err, PolicyError::ShellDisabled));
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
        allowed_loopback: vec![LoopbackAllow {
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
#[tokio::test]
async fn ac1_7_check_process_signal_rejects_pid_outside_tree() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        process_tree_root: Some(424242),
        ..PolicySettings::default()
    });

    let err = policy
        .check_process_signal(424243)
        .await
        .expect_err("pid outside configured process tree must be denied");
    assert!(matches!(
        err,
        PolicyError::ProcessOutsideTree { pid: 424243 }
    ));
}

// ------------- AC2: universal NetworkPolicy semantics

#[tokio::test]
async fn review_hook_runtime_ac2_1_loopback_denied_when_allowed_loopback_empty() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        network_enabled: true,
        ..PolicySettings::default()
    });
    let err = policy
        .check_network("http://127.0.0.1:8080/")
        .await
        .expect_err("loopback must be denied when allowed_loopback is empty");
    assert!(matches!(err, PolicyError::NetworkDenied { .. }));
}

#[tokio::test]
async fn review_hook_runtime_ac2_2_loopback_allowed_when_entry_matches() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        network_enabled: true,
        allowed_loopback: vec![LoopbackAllow {
            host: "127.0.0.1".to_owned(),
            port: None,
        }],
        ..PolicySettings::default()
    });
    policy
        .check_network("http://127.0.0.1:8080/")
        .await
        .expect("matching loopback entry should allow");
}

#[tokio::test]
async fn review_hook_runtime_ac2_3_loopback_denied_when_port_mismatches() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        network_enabled: true,
        allowed_loopback: vec![LoopbackAllow {
            host: "localhost".to_owned(),
            port: Some(9090),
        }],
        ..PolicySettings::default()
    });
    let err = policy
        .check_network("http://localhost:8080/")
        .await
        .expect_err("wrong port must be denied");
    assert!(matches!(err, PolicyError::NetworkDenied { .. }));
}

#[tokio::test]
async fn review_hook_runtime_ac2_4_loopback_allows_any_port_when_port_none() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        network_enabled: true,
        allowed_loopback: vec![LoopbackAllow {
            host: "localhost".to_owned(),
            port: None,
        }],
        ..PolicySettings::default()
    });
    policy
        .check_network("http://localhost:8080/")
        .await
        .expect("port None should allow any port");
    policy
        .check_network("http://localhost:9090/")
        .await
        .expect("port None should allow any port");
}

#[tokio::test]
async fn review_hook_runtime_ac2_5_default_star_allows_remote_host() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        network_enabled: true,
        ..PolicySettings::default()
    });
    policy
        .check_network("https://api.example.com/")
        .await
        .expect("default allowed_hosts=[\"*\"] should allow any remote host");
}

#[tokio::test]
async fn review_hook_runtime_ac2_6_narrowed_allowed_hosts_denies_mismatch() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        network_enabled: true,
        allowed_hosts: vec!["only.example.com".to_owned()],
        ..PolicySettings::default()
    });
    let err = policy
        .check_network("https://api.example.com/")
        .await
        .expect_err("narrowed allowlist must deny mismatched host");
    assert!(matches!(err, PolicyError::NetworkDenied { .. }));
}

#[tokio::test]
async fn review_hook_runtime_ac2_7_kill_switch_denies_everything() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        network_enabled: false,
        allowed_hosts: vec!["*".to_owned()],
        allowed_loopback: vec![LoopbackAllow {
            host: "localhost".to_owned(),
            port: None,
        }],
        ..PolicySettings::default()
    });
    for url in [
        "https://api.example.com/",
        "http://localhost:8080/",
        "http://127.0.0.1/",
    ] {
        let err = policy
            .check_network(url)
            .await
            .expect_err("kill switch must deny every request");
        assert!(matches!(err, PolicyError::NetworkDenied { .. }));
    }
}

// ------------- H33: blocking-variant write check mirrors the async one
//
// The ast/replace tool runs on a blocking worker and now invokes the policy
// immediately before the write. The sync `check_write_path_blocking` helper
// must produce the same accept/reject decisions and identical `CanonicalPath`
// output as the async `check_write_path` — otherwise the TOCTOU move could
// accidentally widen or narrow the enforced surface.

#[tokio::test]
async fn h33_check_write_path_blocking_accepts_what_async_accepts() {
    let dir = TempDir::new().expect("tempdir");
    let policy = tmp_policy(dir.path());
    let target = dir.path().join("sub").join("out.txt");
    std::fs::create_dir_all(target.parent().unwrap()).unwrap();

    let async_ok = policy
        .check_write_path(&target)
        .await
        .expect("async accepts write under allowed root");
    let sync_ok = policy
        .check_write_path_blocking(&target)
        .expect("sync mirror accepts the same path");
    assert_eq!(async_ok.path(), sync_ok.path());
}

#[test]
fn h33_check_write_path_blocking_rejects_sensitive_path() {
    let dir = TempDir::new().expect("tempdir");
    let policy = tmp_policy(dir.path());

    // A sensitive path is rejected by the glob before any I/O —
    // the sync mirror must reject it for the same reason.
    let err = policy
        .check_write_path_blocking(std::path::Path::new(SENSITIVE_PATH))
        .expect_err("sensitive path must be denied by sync mirror");
    assert!(matches!(err, PolicyError::SensitivePathDenied { .. }));
}

#[test]
#[cfg(unix)]
fn h33_check_write_path_blocking_rejects_symlink_escape() {
    let base = TempDir::new().expect("base tempdir");
    let outside = TempDir::new().expect("outside tempdir");
    let inside = base.path().join("allowed");
    std::fs::create_dir(&inside).unwrap();
    let link = inside.join("escape");
    std::os::unix::fs::symlink(outside.path(), &link).unwrap();

    let policy = DefaultToolPolicy::new(PolicySettings {
        allowed_write_roots: vec![base.path().to_path_buf()],
        allowed_read_roots: vec![base.path().to_path_buf()],
        ..PolicySettings::default()
    });

    let err = policy
        .check_write_path_blocking(&link.join("pwned.txt"))
        .expect_err("symlink traversal must be denied by sync mirror");
    assert!(matches!(
        err,
        PolicyError::NotInRoot { .. } | PolicyError::SymlinkEscape { .. }
    ));
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
            // Windows qualifies a rooted path with the current drive.
            assert_eq!(root, std::path::absolute(&ghost).unwrap());
        }
        other => panic!("expected NonexistentRoot, got {other:?}"),
    }
}

#[test]
fn ac1_14_only_the_builtin_scratch_root_is_created_on_demand() {
    let dir = TempDir::new().expect("tempdir");
    let missing = dir.path().join("missing");
    let scratch = super::default_write_root();

    let created = super::canonicalize_root_list(std::slice::from_ref(&scratch))
        .expect("the built-in scratch root is created when absent");
    assert_eq!(created, [std::fs::canonicalize(&scratch).unwrap()]);
    assert!(scratch.is_dir());

    match super::canonicalize_root_list(&[scratch, missing.clone()]) {
        Err(PolicyError::NonexistentRoot { root }) => assert_eq!(root, missing),
        other => panic!("a missing configured root must still fail, got {other:?}"),
    }
    assert!(!missing.exists(), "configured roots are never created");
}

// ------------- Issue #115: sensitive-path glob set cache

#[tokio::test]
async fn default_policy_compiles_lazily() {
    let policy = DefaultToolPolicy::new(PolicySettings::default());
    let err = policy
        .check_read_path(std::path::Path::new(SENSITIVE_PATH), 16)
        .await
        .expect_err("shadow read before initialization must be denied");
    assert!(matches!(err, PolicyError::SensitivePathDenied { .. }));
}

#[tokio::test]
async fn custom_sensitive_pattern_cached_and_enforced() {
    let dir = TempDir::new().expect("tempdir");
    let target_secret = dir.path().join(".custom_secret");
    std::fs::write(&target_secret, "secret").unwrap();
    let target_normal = dir.path().join("plain.txt");
    std::fs::write(&target_normal, "plain").unwrap();

    let policy = DefaultToolPolicy::new(PolicySettings {
        allowed_read_roots: vec![dir.path().to_path_buf()],
        allowed_write_roots: vec![dir.path().to_path_buf()],
        sensitive_path_patterns: vec!["**/.custom_secret".to_owned()],
        ..PolicySettings::default()
    });

    // First denial populates the cache.
    let err = policy
        .check_read_path(&target_secret, 6)
        .await
        .expect_err("custom sensitive file must be denied");
    assert!(matches!(err, PolicyError::SensitivePathDenied { .. }));

    // Subsequent checks reuse the cached glob set and still deny.
    for _ in 0..3 {
        let err = policy
            .check_read_path(&target_secret, 6)
            .await
            .expect_err("cached custom sensitive file must still be denied");
        assert!(matches!(err, PolicyError::SensitivePathDenied { .. }));
    }

    // Non-sensitive file is unaffected.
    policy
        .check_read_path(&target_normal, 5)
        .await
        .expect("non-sensitive file under allowed root must succeed");
}

#[tokio::test]
async fn empty_patterns_allow_all() {
    let dir = TempDir::new().expect("tempdir");
    let target = dir.path().join("anything.txt");
    std::fs::write(&target, "any").unwrap();

    let policy = DefaultToolPolicy::new(PolicySettings {
        allowed_read_roots: vec![dir.path().to_path_buf()],
        allowed_write_roots: vec![dir.path().to_path_buf()],
        sensitive_path_patterns: vec![],
        ..PolicySettings::default()
    });

    policy
        .check_read_path(&target, 3)
        .await
        .expect("empty sensitive patterns must not deny reads");
}

#[tokio::test]
async fn invalid_pattern_surfaces_on_check() {
    let policy = DefaultToolPolicy::new(PolicySettings {
        sensitive_path_patterns: vec!["[z-a".to_owned()],
        ..PolicySettings::default()
    });

    for _ in 0..2 {
        let err = policy
            .check_read_path(std::path::Path::new("/etc/shadow"), 16)
            .await
            .expect_err("malformed pattern must surface as invalid_pattern");
        assert!(
            matches!(
                err,
                PolicyError::SensitivePathDenied {
                    rule: "invalid_pattern",
                    ..
                }
            ),
            "expected invalid_pattern, got {err:?}"
        );
    }
}

#[tokio::test]
async fn clone_preserves_cache_semantics() {
    let dir = TempDir::new().expect("tempdir");
    let policy = DefaultToolPolicy::new(PolicySettings {
        allowed_read_roots: vec![dir.path().to_path_buf()],
        allowed_write_roots: vec![dir.path().to_path_buf()],
        ..PolicySettings::default()
    });

    // Clone before initialization; each instance lazily compiles independently.
    let cloned_uninit = policy.clone();
    let err = cloned_uninit
        .check_read_path(std::path::Path::new(SENSITIVE_PATH), 16)
        .await
        .expect_err("clone before init must still deny sensitive path");
    assert!(matches!(err, PolicyError::SensitivePathDenied { .. }));

    // Force initialization of the original policy.
    let _ = policy
        .check_read_path(std::path::Path::new(SENSITIVE_PATH), 16)
        .await;

    // Cloning an initialized policy copies the compiled glob set.
    let cloned_init = policy.clone();
    let err = cloned_init
        .check_read_path(std::path::Path::new(SENSITIVE_PATH), 16)
        .await
        .expect_err("clone after init must deny sensitive path");
    assert!(matches!(err, PolicyError::SensitivePathDenied { .. }));
}

#[tokio::test(flavor = "multi_thread")]
async fn concurrent_sensitive_checks_are_safe() {
    use std::sync::Arc;

    let policy = Arc::new(DefaultToolPolicy::new(PolicySettings::default()));
    let mut handles = Vec::new();

    for _ in 0..8 {
        let p = Arc::clone(&policy);
        handles.push(tokio::spawn(async move {
            p.check_read_path(std::path::Path::new(SENSITIVE_PATH), 16)
                .await
        }));
    }

    for handle in handles {
        let err = handle
            .await
            .expect("task joined")
            .expect_err("concurrent sensitive path must be denied");
        assert!(matches!(err, PolicyError::SensitivePathDenied { .. }));
    }
}

/// Bounded exhaustive check of URL host extraction: every authority of up to
/// six bytes drawn from the bytes the parser branches on. Any extracted host
/// must be non-empty and free of userinfo, path, query, and fragment
/// delimiters. Enumeration is used instead of a Kani proof because symbolic
/// execution of std's substring search exhausts memory at even four bytes.
#[test]
fn extracted_host_never_contains_url_delimiters_exhaustive() {
    const ALPHABET: &[u8] = b":/@[]?#a1";
    const MAX_AUTHORITY_LEN: usize = 6;

    let mut authority = String::with_capacity(MAX_AUTHORITY_LEN);
    for len in 0..=MAX_AUTHORITY_LEN {
        for mut code in 0..ALPHABET.len().pow(len as u32) {
            authority.clear();
            for _ in 0..len {
                authority.push(char::from(ALPHABET[code % ALPHABET.len()]));
                code /= ALPHABET.len();
            }
            let url = format!("h://{authority}");
            if let Some((host, _port)) = super::extract_host_and_port(&url) {
                assert!(!host.is_empty(), "empty host for {url:?}");
                assert!(
                    !host.contains(['/', '?', '#', '@']),
                    "host {host:?} contains a delimiter for {url:?}"
                );
            }
        }
    }
}
