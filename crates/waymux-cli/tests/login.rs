// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for `waymux login` and the `--remote` flag's credential
//! requirement. These shell out to the built `waymux` binary so they exercise
//! argument parsing exactly as a user would.

use std::os::unix::fs::PermissionsExt;
use std::process::Command;

fn waymux_bin() -> &'static str {
    // Cargo populates this env var for any integration test in the same
    // package as the binary.
    env!("CARGO_BIN_EXE_waymux")
}

#[test]
fn login_writes_creds_file_with_0600() {
    let dir = tempfile::tempdir().unwrap();
    let creds_path = dir.path().join("nested").join("credentials.toml");

    let out = Command::new(waymux_bin())
        .args([
            "login",
            "--api-key",
            "wmx_secrettoken12345",
            "--base-url",
            "https://example.test",
        ])
        .env("WAYMUX_CREDENTIALS", &creds_path)
        // Make sure no real config gets touched.
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("WAYMUX_REMOTE")
        .output()
        .expect("spawn waymux");

    assert!(
        out.status.success(),
        "waymux login failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    // Redacted prefix should appear; full key must not.
    assert!(
        stdout.contains("wmx_secr"),
        "expected redacted prefix in stdout: {stdout}"
    );
    assert!(
        !stdout.contains("wmx_secrettoken12345"),
        "stdout leaked full key: {stdout}"
    );
    assert!(stdout.contains("https://example.test"));

    // File exists, perms are 0600.
    let meta = std::fs::metadata(&creds_path).expect("creds file exists");
    let mode = meta.permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "creds file mode = {:o}", mode);

    // Parent dir is 0700.
    let parent_meta = std::fs::metadata(creds_path.parent().unwrap()).unwrap();
    let parent_mode = parent_meta.permissions().mode() & 0o777;
    assert_eq!(parent_mode, 0o700, "parent dir mode = {:o}", parent_mode);

    // Round-trip: load and verify content.
    let body = std::fs::read_to_string(&creds_path).unwrap();
    assert!(body.contains("wmx_secrettoken12345"));
    assert!(body.contains("https://example.test"));
    assert!(body.contains("[default]"));
}

#[test]
fn login_rejects_missing_api_key() {
    let dir = tempfile::tempdir().unwrap();
    let creds_path = dir.path().join("creds.toml");

    let out = Command::new(waymux_bin())
        .args(["login", "--base-url", "https://example.test"])
        .env("WAYMUX_CREDENTIALS", &creds_path)
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("WAYMUX_REMOTE")
        .output()
        .expect("spawn waymux");

    assert!(
        !out.status.success(),
        "expected nonzero exit when --api-key missing"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("api-key") || stderr.contains("api_key") || stderr.contains("API"),
        "stderr should hint at api-key: {stderr}"
    );
    assert!(
        !creds_path.exists(),
        "creds file must not be written on failure"
    );
}

#[test]
fn remote_subcommand_without_creds_errors() {
    let dir = tempfile::tempdir().unwrap();
    let creds_path = dir.path().join("creds.toml");
    assert!(!creds_path.exists());

    let out = Command::new(waymux_bin())
        .args(["--remote", "ls"])
        .env("WAYMUX_CREDENTIALS", &creds_path)
        .env_remove("XDG_CONFIG_HOME")
        .env_remove("WAYMUX_REMOTE")
        .output()
        .expect("spawn waymux");

    assert!(
        !out.status.success(),
        "expected nonzero exit when no creds and --remote"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("waymux login"),
        "stderr should tell user to log in: {stderr}"
    );
}
