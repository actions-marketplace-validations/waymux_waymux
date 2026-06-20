// SPDX-License-Identifier: Apache-2.0

//! Hermetic end-to-end test: drive `tools/call` against a FAKE `waymux` binary
//! (a tiny shell script set via `WAYMUX_BIN`) that echoes a known `--json`
//! envelope. No real daemon, no network. Asserts the MCP response shape for
//! both a success and an error envelope.
//!
//! The fake script inspects its arguments to decide which envelope to emit, so
//! the same script serves both cases. It also verifies (indirectly) that the
//! server invokes the binary with `--json` as the first argument.

use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::{json, Value};
use waymux_mcp::exec::run_tool;
use waymux_mcp::registry::find_tool;

/// Tests in this file mutate the process-global `WAYMUX_BIN` env var, so they
/// must not run concurrently. Serialize them through this mutex.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Write a fake `waymux` shell script to a temp dir and return its path. The
/// script asserts `$1 == --json`, then branches on the verb to emit a fixed
/// envelope on stdout with the matching exit code.
fn write_fake_waymux(dir: &std::path::Path) -> PathBuf {
    let path = dir.join("waymux");
    let mut f = std::fs::File::create(&path).unwrap();
    // POSIX sh. `$1` must be --json (the server always passes it first).
    let script = r#"#!/bin/sh
if [ "$1" != "--json" ]; then
  echo "FAKE: expected --json as first arg, got: $*" >&2
  exit 99
fi
shift
verb="$1"
sub="$2"
case "$verb" in
  ls)
    echo '{"ok":true,"verb":"ls","data":{"sessions":[{"name":"eagle"}]}}'
    exit 0
    ;;
  info)
    # Emulate a not-found error (non-zero exit + error envelope on stdout).
    echo '{"ok":false,"verb":"info","error":{"code":"E_NOT_FOUND","message":"no such session: ghost","detail":null}}'
    exit 1
    ;;
  screenshot)
    echo '{"ok":true,"verb":"screenshot","data":{"width":2,"height":2,"png_b64":"QUJD"}}'
    exit 0
    ;;
  inject)
    # A success verb that prints nothing under --json.
    exit 0
    ;;
  *)
    echo "{\"ok\":true,\"verb\":\"$verb\",\"data\":{\"echoed_verb\":\"$verb\",\"echoed_sub\":\"$sub\"}}"
    exit 0
    ;;
esac
"#;
    f.write_all(script.as_bytes()).unwrap();
    drop(f);
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

/// Write a fake `waymux` that DUMPS every argv element it receives, one per
/// line, into the file named by `$WAYMUX_ARGV_DUMP`. This lets a test assert
/// the EXACT argv the MCP exec layer handed to the binary, proving values are
/// passed verbatim (no shell evaluation, no flag re-interpretation).
///
/// It then emits a trivial success envelope so `run_tool` is happy. Crucially
/// the dump uses `printf '%s\n'` over `"$@"`, so each argv element is one line
/// regardless of spaces, semicolons, quotes, `$(...)`, backticks, or pipes:
/// none of which a shell could expand here because they arrive as literal
/// argv strings, never as a command string.
fn write_argv_dumping_waymux(dir: &std::path::Path) -> PathBuf {
    let path = dir.join("waymux");
    let mut f = std::fs::File::create(&path).unwrap();
    let script = r#"#!/bin/sh
# Record every argument verbatim, one per line, into the dump file.
: > "$WAYMUX_ARGV_DUMP"
for a in "$@"; do
  printf '%s\n' "$a" >> "$WAYMUX_ARGV_DUMP"
done
# Emit a minimal success envelope so the exec layer maps it to an ok outcome.
echo '{"ok":true,"verb":"spawn","data":{}}'
exit 0
"#;
    f.write_all(script.as_bytes()).unwrap();
    drop(f);
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&path, perms).unwrap();
    path
}

fn args(v: Value) -> serde_json::Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn tools_call_success_against_fake_cli() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let fake = write_fake_waymux(tmp.path());
    std::env::set_var("WAYMUX_BIN", &fake);

    let spec = find_tool("waymux_ls").unwrap();
    let out = run_tool(spec, args(json!({})));

    assert!(!out.is_error, "ls should succeed");
    let structured = out.structured.expect("structured content present");
    assert_eq!(structured["sessions"][0]["name"], json!("eagle"));

    std::env::remove_var("WAYMUX_BIN");
}

#[test]
fn tools_call_error_against_fake_cli() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let fake = write_fake_waymux(tmp.path());
    std::env::set_var("WAYMUX_BIN", &fake);

    let spec = find_tool("waymux_info").unwrap();
    let out = run_tool(spec, args(json!({ "name": "ghost" })));

    assert!(out.is_error, "info on a missing session is an error");
    let text = out.content[0]["text"].as_str().unwrap();
    assert!(text.contains("E_NOT_FOUND"), "{text}");
    assert!(text.contains("no such session: ghost"), "{text}");

    std::env::remove_var("WAYMUX_BIN");
}

#[test]
fn tools_call_screenshot_returns_image_block_against_fake_cli() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let fake = write_fake_waymux(tmp.path());
    std::env::set_var("WAYMUX_BIN", &fake);

    let spec = find_tool("waymux_screenshot").unwrap();
    // No `output` supplied: the placeholder injection should make clap-equiv
    // happy in the real CLI; the fake ignores it.
    let out = run_tool(spec, args(json!({ "name": "eagle", "window_id": 1 })));

    assert!(!out.is_error);
    assert!(out
        .content
        .iter()
        .any(|c| c["type"] == "image" && c["data"] == json!("QUJD")));

    std::env::remove_var("WAYMUX_BIN");
}

#[test]
fn tools_call_empty_stdout_success_is_ok() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let fake = write_fake_waymux(tmp.path());
    std::env::set_var("WAYMUX_BIN", &fake);

    let spec = find_tool("waymux_inject").unwrap();
    let out = run_tool(spec, args(json!({ "name": "eagle", "ops": "[]" })));

    assert!(!out.is_error, "inject with empty stdout + exit 0 is ok");

    std::env::remove_var("WAYMUX_BIN");
}

#[test]
fn record_start_uses_two_word_argv_against_fake_cli() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let fake = write_fake_waymux(tmp.path());
    std::env::set_var("WAYMUX_BIN", &fake);

    let spec = find_tool("waymux_record_start").unwrap();
    let out = run_tool(spec, args(json!({ "name": "eagle" })));

    assert!(!out.is_error);
    let structured = out.structured.expect("structured present");
    // The fake echoes verb=record sub=start, proving the two-word argv path.
    assert_eq!(structured["echoed_verb"], json!("record"));
    assert_eq!(structured["echoed_sub"], json!("start"));

    std::env::remove_var("WAYMUX_BIN");
}

/// SECURITY REGRESSION TEST: the property the whole MCP model rests on.
///
/// Run `tools/call` with pathological argument VALUES (shell metacharacters,
/// command substitution, pipes, quotes, embedded newlines, and a `--`-looking
/// value) and assert that the fake `waymux` receives each one as a SINGLE,
/// VERBATIM argv element: never shell-evaluated, never split, and a
/// `--`-looking value never treated as a flag. This locks in the no-shell
/// guarantee: `std::process::Command` passes argv elements straight to
/// `execve`, so none of these can ever be interpreted.
#[test]
fn arguments_pass_through_verbatim_no_shell() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let tmp = tempfile::tempdir().unwrap();
    let fake = write_argv_dumping_waymux(tmp.path());
    let dump = tmp.path().join("argv_dump.txt");
    std::env::set_var("WAYMUX_BIN", &fake);
    std::env::set_var("WAYMUX_ARGV_DUMP", &dump);

    // A canary file that command-substitution / a subshell would create if any
    // of these values were ever evaluated by a shell. It MUST NOT appear.
    let canary = tmp.path().join("pwned");

    // Pathological values. `spawn`'s argv is an array of trailing elements, so
    // every one of these lands as a discrete trailing argv element after `--`.
    let payloads = vec![
        "; echo pwned".to_string(),
        format!("$(touch {})", canary.display()),
        "`touch /tmp/should_not_exist_waymux_test`".to_string(),
        "a | b".to_string(),
        "a && b".to_string(),
        "\"quoted\" 'single'".to_string(),
        "line1\nline2".to_string(),
        "--this-looks-like-a-flag".to_string(),
        "--app=https://example.com/?q=$(id)".to_string(),
        "normal-value".to_string(),
    ];

    let spec = find_tool("waymux_spawn").unwrap();
    let out = run_tool(spec, args(json!({ "name": "s", "argv": payloads.clone() })));
    assert!(!out.is_error, "spawn against dumping fake should succeed");

    // Canary must not exist: nothing was shell-evaluated.
    assert!(
        !canary.exists(),
        "command substitution canary file was created: a value was shell-evaluated!"
    );

    // Read back the EXACT argv the binary received.
    let dumped = std::fs::read_to_string(&dump).expect("argv dump written");
    // Each argv element is one line. Embedded newlines in a payload would split
    // across multiple lines in the dump, so we reconstruct by matching the
    // dump's tail against the expected full argv.
    //
    // Expected full argv (after the binary name): --json spawn s -- <payloads...>
    let mut expected: Vec<String> = vec![
        "--json".to_string(),
        "spawn".to_string(),
        "s".to_string(),
        "--".to_string(),
    ];
    expected.extend(payloads.iter().cloned());

    // The dump is one line per argv element; join with '\n' to reproduce the
    // exact byte stream and compare element-by-element. Because a payload may
    // itself contain '\n', compare the joined forms rather than line counts.
    let expected_joined = expected.join("\n");
    let dumped_trimmed = dumped.strip_suffix('\n').unwrap_or(&dumped);
    assert_eq!(
        dumped_trimmed, expected_joined,
        "argv did not pass through verbatim.\n got: {dumped_trimmed:?}\nwant: {expected_joined:?}"
    );

    // Spot-check the two most important properties explicitly:
    // 1) the `--`-looking value is present as data, NOT consumed as a flag.
    assert!(
        expected
            .iter()
            .filter(|e| *e == "--this-looks-like-a-flag")
            .count()
            == 1
            && dumped.contains("--this-looks-like-a-flag"),
        "the --flag-looking value must survive as a literal argv element"
    );
    // 2) the command-substitution value is present LITERALLY (unexpanded).
    assert!(
        dumped.contains(&format!("$(touch {})", canary.display())),
        "command-substitution value must appear literally, unexpanded"
    );

    std::env::remove_var("WAYMUX_ARGV_DUMP");
    std::env::remove_var("WAYMUX_BIN");
}
