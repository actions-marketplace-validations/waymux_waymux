// SPDX-License-Identifier: Apache-2.0

//! The tool registry: the SINGLE source of truth mapping MCP tools to discrete
//! `waymux` CLI verbs, plus each tool's JSON-Schema `inputSchema`.
//!
//! Naming scheme: every tool is `waymux_<verb>` where `<verb>` is the CLI verb
//! with non-alphanumeric characters replaced by `_`. So the CLI verb
//! `screenshot-desktop` becomes the tool `waymux_screenshot_desktop`, and the
//! two-word subcommands `record start` / `viewer status` become
//! `waymux_record_start` / `waymux_viewer_status`.
//!
//! A tool's `argv` is the CLI argument vector AFTER the global `--json` flag and
//! BEFORE per-call argument substitution. For `record start` it is
//! `["record", "start"]`; for `ls` it is `["ls"]`. The exec layer prepends the
//! binary path and `--json`, then appends the positional + flag args derived
//! from the tool's parameter spec. Because every value is appended as a
//! discrete argv element (never concatenated into a shell string), there is no
//! shell-injection surface (see `exec.rs`).
//!
//! Keeping the verb argv, the parameter list, and the schema in ONE table is
//! deliberate: a contract test enumerates this registry and asserts every
//! discrete CLI verb is covered, so the MCP surface cannot silently drift from
//! the CLI.

use serde_json::{json, Map, Value};

/// How a tool parameter is rendered into the CLI argv.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// A positional argument: its value is appended directly to the argv in
    /// declaration order (after any earlier positionals).
    Positional,
    /// A `--flag value` option: appended as `--<cli-name>` followed by the
    /// stringified value.
    Flag,
    /// A boolean `--flag` switch: appended as `--<cli-name>` only when the
    /// value is `true`; omitted entirely when `false` or absent.
    BoolFlag,
}

/// The JSON-Schema scalar type a parameter accepts. Drives both the emitted
/// `inputSchema` and the argv stringification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamType {
    StringT,
    IntegerT,
    NumberT,
    BooleanT,
    /// A JSON value passed through verbatim as a string (e.g. `--ops` takes a
    /// JSON array string). Rendered in the schema as `type: string` with a note
    /// that it must contain JSON.
    JsonStringT,
    /// An array of strings, rendered as repeated positional args (used by
    /// `tag`'s trailing `tags...`).
    StringArrayT,
}

/// One tool parameter: a CLI arg/flag the tool exposes.
#[derive(Clone, Debug)]
pub struct Param {
    /// JSON-Schema property name (also the MCP argument name the client sends).
    pub name: &'static str,
    /// The CLI flag name WITHOUT leading dashes, for `Flag`/`BoolFlag` kinds.
    /// Ignored for positionals.
    pub cli_name: &'static str,
    pub kind: ParamKind,
    pub ty: ParamType,
    pub required: bool,
    pub description: &'static str,
}

/// One MCP tool = one discrete CLI verb.
#[derive(Clone, Debug)]
pub struct ToolSpec {
    /// MCP tool name, e.g. `waymux_ls`.
    pub name: &'static str,
    /// The canonical CLI verb string this tool maps to (matches the CLI's
    /// envelope `verb` field for single-word verbs; for subcommands it is the
    /// space-joined form, e.g. `record start`). Used by the contract test.
    pub verb: &'static str,
    /// The leading argv elements (verb / subcommand path) inserted after
    /// `--json` and before the per-call args.
    pub argv: &'static [&'static str],
    pub description: &'static str,
    pub params: &'static [Param],
}

/// Convenience constructors for `Param` to keep the table compact.
const fn pos(name: &'static str, ty: ParamType, required: bool, d: &'static str) -> Param {
    Param {
        name,
        cli_name: name,
        kind: ParamKind::Positional,
        ty,
        required,
        description: d,
    }
}

const fn flag(
    name: &'static str,
    cli_name: &'static str,
    ty: ParamType,
    required: bool,
    d: &'static str,
) -> Param {
    Param {
        name,
        cli_name,
        kind: ParamKind::Flag,
        ty,
        required,
        description: d,
    }
}

const fn boolflag(name: &'static str, cli_name: &'static str, d: &'static str) -> Param {
    Param {
        name,
        cli_name,
        kind: ParamKind::BoolFlag,
        ty: ParamType::BooleanT,
        required: false,
        description: d,
    }
}

/// The full registry of MCP tools. ONE entry per discrete (request/response)
/// CLI verb. The streaming verbs `events` and `logs` are intentionally absent
/// (they are not request/response and must stay CLI-only), as is `login`
/// (it writes credentials and is not a session-control capability).
pub fn tools() -> &'static [ToolSpec] {
    &TOOLS
}

static TOOLS: [ToolSpec; 23] = [
    ToolSpec {
        name: "waymux_ls",
        verb: "ls",
        argv: &["ls"],
        description: "List all waymux sessions.",
        params: &[],
    },
    ToolSpec {
        name: "waymux_new",
        verb: "new",
        argv: &["new"],
        description: "Create a new waymux session.",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            flag(
                "size",
                "size",
                ParamType::StringT,
                false,
                "Size in WxH pixels, e.g. 1920x1080. Defaults to 1920x1080.",
            ),
            flag(
                "scale",
                "scale",
                ParamType::IntegerT,
                false,
                "Output scale. Defaults to 1.",
            ),
            boolflag(
                "share_audio",
                "share-audio",
                "Share host PulseAudio/PipeWire sockets with the session (local-only).",
            ),
            flag(
                "mem_cap_mb",
                "mem-cap-mb",
                ParamType::IntegerT,
                false,
                "Aggregate memory cap in MiB via cgroup-v2.",
            ),
            flag(
                "cpu_cap_pct",
                "cpu-cap-pct",
                ParamType::IntegerT,
                false,
                "Aggregate CPU cap as percent of one core (200 = two cores).",
            ),
            flag(
                "disk_quota_mb",
                "disk-quota-mb",
                ParamType::IntegerT,
                false,
                "Per-session disk quota in MiB for the runtime dir.",
            ),
            flag(
                "fd_limit",
                "fd-limit",
                ParamType::IntegerT,
                false,
                "File-descriptor cap (RLIMIT_NOFILE) for the session subprocess.",
            ),
            flag(
                "api_key_id",
                "api-key-id",
                ParamType::StringT,
                false,
                "Waymux API-key id (UUID) to attribute usage to.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_rm",
        verb: "rm",
        argv: &["rm"],
        description: "Destroy a waymux session.",
        params: &[pos(
            "name",
            ParamType::StringT,
            true,
            "Session name to destroy.",
        )],
    },
    ToolSpec {
        name: "waymux_info",
        verb: "info",
        argv: &["info"],
        description: "Show details for a single waymux session.",
        params: &[pos("name", ParamType::StringT, true, "Session name.")],
    },
    ToolSpec {
        name: "waymux_spawn",
        verb: "spawn",
        argv: &["spawn"],
        description: "Launch a client (argv) inside a session (local-only). \
                      The command and its arguments are passed as a vector \
                      after a `--` separator; no shell is involved. argv is \
                      bounded (at most ~1024 elements, each at most ~256 KiB) \
                      and argv[0] must be an absolute path.",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            boolflag(
                "compositor",
                "compositor",
                "Run the spawned client as a nested compositor.",
            ),
            // argv is handled specially in exec (trailing `-- <argv...>`); it is
            // declared here so the schema advertises it and the contract test
            // sees a complete param set.
            // An array-valued trailing param. Its `kind` is `Positional`; the
            // exec layer special-cases `ty == StringArrayT` to emit repeated
            // argv elements (and a leading `--` for `argv`).
            Param {
                name: "argv",
                cli_name: "argv",
                kind: ParamKind::Positional,
                ty: ParamType::StringArrayT,
                required: true,
                description: "Command and arguments to run, as an array of strings \
                              (e.g. [\"chromium\", \"--app=https://example.com\"]). \
                              Passed after `--`; never interpreted by a shell.",
            },
        ],
    },
    ToolSpec {
        name: "waymux_windows",
        verb: "windows",
        argv: &["windows"],
        description: "List windows in a session, optionally filtered by tag.",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            flag(
                "tag",
                "tag",
                ParamType::StringT,
                false,
                "Only list windows whose tag set contains this tag.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_tag",
        verb: "tag",
        argv: &["tag"],
        description: "Replace a window's tag set with one or more free-form tags.",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            pos(
                "window_id",
                ParamType::IntegerT,
                true,
                "Window id (see waymux_windows).",
            ),
            Param {
                name: "tags",
                cli_name: "tags",
                kind: ParamKind::Positional,
                ty: ParamType::StringArrayT,
                required: true,
                description: "One or more tags to set on the window (at least one).",
            },
        ],
    },
    ToolSpec {
        name: "waymux_resize",
        verb: "resize",
        argv: &["resize"],
        description: "Resize a session's virtual output (local-only).",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            pos(
                "size",
                ParamType::StringT,
                true,
                "New size in WxH pixels, e.g. 1280x720.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_screenshot",
        verb: "screenshot",
        argv: &["screenshot"],
        description: "Capture a window's current buffer as a PNG. Under --json the \
                      PNG is returned base64-encoded (no file is written).",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            pos(
                "window_id",
                ParamType::IntegerT,
                true,
                "Window id (see waymux_windows).",
            ),
            // The CLI requires -o/--output, but under --json the bytes come back
            // in the envelope and the file is ignored. We always pass a
            // placeholder so clap is satisfied; the client need not supply one.
            flag(
                "output",
                "output",
                ParamType::StringT,
                false,
                "Ignored under --json (bytes are returned in the result); a \
                 placeholder is supplied automatically if omitted.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_screenshot_desktop",
        verb: "screenshot-desktop",
        argv: &["screenshot-desktop"],
        description: "Composite every window in the session into a single desktop \
                      PNG. Under --json the PNG is returned base64-encoded.",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            flag(
                "output",
                "output",
                ParamType::StringT,
                false,
                "Ignored under --json (bytes are returned in the result); a \
                 placeholder is supplied automatically if omitted.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_idle",
        verb: "idle",
        argv: &["idle"],
        description: "Wait until the session has been quiescent for --quiet-ms.",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            flag(
                "quiet_ms",
                "quiet-ms",
                ParamType::IntegerT,
                false,
                "Required quiet window in milliseconds. Defaults to 500.",
            ),
            flag(
                "timeout_ms",
                "timeout-ms",
                ParamType::IntegerT,
                false,
                "Overall timeout in milliseconds. Defaults to 10000.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_wait",
        verb: "wait",
        argv: &["wait"],
        description: "Block until a window matching a selector appears, or timeout \
                      (local-only).",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            flag(
                "app_id",
                "app-id",
                ParamType::StringT,
                false,
                "Match windows with this app id.",
            ),
            flag(
                "title",
                "title",
                ParamType::StringT,
                false,
                "Match windows with this exact title.",
            ),
            flag(
                "tag",
                "tag",
                ParamType::StringT,
                false,
                "Match windows whose tag set contains this tag.",
            ),
            flag(
                "pid",
                "pid",
                ParamType::IntegerT,
                false,
                "Match windows owned by this pid.",
            ),
            flag(
                "nth",
                "nth",
                ParamType::IntegerT,
                false,
                "Select the nth (0-based) matching window.",
            ),
            flag(
                "timeout_ms",
                "timeout-ms",
                ParamType::IntegerT,
                false,
                "Timeout in milliseconds. Defaults to 5000.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_key",
        verb: "key",
        argv: &["key"],
        description: "Send a synthetic key event (local-only).",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            pos(
                "keycode",
                ParamType::IntegerT,
                true,
                "Linux input keycode to send.",
            ),
            boolflag(
                "release",
                "release",
                "Send a key release instead of a press.",
            ),
            flag(
                "modifiers",
                "modifiers",
                ParamType::IntegerT,
                false,
                "Modifier bitmask. Defaults to 0.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_click",
        verb: "click",
        argv: &["click"],
        description: "Move the pointer and optionally press a button (local-only).",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            pos("x", ParamType::NumberT, true, "Pointer x coordinate."),
            pos("y", ParamType::NumberT, true, "Pointer y coordinate."),
            flag(
                "button",
                "button",
                ParamType::IntegerT,
                false,
                "Button code to click. 0 (default) just moves the pointer.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_inject",
        verb: "inject",
        argv: &["inject"],
        description: "Inject one or more input ops, supplied as a JSON array string.",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            flag(
                "ops",
                "ops",
                ParamType::JsonStringT,
                true,
                "JSON array of inject ops, e.g. [{\"type\":\"key\",\"keycode\":30}].",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_attach",
        verb: "attach",
        argv: &["attach"],
        description: "Return the path of the attach Wayland socket for a session \
                      (local-only). Under --json this does NOT launch the \
                      interactive client; it returns the socket path.",
        params: &[pos("name", ParamType::StringT, true, "Session name.")],
    },
    ToolSpec {
        name: "waymux_detach",
        verb: "detach",
        argv: &["detach"],
        description: "Mark a session as detached (local-only).",
        params: &[pos("name", ParamType::StringT, true, "Session name.")],
    },
    ToolSpec {
        name: "waymux_record_start",
        verb: "record start",
        argv: &["record", "start"],
        description: "Start recording a session's composited output to an MKV file \
                      (local-only).",
        params: &[
            pos("name", ParamType::StringT, true, "Session name to record."),
            pos(
                "output",
                ParamType::StringT,
                false,
                "Output MKV path (must be inside the recordings dir). Defaults to \
                 an auto-named file.",
            ),
            flag(
                "codec",
                "codec",
                ParamType::StringT,
                false,
                "Video encoder: ffv1 (default), h264-nvenc, h264-vaapi, \
                 h264-vulkan, ffv1-vulkan, h264-vulkan-lossless, \
                 hevc-vulkan-lossless.",
            ),
            flag(
                "secondary_codec",
                "secondary-codec",
                ParamType::StringT,
                false,
                "Optional second encoder writing <output>.secondary.mkv.",
            ),
            flag(
                "mode",
                "mode",
                ParamType::StringT,
                false,
                "Capture mode: focused-window (default) or whole-desktop.",
            ),
            flag(
                "min_fps",
                "min-fps",
                ParamType::IntegerT,
                false,
                "Minimum fps pacing; re-encodes the last frame when idle.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_record_stop",
        verb: "record stop",
        argv: &["record", "stop"],
        description: "Stop a session's recording (local-only).",
        params: &[pos("name", ParamType::StringT, true, "Session name.")],
    },
    ToolSpec {
        name: "waymux_record_status",
        verb: "record status",
        argv: &["record", "status"],
        description: "Report whether a session is currently recording, plus the output \
                      path(s) and codec when active (local-only).",
        params: &[pos("name", ParamType::StringT, true, "Session name.")],
    },
    ToolSpec {
        name: "waymux_viewer_start",
        verb: "viewer start",
        argv: &["viewer", "start"],
        description: "Start a browser WebRTC viewer for a session and return its URL \
                      (local-only).",
        params: &[
            pos("name", ParamType::StringT, true, "Session name."),
            flag(
                "bind",
                "bind",
                ParamType::StringT,
                false,
                "Bind address for the bridge. Defaults to 127.0.0.1.",
            ),
            flag(
                "port",
                "port",
                ParamType::IntegerT,
                false,
                "Explicit TCP port for the bridge. Defaults to an ephemeral port.",
            ),
        ],
    },
    ToolSpec {
        name: "waymux_viewer_stop",
        verb: "viewer stop",
        argv: &["viewer", "stop"],
        description: "Stop the active viewer for a session (local-only).",
        params: &[pos("name", ParamType::StringT, true, "Session name.")],
    },
    ToolSpec {
        name: "waymux_viewer_status",
        verb: "viewer status",
        argv: &["viewer", "status"],
        description: "Return the viewer URL if one is active for the session, else \
                      a null url (local-only).",
        params: &[pos("name", ParamType::StringT, true, "Session name.")],
    },
];

/// Look up a tool spec by its MCP tool name.
pub fn find_tool(name: &str) -> Option<&'static ToolSpec> {
    TOOLS.iter().find(|t| t.name == name)
}

/// Render a single param's JSON-Schema scalar type fragment.
fn schema_type_fragment(p: &Param) -> Value {
    match p.ty {
        ParamType::StringT => json!({ "type": "string", "description": p.description }),
        ParamType::IntegerT => json!({ "type": "integer", "description": p.description }),
        ParamType::NumberT => json!({ "type": "number", "description": p.description }),
        ParamType::BooleanT => json!({ "type": "boolean", "description": p.description }),
        ParamType::JsonStringT => json!({
            "type": "string",
            "description": format!("{} (must be a JSON-encoded string)", p.description),
        }),
        ParamType::StringArrayT => json!({
            "type": "array",
            "items": { "type": "string" },
            "description": p.description,
        }),
    }
}

/// Build the JSON-Schema `inputSchema` object for a tool.
pub fn input_schema(spec: &ToolSpec) -> Value {
    let mut properties = Map::new();
    let mut required: Vec<Value> = Vec::new();
    for p in spec.params {
        properties.insert(p.name.to_string(), schema_type_fragment(p));
        if p.required {
            required.push(json!(p.name));
        }
    }
    let mut schema = Map::new();
    schema.insert("type".into(), json!("object"));
    schema.insert("properties".into(), Value::Object(properties));
    if !required.is_empty() {
        schema.insert("required".into(), Value::Array(required));
    }
    // Disallow unknown args so clients get a clear signal rather than a silent
    // drop; clap would reject them anyway.
    schema.insert("additionalProperties".into(), json!(false));
    Value::Object(schema)
}

/// Build the `tools/list` array: one entry per tool with name, description, and
/// inputSchema.
pub fn tools_list() -> Value {
    let arr: Vec<Value> = TOOLS
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": input_schema(t),
            })
        })
        .collect();
    Value::Array(arr)
}
