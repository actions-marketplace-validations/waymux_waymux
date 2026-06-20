// SPDX-License-Identifier: Apache-2.0

//! Contract tests: the MCP tool surface MUST cover every discrete (request/
//! response) CLI verb, and must NOT expose the streaming verbs. This is the
//! point of the test suite: it pins the MCP surface to the CLI so the two
//! cannot drift.

use std::collections::BTreeSet;

use waymux_mcp::registry::{tools, ToolSpec};

/// Every discrete CLI verb that MUST have an MCP tool. Subcommands are written
/// in their space-joined canonical form (matching `ToolSpec::verb`).
///
/// This list is the CLI's discrete request/response verbs from
/// `crates/waymux-cli/src/main.rs`, EXCLUDING:
///   - `events`, `logs`  (streaming; not request/response)
///   - `login`           (writes credentials; not a session-control capability)
const EXPECTED_VERBS: &[&str] = &[
    "ls",
    "new",
    "rm",
    "info",
    "spawn",
    "windows",
    "tag",
    "resize",
    "screenshot",
    "screenshot-desktop",
    "idle",
    "wait",
    "key",
    "click",
    "inject",
    "attach",
    "detach",
    "record start",
    "record stop",
    "record status",
    "viewer start",
    "viewer stop",
    "viewer status",
];

/// Verbs that MUST NOT be exposed as MCP tools.
const FORBIDDEN_VERBS: &[&str] = &["events", "logs", "login"];

#[test]
fn every_discrete_verb_has_a_tool() {
    let covered: BTreeSet<&str> = tools().iter().map(|t| t.verb).collect();
    let expected: BTreeSet<&str> = EXPECTED_VERBS.iter().copied().collect();

    let missing: Vec<&str> = expected.difference(&covered).copied().collect();
    assert!(
        missing.is_empty(),
        "discrete CLI verbs with no MCP tool: {missing:?}"
    );

    // And no extra/unexpected verbs leaked in (catches a typo'd verb string).
    let extra: Vec<&str> = covered.difference(&expected).copied().collect();
    assert!(
        extra.is_empty(),
        "MCP tools mapping to unexpected verbs: {extra:?}"
    );
}

#[test]
fn streaming_and_login_verbs_are_not_exposed() {
    let covered: BTreeSet<&str> = tools().iter().map(|t| t.verb).collect();
    for forbidden in FORBIDDEN_VERBS {
        assert!(
            !covered.contains(forbidden),
            "verb `{forbidden}` must NOT be an MCP tool"
        );
    }
}

#[test]
fn tool_names_follow_waymux_underscore_scheme() {
    for t in tools() {
        assert!(
            t.name.starts_with("waymux_"),
            "tool {} does not follow the waymux_<verb> naming scheme",
            t.name
        );
        // The name must be the verb with non-alphanumerics -> '_'.
        let expected: String = format!(
            "waymux_{}",
            t.verb
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect::<String>()
        );
        assert_eq!(t.name, expected, "tool name/verb mismatch for {}", t.verb);
    }
}

#[test]
fn tool_names_are_unique() {
    let names: BTreeSet<&str> = tools().iter().map(|t| t.name).collect();
    assert_eq!(names.len(), tools().len(), "duplicate tool names present");
}

#[test]
fn each_tool_emits_object_schema_with_declared_required() {
    for t in tools() {
        let schema = waymux_mcp::registry::input_schema(t);
        assert_eq!(schema["type"], serde_json::json!("object"), "{}", t.name);
        assert!(
            schema["properties"].is_object(),
            "{} schema has no properties object",
            t.name
        );
        // Every required param appears in the schema's `required` list.
        let required: BTreeSet<String> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        for p in t.params {
            if p.required {
                assert!(
                    required.contains(p.name),
                    "tool {} param {} is required but missing from schema.required",
                    t.name,
                    p.name
                );
            }
        }
    }
}

/// Sanity: the registry is non-empty and each spec has a non-empty argv whose
/// first element matches the verb's first word.
#[test]
fn registry_specs_are_well_formed() {
    assert!(!tools().is_empty());
    for t in tools() {
        let spec: &ToolSpec = t;
        assert!(!spec.argv.is_empty(), "{} has empty argv", spec.name);
        let first_word = spec.verb.split(' ').next().unwrap();
        assert_eq!(
            spec.argv[0], first_word,
            "{} argv[0] does not match verb first word",
            spec.name
        );
    }
}
