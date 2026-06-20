// SPDX-License-Identifier: Apache-2.0

//! HTTP-level tests for `RemoteTransport`. We exercise the trait directly
//! against a `wiremock` server and assert that the right headers + bodies are
//! sent on the wire.

use serde_json::json;
use waymux_cli::transport::{RemoteTransport, Transport};
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn remote_ls_sends_bearer_header() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/sessions"))
        .and(header("authorization", "Bearer wmx_test_key_xyz"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "sessions": [
                { "name": "demo", "width": 1920, "height": 1080,
                  "created_at": "2026-05-06T00:00:00Z" }
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut t = RemoteTransport::new(server.uri(), "wmx_test_key_xyz").unwrap();
    let sessions = t.list_sessions().await.expect("list_sessions ok");
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].name, "demo");
    assert_eq!(sessions[0].width, 1920);
    assert_eq!(sessions[0].created_at, "2026-05-06T00:00:00Z");
}

#[tokio::test]
async fn remote_new_posts_json_body() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/sessions"))
        .and(header("authorization", "Bearer wmx_key_abc"))
        .and(header("content-type", "application/json"))
        .and(body_json(json!({
            "name": "new-sess",
            "width": 1280,
            "height": 720,
            "mem_cap_mb": 512
        })))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "name": "new-sess",
            "width": 1280,
            "height": 720,
            "created_at": "2026-05-06T00:00:01Z"
        })))
        .expect(1)
        .mount(&server)
        .await;

    let mut t = RemoteTransport::new(server.uri(), "wmx_key_abc").unwrap();
    let s = t
        .create_session(
            "new-sess",
            1280,
            720,
            /*scale*/ 1,
            /*share_audio*/ false,
            Some(512),
            None,
            None,
            None,
            None,
        )
        .await
        .expect("create_session ok");
    assert_eq!(s.name, "new-sess");
    assert_eq!(s.width, 1280);
    assert_eq!(s.height, 720);
}

#[tokio::test]
async fn remote_401_yields_login_hint() {
    let server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/sessions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .mount(&server)
        .await;

    let mut t = RemoteTransport::new(server.uri(), "wmx_bad").unwrap();
    let err = t.list_sessions().await.expect_err("should error on 401");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("waymux login") || msg.contains("authenticated"),
        "unexpected 401 error msg: {msg}"
    );
}

#[tokio::test]
async fn remote_402_yields_quota_msg() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/sessions"))
        .respond_with(ResponseTemplate::new(402).set_body_string("over"))
        .mount(&server)
        .await;

    let mut t = RemoteTransport::new(server.uri(), "wmx_x").unwrap();
    let err = t
        .create_session("x", 800, 600, 1, false, None, None, None, None, None)
        .await
        .expect_err("should error on 402");
    assert!(
        format!("{err:#}").contains("quota"),
        "expected quota in err: {err:#}"
    );
}

#[tokio::test]
async fn remote_inject_posts_ops_array() {
    use waymux_cli::transport::InjectOp;
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/sessions/demo/inject"))
        .and(header("authorization", "Bearer wmx_k"))
        .and(body_json(json!({
            "ops": [
                {"type": "key", "keycode": 30, "release": false, "modifiers": 0},
                {"type": "pointer", "x": 100.0, "y": 200.0, "button": 272, "state": "press"}
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&server)
        .await;

    let mut t = RemoteTransport::new(server.uri(), "wmx_k").unwrap();
    t.inject(
        "demo",
        &[
            InjectOp::Key {
                keycode: 30,
                release: false,
                modifiers: 0,
            },
            InjectOp::Pointer {
                x: 100.0,
                y: 200.0,
                button: 272,
                state: "press".into(),
            },
        ],
    )
    .await
    .expect("inject ok");
}
