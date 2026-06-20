// SPDX-License-Identifier: Apache-2.0

//! `waymux viewer token` — mint a throwaway Ed25519 keypair and a short-lived
//! EdDSA viewer JWT for the LOCAL/dev laptop-viewer pathway (Fire-tablet-on-LAN
//! test). This is the Rust port of `scripts/laptop-mint-viewer-token.py`; that
//! helper stays for shell-eval use, this verb supplements it.
//!
//! What it does (mirrors the python helper exactly):
//!   1. Generates an EPHEMERAL Ed25519 keypair from the OS CSPRNG.
//!   2. Builds a JWT with header `{"alg":"EdDSA","typ":"JWT"}` and claims
//!      `{sub, aud:"viewer", vm_session_id, iat, exp}`, base64url-encoded
//!      (no padding) as `header.payload`, signed with the private key, and
//!      appends `.signature` (base64url, no padding).
//!   3. Emits the signed token plus the RAW 32-byte public key as STANDARD
//!      base64 — the exact form the bridge consumes via the session's
//!      `WAYMUX_VIEWER_TOKEN_ED25519_PK` env var.
//!
//! SECURITY (local/dev scope): the PRIVATE key is ephemeral. It lives only in
//! this process, is used once to sign the token, and is dropped when this
//! function returns. It is NEVER persisted, printed, or written to disk. The
//! minted token is short-lived (8 h default, matching the python helper's local
//! test window). This is a developer/self-host convenience for the loopback /
//! LAN viewer path, not a control-plane mint endpoint — the production control
//! plane (`waymux-api`) holds a long-lived private key and mints tokens the
//! whole fleet's bridges trust; this verb mints a token only the session you
//! configure with the emitted public key will trust.

use anyhow::{Context, Result};
use base64::Engine;
use serde::Serialize;
use uuid::Uuid;

/// The `aud` claim every viewer token carries and the bridge requires
/// (`crates/waymux-neko-bridge/internal/server/server.go` rejects any token
/// whose `aud` does not contain `"viewer"`). Matches
/// `waymux_api::sessions::VIEWER_TOKEN_AUDIENCE`.
pub const VIEWER_TOKEN_AUDIENCE: &str = "viewer";

/// Default token lifetime in seconds: 8 h. Matches the python helper's
/// `now + 8 * 3600` local test window and `waymux-api`'s 8 h viewer mint.
pub const DEFAULT_EXP_SECS: u64 = 8 * 3600;

/// JWT claims for the viewer token. Mirrors `waymux_api::sessions::ViewerClaims`
/// and the python helper's claim set so a token minted here passes the bridge's
/// `validateViewerToken`:
///   * `sub`           — a UUID (bridge parses it as `uuid.Parse`)
///   * `vm_session_id` — must equal the session's `WAYMUX_VM_SESSION_ID`
///   * `exp` / `iat`   — Unix seconds (bridge requires `exp`)
///   * `aud`           — always `"viewer"`
#[derive(Debug, Serialize)]
struct ViewerClaims {
    sub: Uuid,
    vm_session_id: Uuid,
    exp: i64,
    iat: i64,
    aud: String,
}

/// JWT header. EdDSA only — the bridge pins `WithValidMethods(["EdDSA"])` and
/// rejects any other family (alg-confusion defence).
#[derive(Debug, Serialize)]
struct JwtHeader {
    alg: &'static str,
    typ: &'static str,
}

/// The result of minting: the signed token, the public key the session must
/// trust, and the timing metadata for human/JSON output.
#[derive(Debug)]
pub struct MintedViewerToken {
    /// The signed EdDSA JWT to put in the viewer URL `?token=`.
    pub token: String,
    /// RAW 32-byte ed25519 public key, STANDARD base64 — the value the
    /// session's `WAYMUX_VIEWER_TOKEN_ED25519_PK` env var must hold so the
    /// bridge verifies this token.
    pub public_key_b64: String,
    /// The `vm_session_id` claim baked into the token (also the value the
    /// session's `WAYMUX_VM_SESSION_ID` must hold).
    pub vm_session_id: Uuid,
    /// `exp` as Unix seconds.
    pub expires_at: i64,
}

/// base64url WITHOUT padding — the JWT standard (RFC 7515) and what the python
/// helper emits (`base64.urlsafe_b64encode(b).rstrip(b"=")`).
fn b64url(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// Mint an ephemeral-key EdDSA viewer token for `vm_session_id`, valid for
/// `exp_secs` seconds. `sub` defaults to a fresh random UUID when `None`
/// (the python helper always uses a random `sub`).
///
/// The private key is generated here, used once, and dropped on return — never
/// persisted, printed, or written to disk.
pub fn mint(vm_session_id: Uuid, exp_secs: u64, sub: Option<Uuid>) -> Result<MintedViewerToken> {
    use ed25519_dalek::{Signer, SigningKey};
    use rand_core::OsRng;

    // Ephemeral keypair. `signing` is the secret half; it is a local binding
    // that goes out of scope (and is dropped) when this function returns. We
    // never serialize, log, or persist it.
    let signing = SigningKey::generate(&mut OsRng);
    let public_key_bytes = signing.verifying_key().to_bytes();
    // STANDARD base64 (with padding) — the form the bridge / session env var
    // `WAYMUX_VIEWER_TOKEN_ED25519_PK` expects (python helper uses
    // `base64.standard_b64encode`; waymux-api uses `public_key_b64()` which is
    // also STANDARD base64). Go decodes it with `base64.StdEncoding`.
    let public_key_b64 = base64::engine::general_purpose::STANDARD.encode(public_key_bytes);

    // Unix-seconds clock. `as i64` is safe well past year 2200.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_secs() as i64;
    let exp = now
        .checked_add(i64::try_from(exp_secs).context("--exp-secs too large for an i64 timestamp")?)
        .context("exp timestamp overflowed i64")?;

    let sub = sub.unwrap_or_else(Uuid::new_v4);
    let header = JwtHeader {
        alg: "EdDSA",
        typ: "JWT",
    };
    let claims = ViewerClaims {
        sub,
        vm_session_id,
        exp,
        iat: now,
        aud: VIEWER_TOKEN_AUDIENCE.to_string(),
    };

    // header.payload, each base64url (no pad) of the compact JSON.
    let header_json = serde_json::to_vec(&header).context("serialize JWT header")?;
    let claims_json = serde_json::to_vec(&claims).context("serialize JWT claims")?;
    let signing_input = format!("{}.{}", b64url(&header_json), b64url(&claims_json));

    // Sign the ASCII signing input; append .signature (base64url, no pad).
    let sig = signing.sign(signing_input.as_bytes());
    let token = format!("{signing_input}.{}", b64url(&sig.to_bytes()));

    Ok(MintedViewerToken {
        token,
        public_key_b64,
        vm_session_id,
        expires_at: exp,
    })
    // `signing` (the private key) is dropped here.
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};
    use serde_json::Value;

    fn decode_b64url(s: &str) -> Vec<u8> {
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(s)
            .expect("valid base64url")
    }

    /// The minted token has the right header (alg EdDSA, typ JWT), decodes to a
    /// payload with all required claims, and VERIFIES against the emitted public
    /// key — a full Ed25519 round-trip over the exact `header.payload` bytes the
    /// bridge feeds its verifier.
    #[test]
    fn minted_token_roundtrips_and_carries_required_claims() {
        let vm = Uuid::new_v4();
        let minted = mint(vm, 300, None).expect("mint succeeds");

        // Three dot-separated segments.
        let parts: Vec<&str> = minted.token.split('.').collect();
        assert_eq!(parts.len(), 3, "JWT must be header.payload.signature");

        // Header: alg EdDSA, typ JWT.
        let header: Value = serde_json::from_slice(&decode_b64url(parts[0])).unwrap();
        assert_eq!(header["alg"], "EdDSA");
        assert_eq!(header["typ"], "JWT");

        // Payload: required claims.
        let payload: Value = serde_json::from_slice(&decode_b64url(parts[1])).unwrap();
        assert_eq!(payload["aud"], "viewer", "aud must be viewer");
        // sub parses as a UUID (the bridge runs uuid.Parse on it).
        let sub_str = payload["sub"].as_str().expect("sub is a string");
        Uuid::parse_str(sub_str).expect("sub must be a UUID");
        // vm_session_id matches what we asked for.
        assert_eq!(payload["vm_session_id"].as_str().unwrap(), vm.to_string());
        // exp is in the future, iat <= exp.
        let exp = payload["exp"].as_i64().expect("exp is i64");
        let iat = payload["iat"].as_i64().expect("iat is i64");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        assert!(exp > now, "exp must be in the future");
        assert!(iat <= exp, "iat must not be after exp");
        assert_eq!(exp, minted.expires_at);

        // Signature VERIFIES against the emitted public key over header.payload.
        let pub_raw = base64::engine::general_purpose::STANDARD
            .decode(&minted.public_key_b64)
            .expect("public key is standard base64");
        assert_eq!(pub_raw.len(), 32, "raw ed25519 public key is 32 bytes");
        let vk = VerifyingKey::from_bytes(&pub_raw.clone().try_into().unwrap())
            .expect("valid verifying key");
        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let sig_bytes = decode_b64url(parts[2]);
        let sig = Signature::from_slice(&sig_bytes).expect("64-byte signature");
        vk.verify(signing_input.as_bytes(), &sig)
            .expect("signature must verify against the emitted public key");
    }

    /// A token minted for one vm_session_id must NOT verify a tampered payload
    /// (defends the signature-binding the bridge relies on for vm_session_id).
    #[test]
    fn tampered_payload_fails_verification() {
        let minted = mint(Uuid::new_v4(), 300, None).unwrap();
        let parts: Vec<&str> = minted.token.split('.').collect();
        let pub_raw = base64::engine::general_purpose::STANDARD
            .decode(&minted.public_key_b64)
            .unwrap();
        let vk = VerifyingKey::from_bytes(&pub_raw.try_into().unwrap()).unwrap();
        let sig = Signature::from_slice(&decode_b64url(parts[2])).unwrap();
        // Flip the signing input: a different vm_session_id payload.
        let forged = format!("{}.{}", parts[0], b64url(b"{\"vm_session_id\":\"other\"}"));
        assert!(
            vk.verify(forged.as_bytes(), &sig).is_err(),
            "tampered payload must fail verification"
        );
    }

    /// An explicit `sub` is honored (the verb's `--sub` flag).
    #[test]
    fn explicit_sub_is_used() {
        let sub = Uuid::new_v4();
        let minted = mint(Uuid::new_v4(), 300, Some(sub)).unwrap();
        let parts: Vec<&str> = minted.token.split('.').collect();
        let payload: Value = serde_json::from_slice(&decode_b64url(parts[1])).unwrap();
        assert_eq!(payload["sub"].as_str().unwrap(), sub.to_string());
    }

    /// Default expiry matches the python helper's 8 h window.
    #[test]
    fn default_exp_is_eight_hours() {
        assert_eq!(DEFAULT_EXP_SECS, 8 * 3600);
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let minted = mint(Uuid::new_v4(), DEFAULT_EXP_SECS, None).unwrap();
        // exp ~= now + 8h (allow a couple seconds of clock drift in the test).
        let expected = before + DEFAULT_EXP_SECS as i64;
        assert!((minted.expires_at - expected).abs() <= 2);
    }
}
