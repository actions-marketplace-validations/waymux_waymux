#!/usr/bin/env python3
# scripts/laptop-mint-viewer-token.py
#
# Mint a throwaway Ed25519 keypair + a short-lived EdDSA viewer token for the
# LOCAL laptop viewer pathway (Fire-tablet-on-LAN test). Prints three
# shell-eval-able lines on stdout:
#
#   WAYMUX_VIEWER_TOKEN_ED25519_PK=<base64 raw 32-byte public key>
#   WAYMUX_VM_SESSION_ID=<uuid>
#   WAYMUX_VIEWER_TOKEN=<signed EdDSA JWT, aud=viewer>
#
# The private key never leaves this process; it is used once to sign the token
# and then discarded. The PK + VM_SESSION_ID go into waymux-session's
# environment so the neko-bridge it spawns can verify the token (fail-closed
# EdDSA path). The token goes in the viewer URL: http://<lan-ip>:8082/?token=...
#
# Adapted from the proven mint block in scripts/v4-user-test-setup.sh:87-116.
import base64
import json
import time
import uuid

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


sk = Ed25519PrivateKey.generate()
pub_raw = sk.public_key().public_bytes(
    serialization.Encoding.Raw, serialization.PublicFormat.Raw
)
pk_b64 = base64.standard_b64encode(pub_raw).decode()
vm_session_id = str(uuid.uuid4())

now = int(time.time())
header = {"alg": "EdDSA", "typ": "JWT"}
claims = {
    "sub": str(uuid.uuid4()),  # bridge requires `sub` to parse as a UUID
    "aud": "viewer",  # bridge requires aud == "viewer"
    "vm_session_id": vm_session_id,  # must match WAYMUX_VM_SESSION_ID
    "iat": now,
    "exp": now + 8 * 3600,  # 8h local test window (bridge requires exp)
}
signing_input = (
    b64url(json.dumps(header, separators=(",", ":")).encode())
    + "."
    + b64url(json.dumps(claims, separators=(",", ":")).encode())
)
jwt = signing_input + "." + b64url(sk.sign(signing_input.encode()))

print(f"WAYMUX_VIEWER_TOKEN_ED25519_PK={pk_b64}")
print(f"WAYMUX_VM_SESSION_ID={vm_session_id}")
print(f"WAYMUX_VIEWER_TOKEN={jwt}")
