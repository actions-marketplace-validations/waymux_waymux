#!/usr/bin/env bash
# cargo-tree-guard.sh: dependency-isolation regression guard.
#
# Run from the root of the EXTRACTED public workspace. For each paid
# crate, `cargo tree -i <crate>` must FAIL to find it (it isn't in the
# public tree). If any paid crate RESOLVES, a public crate has pulled a
# paid dependency back in: that is a release-blocking regression, so we
# exit non-zero.
set -uo pipefail

PAID=(waymux-api waymux-meter waymux-bridge)
fail=0

for crate in "${PAID[@]}"; do
  # `cargo tree -i` exits non-zero and prints "error: ... not found" when
  # the package is absent from the dependency graph. That is what we want.
  if out="$(cargo tree -i "$crate" 2>&1)"; then
    echo "REGRESSION: public tree resolves paid crate '$crate':" >&2
    echo "$out" >&2
    fail=1
  else
    echo "OK: '$crate' is not in the public dependency graph"
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "cargo-tree-guard: FAILED (a paid crate was reintroduced)" >&2
  exit 1
fi
echo "cargo-tree-guard: PASSED (no paid crate in the public graph)"
