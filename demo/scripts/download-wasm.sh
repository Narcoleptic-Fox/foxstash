#!/usr/bin/env bash
# download-wasm.sh — fetch the pre-built WASM bundle from the latest GitHub Release
# and extract it to demo/pkg/ so the static site can serve it.
#
# Requires: curl, python3 (standard on Vercel build images and most Linux systems).

set -euo pipefail

REPO="Narcoleptic-Fox/foxstash"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"

# Resolve demo/ relative to this script, regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

echo "Fetching latest release metadata from ${API_URL}..."
RELEASE_JSON=$(curl -fsSL "$API_URL")

# Extract the browser_download_url for the WASM tarball using python3 (always available).
WASM_URL=$(python3 - <<'EOF'
import sys, json

data = json.loads(sys.stdin.read())
assets = data.get("assets", [])
asset = next(
    (a for a in assets
     if a["name"].startswith("foxstash-wasm") and a["name"].endswith(".tar.gz")),
    None,
)
if not asset:
    names = [a["name"] for a in assets]
    raise SystemExit(f"No WASM .tar.gz asset found in latest release. Available: {names}")
print(asset["browser_download_url"])
EOF
<<< "$RELEASE_JSON")

echo "Downloading ${WASM_URL}..."
curl -fsSL "$WASM_URL" | tar -xz -C "$DEMO_DIR"

echo "WASM bundle extracted to ${DEMO_DIR}/pkg/"
