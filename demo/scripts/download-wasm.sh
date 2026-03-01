#!/usr/bin/env bash
# download-wasm.sh — fetch the pre-built WASM bundle from the latest GitHub Release
# and extract it to demo/pkg/ so the static site can serve it.
#
# Requires: curl, python3 (standard on Vercel build images and most Linux systems).
# Optional: set GITHUB_TOKEN to avoid the 60 req/hr anonymous rate limit.

set -euo pipefail

REPO="Narcoleptic-Fox/foxstash"
API_URL="https://api.github.com/repos/${REPO}/releases/latest"

# Resolve demo/ relative to this script, regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$SCRIPT_DIR")"

# Build curl auth header if a token is available.
AUTH_HEADER=""
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  AUTH_HEADER="Authorization: Bearer ${GITHUB_TOKEN}"
fi

echo "Fetching latest release metadata from ${API_URL}..."
if [[ -n "$AUTH_HEADER" ]]; then
  RELEASE_JSON=$(curl -fsSL -H "Accept: application/vnd.github+json" -H "$AUTH_HEADER" "$API_URL")
else
  RELEASE_JSON=$(curl -fsSL -H "Accept: application/vnd.github+json" "$API_URL")
fi

# Pass the JSON via env var so the heredoc can supply the Python source via stdin
# without conflicting with sys.stdin.read().
WASM_URL=$(RELEASE_DATA="$RELEASE_JSON" python3 - <<'PYEOF'
import os, json, sys

try:
    data = json.loads(os.environ["RELEASE_DATA"])
except json.JSONDecodeError as e:
    sys.exit(f"Failed to parse GitHub API response: {e}\nResponse was: {os.environ['RELEASE_DATA'][:200]}")

assets = data.get("assets", [])
asset = next(
    (a for a in assets
     if a["name"].startswith("foxstash-wasm") and a["name"].endswith(".tar.gz")),
    None,
)
if not asset:
    names = [a["name"] for a in assets]
    sys.exit(f"No WASM .tar.gz asset found in latest release. Available assets: {names}")

print(asset["browser_download_url"])
PYEOF
)

echo "Downloading ${WASM_URL}..."
curl -fsSL "$WASM_URL" | tar -xz -C "$DEMO_DIR"

echo "WASM bundle extracted to ${DEMO_DIR}/pkg/"
