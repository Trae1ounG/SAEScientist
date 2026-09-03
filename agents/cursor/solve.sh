#!/usr/bin/env bash
set -euo pipefail

agent_home="${SAE_AGENT_HOME:?set SAE_AGENT_HOME}"
agent_model="${SAE_AGENT_MODEL:?set SAE_AGENT_MODEL}"
cursor_bin="${SAE_AGENT_CLI:?set SAE_AGENT_CLI}"
python_bin="${SAE_AGENT_PYTHON:?set SAE_AGENT_PYTHON}"
probe_mcp="${SAE_PROBE_MCP:?set SAE_PROBE_MCP}"
probe_url="${SAE_PROBE_URL:?set SAE_PROBE_URL}"
cd "$agent_home"
mkdir -p .cursor
mkdir -p .cursor-config .cursor-data
mkdir -p .cursor-home

# Cursor discovers user skills from os.homedir(), independently of its config
# and data directories. Authenticate the CLI for this process, then give it an
# empty per-episode HOME so those user skills cannot enter the model context.
cursor_auth_token="${CURSOR_AUTH_TOKEN:-}"
if [[ -z "$cursor_auth_token" ]] && command -v security >/dev/null 2>&1; then
  cursor_auth_token="$(security find-generic-password -a cursor-user -s cursor-access-token -w 2>/dev/null || true)"
fi
if [[ -z "$cursor_auth_token" && -z "${CURSOR_API_KEY:-}" ]]; then
  echo "Cursor authentication is unavailable for the isolated episode" >&2
  exit 2
fi

# Keep user-level rules, skills, and project history out of scored episodes.
# Forced egress also disables Cursor web tools and denies shell network access,
# while retaining loopback access for the benchmark-owned MCP subprocess.
export CURSOR_CONFIG_DIR="$agent_home/.cursor-config"
export CURSOR_DATA_DIR="$agent_home/.cursor-data"
export HOME="$agent_home/.cursor-home"
# Direct authentication must remain process-local. Cursor's default macOS
# credential store tries to persist it in Keychain even for an isolated run;
# the memory backend avoids both that failure and copying credentials into the
# episode workspace.
export AGENT_CLI_CREDENTIAL_STORE=memory
export CURSOR_FORCED_SHELL_EGRESS=1
export CURSOR_FORCED_SHELL_EGRESS_NETWORK_DEFAULT=deny
export CURSOR_FORCED_SHELL_EGRESS_ALLOW_WEB_TOOLS=0

python3 - "$agent_home/.cursor/mcp.json" "$python_bin" "$probe_mcp" "$probe_url" <<'PY'
import json
from pathlib import Path
import sys

path, python_bin, probe_mcp, probe_url = sys.argv[1:]
Path(path).write_text(json.dumps({
    "mcpServers": {
        "sae_probe": {
            "command": python_bin,
            "args": [probe_mcp],
            "env": {"SAE_PROBE_URL": probe_url},
        }
    }
}, indent=2) + "\n")
PY

# Cursor resolves project MCP configuration from the nearest Git root. Each
# episode workspace is intentionally its own tiny project boundary.
git init -q

# Pass the OAuth token through Cursor's hidden direct-auth option, not the
# process environment inherited by agent-created shells. The trace auditor
# rejects process inspection commands as an additional guard around argv.
exec env -u CURSOR_AUTH_TOKEN -u CURSOR_API_KEY \
  "$cursor_bin" --auth-token "$cursor_auth_token" \
  -p --output-format stream-json --trust \
  --sandbox enabled --approve-mcps --force \
  --model "$agent_model" --workspace "$agent_home" \
  "$(cat agent_input.md)"

