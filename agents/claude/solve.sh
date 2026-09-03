#!/usr/bin/env bash
set -euo pipefail

agent_home="${SAE_AGENT_HOME:?set SAE_AGENT_HOME}"
agent_model="${SAE_AGENT_MODEL:?set SAE_AGENT_MODEL}"
claude_bin="${SAE_AGENT_CLI:?set SAE_AGENT_CLI}"
python_bin="${SAE_AGENT_PYTHON:?set SAE_AGENT_PYTHON}"
probe_mcp="${SAE_PROBE_MCP:?set SAE_PROBE_MCP}"
probe_url="${SAE_PROBE_URL:?set SAE_PROBE_URL}"
cd "$agent_home"

python3 - "$agent_home/mcp.json" "$python_bin" "$probe_mcp" "$probe_url" <<'PY'
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

exec "$claude_bin" --print --bare --verbose \
  --output-format stream-json --no-session-persistence \
  --strict-mcp-config --mcp-config "$agent_home/mcp.json" \
  --tools 'Write,mcp__sae_probe__probe_sae' \
  --permission-mode dontAsk --model "$agent_model" \
  "$(cat agent_input.md)"

