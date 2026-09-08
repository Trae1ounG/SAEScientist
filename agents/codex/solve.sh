#!/usr/bin/env bash
set -euo pipefail

agent_home="${SAE_AGENT_HOME:?set SAE_AGENT_HOME}"
agent_model="${SAE_AGENT_MODEL:?set SAE_AGENT_MODEL}"
codex_bin="${SAE_AGENT_CLI:?set SAE_AGENT_CLI}"
isolation_canary="${SAE_AGENT_ISOLATION_CANARY:?set SAE_AGENT_ISOLATION_CANARY}"
python_bin="${SAE_AGENT_PYTHON:?set SAE_AGENT_PYTHON}"
probe_mcp="${SAE_PROBE_MCP:?set SAE_PROBE_MCP}"
probe_url="${SAE_PROBE_URL:?set SAE_PROBE_URL}"
reasoning_effort="${SAE_AGENT_REASONING_EFFORT:-}"
cd "$agent_home"

reasoning_args=()
if [[ -n "$reasoning_effort" ]]; then
  reasoning_args=(-c "model_reasoning_effort=\"$reasoning_effort\"")
fi

permission_profile='permissions.sae-scientist={description="SAEScientist-Bench source and network isolation",filesystem={":minimal"="read",":workspace_roots"="write"},network={enabled=false}}'

set +e
"$codex_bin" sandbox \
  -c "$permission_profile" \
  --permission-profile sae-scientist \
  --cd "$agent_home" \
  /bin/test -r "$isolation_canary"
preflight_status=$?
set -e
case "$preflight_status" in
  1) echo "isolation preflight passed" >&2 ;;
  0) echo "isolation preflight failed: canary outside workspace is readable" >&2; exit 70 ;;
  *) echo "isolation preflight failed with status $preflight_status" >&2; exit 70 ;;
esac

exec "$codex_bin" \
  --cd "$agent_home" \
  exec --json --ephemeral --ignore-user-config --ignore-rules \
  --skip-git-repo-check --model "$agent_model" \
  ${reasoning_args[@]+"${reasoning_args[@]}"} \
  -c 'approval_policy="never"' \
  -c 'default_permissions="sae-scientist"' \
  -c "$permission_profile" \
  -c "mcp_servers.sae_probe.command=\"$python_bin\"" \
  -c "mcp_servers.sae_probe.args=[\"$probe_mcp\"]" \
  -c "mcp_servers.sae_probe.env={SAE_PROBE_URL=\"$probe_url\"}" \
  -c 'mcp_servers.sae_probe.default_tools_approval_mode="approve"' \
  - < agent_input.md
