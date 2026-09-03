#!/usr/bin/env bash
set -euo pipefail

agent_home="${SAE_AGENT_HOME:?set SAE_AGENT_HOME}"
agent_model="${SAE_AGENT_MODEL:?set SAE_AGENT_MODEL}"
opencode_bin="${SAE_AGENT_CLI:?set SAE_AGENT_CLI}"
cd "$agent_home"

exec "$opencode_bin" run --model "$agent_model" --format json < prompt.md

