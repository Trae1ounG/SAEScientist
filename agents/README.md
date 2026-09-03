# Agent harnesses

Harness adapters only translate the shared workspace contract into a native CLI call. They receive the same `task.json`, `prompt.md`, writable workspace, and disabled-network policy. Credentials remain in the operator environment and are never copied into a run directory.

The adapters must write the same minimal `submission.json`; harness-specific reasoning stays in the native trace under `logs/`. If direct file writing is unavailable, the runner may recover the same one-field JSON object from the final response. Agents receive only one task-specific external capability: the local `probe_sae` MCP tool. It returns measured feature IDs, activations, and ranks but no labels or expert metadata. Codex additionally disables shell network access. Claude exposes only `Write` and `probe_sae`. Cursor runs with its native sandbox and requires a trace audit before a result can be scored; task/subagent calls, web tools, and non-probe MCP calls make the run ineligible.

