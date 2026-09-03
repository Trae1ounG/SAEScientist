from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class HarnessPolicyTest(unittest.TestCase):
    def test_codex_disables_agent_network_and_web_search(self):
        adapter = (REPO_ROOT / "agents" / "codex" / "solve.sh").read_text()
        self.assertIn("network={enabled=false}", adapter)
        self.assertIn('default_tools_approval_mode="approve"', adapter)
        self.assertNotIn("--search", adapter)

    def test_prompt_forbids_public_feature_metadata(self):
        prompt = (REPO_ROOT / "prompts" / "discover_feature.md").read_text()
        self.assertIn("Network access and web search are disabled", prompt)
        self.assertIn("Neuronpedia", prompt)
        self.assertIn("measured activations", prompt)

    def test_runner_injects_only_the_probe_endpoint(self):
        runner = (REPO_ROOT / "scripts" / "run_agent.py").read_text()
        self.assertIn("SAE_PROBE_MCP", runner)
        self.assertIn("SAE_PROBE_URL", runner)
        self.assertIn("SAE_AGENT_REASONING_EFFORT", runner)

    def test_codex_accepts_explicit_reasoning_effort(self):
        adapter = (REPO_ROOT / "agents" / "codex" / "solve.sh").read_text()
        self.assertIn("SAE_AGENT_REASONING_EFFORT", adapter)
        self.assertIn("model_reasoning_effort", adapter)

    def test_probe_tool_is_declared_read_only(self):
        probe = (REPO_ROOT / "scripts" / "probe_mcp.py").read_text()
        self.assertIn('"readOnlyHint": True', probe)
        self.assertIn('"destructiveHint": False', probe)

    def test_claude_exposes_no_shell_or_web_tools(self):
        adapter = (REPO_ROOT / "agents" / "claude" / "solve.sh").read_text()
        self.assertIn("Write,mcp__sae_probe__probe_sae", adapter)
        self.assertNotIn("WebSearch", adapter)
        self.assertNotIn("Bash,", adapter)

    def test_cursor_uses_native_sandbox(self):
        adapter = (REPO_ROOT / "agents" / "cursor" / "solve.sh").read_text()
        self.assertIn("--sandbox enabled", adapter)
        self.assertIn("--approve-mcps", adapter)
        self.assertIn("--force", adapter)
        self.assertNotIn("--auto-review", adapter)
        self.assertIn("CURSOR_CONFIG_DIR", adapter)
        self.assertIn("CURSOR_DATA_DIR", adapter)
        self.assertIn("CURSOR_FORCED_SHELL_EGRESS_NETWORK_DEFAULT=deny", adapter)
        self.assertIn("CURSOR_FORCED_SHELL_EGRESS_ALLOW_WEB_TOOLS=0", adapter)
        self.assertIn("AGENT_CLI_CREDENTIAL_STORE=memory", adapter)
        self.assertIn("git init -q", adapter)
        self.assertIn('--auth-token "$cursor_auth_token"', adapter)
        self.assertIn("env -u CURSOR_AUTH_TOKEN -u CURSOR_API_KEY", adapter)
        self.assertNotIn('env CURSOR_AUTH_TOKEN="$cursor_auth_token"', adapter)


if __name__ == "__main__":
    unittest.main()

