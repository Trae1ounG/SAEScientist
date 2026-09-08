import hashlib
import json

from sae_scientist.provenance import environment_record


def test_manifest_records_code_without_environment_secrets(tmp_path, monkeypatch):
    script = tmp_path / "scripts" / "example.py"
    script.parent.mkdir()
    script.write_text("pass\n")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "synthetic-do-not-record")
    result = environment_record(tmp_path)
    assert result["source_sha256"]["scripts/example.py"] == hashlib.sha256(b"pass\n").hexdigest()
    assert result["python"]
    assert "synthetic-do-not-record" not in json.dumps(result)
    assert result["git_revision"] is None
