import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sae_scientist.paper_scores import paper_scores


@pytest.fixture
def measurements():
    activation = {
        "activation_rank": {"positive": {"mean_rank": 10}, "activation_auroc": 0.9},
        "expert_activation_rank": {"positive": {"mean_rank": 10}},
    }
    judgment = {
        "expected_rows": 40, "valid_rows": 40, "error_rows": 0,
        "conditions": {
            "feature": {"target_relevance": 3},
            "baseline": {"target_relevance": 1},
            "random": {"target_relevance": 0.5},
        },
    }
    return activation, judgment


def test_paper_formula_example(measurements):
    result = paper_scores(*measurements)
    assert result["scoring_version"] == "rank-auroc-steering-v1"
    assert result["scores"] == pytest.approx({
        "Rank": 100, "Activation": 80, "Steering": 50, "Overall": 230 / 3,
    })


def test_rank_can_exceed_expert(measurements):
    activation, judgment = measurements
    activation["activation_rank"]["positive"]["mean_rank"] = 5
    assert paper_scores(activation, judgment)["scores"]["Rank"] == pytest.approx(400 / 3)


def test_stronger_random_control_is_used(measurements):
    activation, judgment = measurements
    judgment["conditions"]["random"]["target_relevance"] = 2
    assert paper_scores(activation, judgment)["scores"]["Steering"] == 25


def test_negative_effect_clipped_but_raw_effect_retained(measurements):
    activation, judgment = measurements
    activation["activation_rank"]["activation_auroc"] = 0.2
    judgment["conditions"]["feature"]["target_relevance"] = 0
    result = paper_scores(activation, judgment)
    assert result["scores"]["Activation"] == result["scores"]["Steering"] == 0
    assert result["measurements"]["target_effect"] == -0.25


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "10"])
def test_reject_invalid_rank(measurements, value):
    activation, judgment = measurements
    activation["activation_rank"]["positive"]["mean_rank"] = value
    with pytest.raises(ValueError):
        paper_scores(activation, judgment)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True])
def test_reject_invalid_auroc(measurements, value):
    activation, judgment = measurements
    activation["activation_rank"]["activation_auroc"] = value
    with pytest.raises(ValueError):
        paper_scores(activation, judgment)


@pytest.mark.parametrize("key,value", [
    ("valid_rows", 39), ("error_rows", 1), ("expected_rows", 0),
    ("valid_rows", True), ("expected_rows", 40.0),
])
def test_reject_incomplete_judgment(measurements, key, value):
    activation, judgment = measurements
    judgment[key] = value
    with pytest.raises(ValueError):
        paper_scores(activation, judgment)


@pytest.mark.parametrize("value", [-1, 5, float("nan"), True])
def test_reject_invalid_rating(measurements, value):
    activation, judgment = measurements
    judgment["conditions"]["feature"]["target_relevance"] = value
    with pytest.raises(ValueError):
        paper_scores(activation, judgment)


def test_inputs_unchanged(measurements):
    original = copy.deepcopy(measurements)
    paper_scores(*measurements)
    assert measurements == original


def test_score_script_and_no_overwrite(measurements, tmp_path):
    activation, judgment = measurements
    a, j, out = [tmp_path / name for name in ("activation.json", "judge.json", "score.json")]
    a.write_text(json.dumps(activation))
    j.write_text(json.dumps(judgment))
    script = Path(__file__).resolve().parents[1] / "scripts/score_paper_run.py"
    command = [sys.executable, str(script), "--activation", str(a),
               "--judgment", str(j), "--output", str(out)]
    subprocess.run(command, check=True, capture_output=True)
    before = out.read_bytes()
    assert json.loads(before)["scores"]["Overall"] == pytest.approx(230 / 3)
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert out.read_bytes() == before
