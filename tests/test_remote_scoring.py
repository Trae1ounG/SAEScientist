import pytest

from sae_scientist.remote_scoring import score_probe_results


def test_score_probe_results_compares_candidate_and_expert():
    cases = [
        {"id": "p", "label": "positive"},
        {"id": "n", "label": "hard_negative"},
        {"id": "z", "label": "neutral"},
    ]
    probed = [
        {"selected_features": [{"activation": 4, "rank": 1}, {"activation": 4, "rank": 1}]},
        {"selected_features": [{"activation": 0, "rank": 8}, {"activation": 0, "rank": 8}]},
        {"selected_features": [{"activation": 0, "rank": 8}, {"activation": 0, "rank": 8}]},
    ]
    result = score_probe_results(cases, probed, 8)
    assert result["gt_normalized"]["mean_score"] == pytest.approx(1)
    assert result["cases"][0]["expert_rank"] == 1


def test_score_probe_results_rejects_incomplete_response():
    with pytest.raises(ValueError, match="length"):
        score_probe_results([{"id": "p", "label": "positive"}], [], 8)


@pytest.mark.parametrize("activation", [0, 1])
def test_constant_expert_does_not_block_current_paper_scores(activation):
    cases = [{"id": label, "label": label} for label in ("positive", "hard_negative", "neutral")]
    probed = [{"selected_features": [{"activation": activation, "rank": 8}] * 2} for _ in cases]
    result = score_probe_results(cases, probed, 8)
    assert result["activation_rank"]["activation_auroc"] == 0.5
    assert result["gt_normalized"]["auroc_recovery"] is None
    assert result["gt_normalized"]["activation_contrast_recovery"] is None
    assert result["gt_normalized"]["mean_score"] is None
