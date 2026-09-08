import json

import pytest

from sae_scientist.activation_summary import summarize_activations
from sae_scientist.admission import activation_failures


def rows(positive, hard_negative, neutral):
    return [{"label": label, "top3_mean": value}
            for label, values in (("positive", positive), ("hard_negative", hard_negative),
                                  ("neutral", neutral)) for value in values]


def test_inactive_feature_has_defined_auc_and_serializable_diagnostics():
    summary = summarize_activations(rows([0, 0], [0, 0], [0]))
    assert summary["auroc"] == 0.5
    assert summary["hard_negative_to_positive_ratio"] is None
    assert activation_failures(summary)
    json.dumps(summary, allow_nan=False)


def test_negative_only_activation_is_valid_measurement():
    summary = summarize_activations(rows([0, 0], [1, 2], [3]))
    assert summary["auroc"] == 0
    assert summary["hard_negative_to_positive_ratio"] is None


def test_regular_feature_preserves_formula_and_ties():
    summary = summarize_activations(rows([2, 4], [1, 2], [0]))
    assert summary["positive_mean"] == 3
    assert summary["hard_negative_to_positive_ratio"] == 0.5
    assert summary["auroc"] == pytest.approx(5.5 / 6)


def test_incomplete_groups_fail_explicitly():
    with pytest.raises(ValueError, match="three text groups"):
        summarize_activations(rows([1], [], [0]))
