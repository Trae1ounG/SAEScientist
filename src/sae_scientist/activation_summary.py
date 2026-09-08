"""Activation diagnostics for a single feature on labeled evaluation texts."""
from statistics import mean


def summarize_activations(rows: list[dict]) -> dict:
    groups = {
        label: [row["top3_mean"] for row in rows if row["label"] == label]
        for label in ("positive", "hard_negative", "neutral")
    }
    if any(not values for values in groups.values()):
        raise ValueError("activation diagnostics require all three text groups")
    positive = groups["positive"]
    negative = groups["hard_negative"] + groups["neutral"]
    positive_mean = mean(positive)
    wins = sum(p > n for p in positive for n in negative)
    ties = sum(p == n for p in positive for n in negative)
    return {
        **{f"{label}_mean": mean(values) for label, values in groups.items()},
        # No positive activation makes this diagnostic undefined. AUROC still
        # has a defined value, including 0.5 when every activation is zero.
        "hard_negative_to_positive_ratio": (
            mean(groups["hard_negative"]) / positive_mean if positive_mean > 0 else None
        ),
        **{f"{label}_active_rate": mean(v > 0 for v in values)
           for label, values in groups.items()},
        "auroc": (wins + 0.5 * ties) / (len(positive) * len(negative)),
    }
