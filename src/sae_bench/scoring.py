from __future__ import annotations

import math


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def binary_auroc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        raise ValueError("AUROC requires positive and negative values")
    wins = sum(left > right for left in positive for right in negative)
    ties = sum(left == right for left in positive for right in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _tied_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def spearman_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman correlation requires equal vectors of length at least two")
    left_rank = _tied_ranks(left)
    right_rank = _tied_ranks(right)
    left_mean = mean(left_rank)
    right_mean = mean(right_rank)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_rank, right_rank)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left_rank)
        * sum((y - right_mean) ** 2 for y in right_rank)
    )
    return numerator / denominator if denominator else 0.0


def summarize_rank_rows(rows: list[dict], feature_count: int) -> dict:
    groups = {
        label: [row for row in rows if row["label"] == label]
        for label in ("positive", "hard_negative", "neutral")
    }
    if any(not group for group in groups.values()):
        raise ValueError("positive, hard_negative, and neutral rows are required")

    def group_summary(label: str) -> dict:
        group = groups[label]
        ranks = [float(row["rank"]) for row in group]
        return {
            "count": len(group),
            "mean_activation": mean([float(row["activation"]) for row in group]),
            "mean_rank": mean(ranks),
            "mean_percentile": mean(
                [(feature_count - rank) / (feature_count - 1) for rank in ranks]
            ),
            "mean_reciprocal_rank": mean([1 / rank for rank in ranks]),
        }

    positive_values = [float(row["activation"]) for row in groups["positive"]]
    negative_values = [
        float(row["activation"])
        for label in ("hard_negative", "neutral")
        for row in groups[label]
    ]
    return {
        "positive": group_summary("positive"),
        "hard_negative": group_summary("hard_negative"),
        "neutral": group_summary("neutral"),
        "activation_auroc": binary_auroc(positive_values, negative_values),
    }


def gt_normalized_metrics(candidate: dict, expert: dict, pattern_spearman: float) -> dict:
    def clip(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    candidate_contrast = candidate["positive"]["mean_activation"] - max(
        candidate["hard_negative"]["mean_activation"],
        candidate["neutral"]["mean_activation"],
    )
    expert_contrast = expert["positive"]["mean_activation"] - max(
        expert["hard_negative"]["mean_activation"],
        expert["neutral"]["mean_activation"],
    )
    metrics = {
        "positive_rank_recovery": clip(
            expert["positive"]["mean_rank"] / candidate["positive"]["mean_rank"]
        ),
        "auroc_recovery": clip(
            (candidate["activation_auroc"] - 0.5)
            / (expert["activation_auroc"] - 0.5)
        ),
        "activation_contrast_recovery": clip(candidate_contrast / expert_contrast),
        "expert_pattern_spearman": clip(pattern_spearman),
    }
    metrics["mean_score"] = mean(list(metrics.values()))
    return metrics

