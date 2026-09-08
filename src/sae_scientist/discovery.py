from __future__ import annotations


def rank_contrast_features(
    positive: list[dict[int, float]],
    negative: list[dict[int, float]],
    limit: int = 20,
) -> list[dict[str, float | int]]:
    """Rank sparse features that activate often and strongly on positive examples."""

    if not positive or not negative:
        raise ValueError("positive and negative examples are required")

    feature_ids = set().union(*(sample.keys() for sample in positive + negative))
    rows = []
    for feature_id in feature_ids:
        pos_values = [sample.get(feature_id, 0.0) for sample in positive]
        neg_values = [sample.get(feature_id, 0.0) for sample in negative]
        pos_rate = sum(value > 0 for value in pos_values) / len(pos_values)
        neg_rate = sum(value > 0 for value in neg_values) / len(neg_values)
        pos_mean = sum(pos_values) / len(pos_values)
        neg_mean = sum(neg_values) / len(neg_values)
        score = (pos_rate - neg_rate) * max(pos_mean - neg_mean, 0.0)
        if score > 0:
            rows.append(
                {
                    "feature_id": feature_id,
                    "score": score,
                    "positive_rate": pos_rate,
                    "negative_rate": neg_rate,
                    "positive_mean": pos_mean,
                    "negative_mean": neg_mean,
                }
            )
    rows.sort(key=lambda row: (row["score"], row["positive_rate"]), reverse=True)
    return rows[:limit]

