from __future__ import annotations

import json
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

from .scoring import gt_normalized_metrics, spearman_correlation, summarize_rank_rows


def query_features(
    probe_url: str,
    texts: list[str],
    feature_ids: list[int],
    timeout: float = 600,
) -> list[dict[str, Any]]:
    request = Request(
        probe_url.rstrip("/") + "/probe",
        data=json.dumps(
            {"texts": texts, "top_k": 1, "feature_ids": feature_ids}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.loads(response.read())["results"]


def score_probe_results(
    cases: list[dict[str, Any]],
    probed: list[dict[str, Any]],
    feature_count: int,
) -> dict[str, Any]:
    if len(probed) != len(cases):
        raise ValueError("probe response length does not match the evaluation suite")
    rows = []
    for case, measured in zip(cases, probed):
        candidate, expert = measured["selected_features"]
        rows.append(
            {
                "id": case["id"],
                "label": case["label"],
                "activation": candidate["activation"],
                "rank": candidate["rank"],
                "expert_activation": expert["activation"],
                "expert_rank": expert["rank"],
            }
        )
    expert_rows = [
        {
            "id": row["id"],
            "label": row["label"],
            "activation": row["expert_activation"],
            "rank": row["expert_rank"],
        }
        for row in rows
    ]
    activation = summarize_rank_rows(rows, feature_count)
    expert_activation = summarize_rank_rows(expert_rows, feature_count)
    spearman = spearman_correlation(
        [row["activation"] for row in rows],
        [row["expert_activation"] for row in rows],
    )
    return {
        "activation_rank": activation,
        "expert_activation_rank": expert_activation,
        "expert_activation_spearman": spearman,
        "gt_normalized": gt_normalized_metrics(activation, expert_activation, spearman),
        "cases": rows,
    }
