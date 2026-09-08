#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_scientist.scoring import (
    gt_normalized_metrics,
    spearman_correlation,
    summarize_rank_rows,
)
from sae_scientist.suites import load_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--full-sae", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--concept-id")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--expert-feature-id", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-id")
    args = parser.parse_args()

    suite = load_suite(args.suite, args.concept_id)
    submission = json.loads(args.submission.read_text(encoding="utf-8"))
    feature_id = int(submission["feature_id"])
    with np.load(args.full_sae) as data:
        w_enc = torch.from_numpy(data["W_enc"].copy()).to(torch.bfloat16)
        w_dec = torch.from_numpy(data["W_dec"].copy()).float()
        b_dec = torch.from_numpy(data["b_dec"].copy()).to(torch.bfloat16)
        b_enc = torch.from_numpy(data["b_enc"].copy()).to(torch.bfloat16)
        thresholds = torch.from_numpy(data["threshold"].copy()).to(torch.bfloat16)
    feature_count = w_enc.shape[1]
    if w_dec.shape[0] != feature_count:
        raise ValueError("unexpected SAE decoder layout")
    if not 0 <= feature_id < feature_count or not 0 <= args.expert_feature_id < feature_count:
        raise ValueError("feature id is outside the SAE feature range")

    candidate_direction = w_dec[feature_id]
    expert_direction = w_dec[args.expert_feature_id]
    direction_cosine = float(
        torch.nn.functional.cosine_similarity(
            candidate_direction.unsqueeze(0), expert_direction.unsqueeze(0)
        )[0]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()
    layer = model.model.layers[args.layer]
    device = model.device
    w_enc = w_enc.to(device)
    b_dec = b_dec.to(device)
    b_enc = b_enc.to(device)
    thresholds = thresholds.to(device)

    rows = []
    for case in suite["activation_cases"]:
        inputs = tokenizer(
            case["text"],
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        ).to(device)
        captured = {}

        def capture(_module, _inputs, output):
            captured["hidden"] = output if torch.is_tensor(output) else output[0]

        handle = layer.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()

        hidden = captured["hidden"][0]
        hidden = hidden[inputs.special_tokens_mask[0] == 0]
        with torch.inference_mode():
            preactivation = (hidden.to(torch.bfloat16) - b_dec) @ w_enc + b_enc
            activation = torch.where(
                preactivation > thresholds, preactivation, torch.zeros_like(preactivation)
            )
            pooled = activation.topk(min(3, activation.shape[0]), dim=0).values.float().mean(0)

        candidate_value = float(pooled[feature_id])
        rank = int((pooled > candidate_value).sum()) + 1 if candidate_value > 0 else feature_count
        expert_value = float(pooled[args.expert_feature_id])
        expert_rank = (
            int((pooled > expert_value).sum()) + 1 if expert_value > 0 else feature_count
        )
        rows.append(
            {
                "id": case["id"],
                "label": case["label"],
                "activation": candidate_value,
                "rank": rank,
                "expert_activation": expert_value,
                "expert_rank": expert_rank,
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
    activation_rank = summarize_rank_rows(rows, feature_count)
    expert_activation_rank = summarize_rank_rows(expert_rows, feature_count)
    expert_activation_spearman = spearman_correlation(
        [row["activation"] for row in rows],
        [row["expert_activation"] for row in rows],
    )
    result = {
        "schema": 1,
        "trial_id": args.trial_id,
        "feature_id": feature_id,
        "expert_feature_id": args.expert_feature_id,
        "exact_match": feature_id == args.expert_feature_id,
        "activation_rank": activation_rank,
        "expert_activation_rank": expert_activation_rank,
        "expert_activation_spearman": expert_activation_spearman,
        "gt_normalized": gt_normalized_metrics(
            activation_rank, expert_activation_rank, expert_activation_spearman
        ),
        "decoder_direction_cosine": direction_cosine,
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
