#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_bench.admission import activation_failures, admission_failures
from sae_bench.sources import require_official_source
from sae_bench.steering import matched_random_direction, steer
from sae_bench.suites import load_suite, steering_sets, target_score


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def topk_mean(values: torch.Tensor, k: int = 3) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.topk(min(k, values.numel())).values.mean())


def binary_auroc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        raise ValueError("AUROC requires positive and negative values")
    wins = sum(p > n for p in positive for n in negative)
    ties = sum(p == n for p in positive for n in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def is_degenerate(text: str) -> bool:
    units = re.findall(r"[A-Za-z0-9_]+|[^\w\s]", text)
    return len(units) < 8 or len(set(units)) / len(units) < 0.35


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--concept-id")
    parser.add_argument("--alphas", default="192,208,224,240,256")
    parser.add_argument("--fallback-alphas", default="")
    parser.add_argument("--positions", choices=("all", "last"), default="all")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    suite = load_suite(args.suite, args.concept_id)
    steering_calibration, steering_evaluation = steering_sets(suite, args.suite)
    if args.evaluation_limit is not None:
        if args.evaluation_limit < 1:
            raise ValueError("--evaluation-limit must be positive")
        steering_evaluation = steering_evaluation[: args.evaluation_limit]
    provenance = json.loads(args.feature.with_suffix(".json").read_text(encoding="utf-8"))
    source = require_official_source(provenance.get("repo", ""))
    if provenance.get("publisher") != source.publisher or not provenance.get("official_source"):
        raise ValueError("feature provenance is not an approved official release")

    with np.load(args.feature) as feature:
        direction = torch.from_numpy(feature["decoder"].copy()).float()
        encoder = torch.from_numpy(feature["encoder"].copy()).float()
        b_dec = torch.from_numpy(feature["b_dec"].copy()).float()
        b_enc = float(feature["b_enc"])
        threshold = float(feature["threshold"])
    if any(vector.numel() != source.d_model for vector in (direction, encoder, b_dec)):
        raise ValueError("feature width does not match the official source")

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()
    layer = model.model.layers[provenance["layer"]]
    encoder_device = encoder.to(model.device)
    b_dec_device = b_dec.to(model.device)

    def encode(hidden: torch.Tensor) -> torch.Tensor:
        preactivation = (hidden.float() - b_dec_device) @ encoder_device + b_enc
        return torch.where(preactivation > threshold, preactivation, torch.zeros_like(preactivation))

    activation_rows = []
    with torch.inference_mode():
        for case in suite["activation_cases"]:
            inputs = tokenizer(
                case["text"],
                add_special_tokens=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            ).to(model.device)
            observed: list[torch.Tensor] = []

            def observe_activation(_module, _inputs, output):
                hidden = output if torch.is_tensor(output) else output[0]
                observed.append(encode(hidden)[0].detach().cpu())

            handle = layer.register_forward_hook(observe_activation)
            try:
                model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=False)
            finally:
                handle.remove()
            values = observed[0][inputs.special_tokens_mask[0].cpu() == 0]
            activation_rows.append(
                {
                    "id": case["id"],
                    "label": case["label"],
                    "top3_mean": topk_mean(values),
                    "max": float(values.max()) if values.numel() else 0.0,
                    "active_tokens": int((values > 0).sum()),
                }
            )

    grouped = {
        label: [row["top3_mean"] for row in activation_rows if row["label"] == label]
        for label in ("positive", "hard_negative", "neutral")
    }
    negatives = grouped["hard_negative"] + grouped["neutral"]
    activation_summary = {
        "positive_mean": mean(grouped["positive"]),
        "hard_negative_mean": mean(grouped["hard_negative"]),
        "neutral_mean": mean(grouped["neutral"]),
        "hard_negative_to_positive_ratio": (
            mean(grouped["hard_negative"]) / mean(grouped["positive"])
        ),
        "positive_active_rate": mean([value > 0 for value in grouped["positive"]]),
        "hard_negative_active_rate": mean([value > 0 for value in grouped["hard_negative"]]),
        "neutral_active_rate": mean([value > 0 for value in grouped["neutral"]]),
        "auroc": binary_auroc(grouped["positive"], negatives),
    }

    def generate(prompt: str, alpha: float, vector: torch.Tensor | None) -> dict:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        activations: list[torch.Tensor] = []

        def observe_generation(_module, _inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            activations.append(encode(hidden).detach().flatten().cpu())

        context = nullcontext() if vector is None else steer(layer, vector, alpha, args.positions)
        with torch.inference_mode(), context:
            handle = layer.register_forward_hook(observe_generation)
            try:
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            finally:
                handle.remove()
        text = tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        values = torch.cat(activations)
        return {
            "target_score": target_score(text, suite),
            "activation_top3_mean": topk_mean(values),
            "activation_max": float(values.max()),
            "degenerate": is_degenerate(text),
            "text": text,
        }

    alpha_rows = []

    def calibrate(values: str) -> None:
        existing = {row["alpha"] for row in alpha_rows}
        for alpha in [float(value) for value in values.split(",") if value]:
            if alpha in existing:
                continue
            runs = [generate(row["prompt"], alpha, direction) for row in steering_calibration]
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "target_score": mean([row["target_score"] for row in runs]),
                    "success_rate": mean([row["target_score"] > 0 for row in runs]),
                    "nondegenerate_rate": mean([not row["degenerate"] for row in runs]),
                    "runs": runs,
                }
            )

    calibrate(args.alphas)
    usable_alphas = [row for row in alpha_rows if row["nondegenerate_rate"] >= 0.9]
    if not usable_alphas and args.fallback_alphas:
        calibrate(args.fallback_alphas)
        usable_alphas = [row for row in alpha_rows if row["nondegenerate_rate"] >= 0.9]
    if not usable_alphas:
        raise RuntimeError("no non-degenerate steering strength found")
    selected = max(
        usable_alphas,
        key=lambda row: (row["success_rate"], row["target_score"], -row["alpha"]),
    )
    selected_alpha = selected["alpha"]

    random_direction = matched_random_direction(direction, args.seed)
    steering_rows = []
    for case in steering_evaluation:
        steering_rows.append(
            {
                "id": case["id"],
                "baseline": generate(case["prompt"], 0.0, None),
                "feature": generate(case["prompt"], selected_alpha, direction),
                "random": generate(case["prompt"], selected_alpha, random_direction),
            }
        )
    reruns = [
        generate(case["prompt"], selected_alpha, direction)["text"]
        for case in steering_evaluation[:5]
    ]
    rerun_agreement = mean(
        [text == steering_rows[index]["feature"]["text"] for index, text in enumerate(reruns)]
    )

    def condition_mean(condition: str, key: str) -> float:
        return mean([row[condition][key] for row in steering_rows])

    evaluation = {
        "heldout_prompts": len(steering_rows),
        "baseline_target_score": condition_mean("baseline", "target_score"),
        "feature_target_score": condition_mean("feature", "target_score"),
        "random_target_score": condition_mean("random", "target_score"),
        "feature_success_rate": mean(
            [row["feature"]["target_score"] > 0 for row in steering_rows]
        ),
        "random_success_rate": mean(
            [row["random"]["target_score"] > 0 for row in steering_rows]
        ),
        "nondegenerate_rate": mean(
            [not row["feature"]["degenerate"] for row in steering_rows]
        ),
        "rerun_agreement": rerun_agreement,
        "scorer": "deterministic suite cue score",
    }
    admission_result = {
        "source": provenance,
        "evaluation": evaluation,
    }
    causal_failures = admission_failures(admission_result)
    activation_failure_reasons = activation_failures(activation_summary)

    result = {
        "schema": 1,
        "trial_id": args.trial_id,
        "model_path": args.model_path,
        "suite": {
            "id": suite["suite_id"],
            "concept_id": args.concept_id,
            "path": str(args.suite),
        },
        "feature": provenance,
        "protocol": {
            "activation_pooling": "top-3 token mean",
            "generation": "greedy",
            "positions": args.positions,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "evaluation_limit": args.evaluation_limit,
        },
        "activation": {"summary": activation_summary, "cases": activation_rows},
        "steering": {
            "calibration": alpha_rows,
            "selected_alpha": selected_alpha,
            "summary": evaluation,
            "cases": steering_rows,
        },
        "quality": {
            "stable": not activation_failure_reasons and not causal_failures,
            "activation_failures": activation_failure_reasons,
            "causal_failures": causal_failures,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"activation": activation_summary, "steering": evaluation, "quality": result["quality"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
