#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_scientist.sources import require_official_source
from sae_scientist.steering import matched_random_direction, steer


DEFAULT_PROMPT = "Explain why sleep matters for learning in one short paragraph."


def generate(model, tokenizer, prompt: str, max_new_tokens: int, context) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with context:
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)


def bullet_lines(text: str) -> int:
    return sum(bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)) for line in text.splitlines())


def cjk_ratio(text: str) -> float:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    return sum("\u4e00" <= char <= "\u9fff" for char in visible) / len(visible)


def is_degenerate(text: str) -> bool:
    units = re.findall(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]|[^\w\s]", text)
    if len(units) < 8:
        return True
    if len(set(units)) / len(units) < 0.35:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) >= 4 and Counter(lines).most_common(1)[0][1] / len(lines) > 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--feature", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=35)
    parser.add_argument("--alphas", default="5,20,100")
    parser.add_argument("--positions", choices=("all", "last"), default="all")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--output", type=Path, default=Path("results/smoke.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()
    layer = model.model.layers[args.layer]
    provenance_path = args.feature.with_suffix(".json")
    if not provenance_path.exists():
        raise ValueError("feature provenance JSON is required")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    source = require_official_source(provenance.get("repo", ""))
    if provenance.get("publisher") != source.publisher or not provenance.get("official_source"):
        raise ValueError("feature provenance is not an approved official release")
    if provenance.get("layer") != args.layer:
        raise ValueError("feature layer does not match the steering layer")

    direction = torch.from_numpy(np.load(args.feature)).float()
    if direction.numel() != model.config.hidden_size:
        raise ValueError("feature direction width does not match model hidden size")

    cases = []

    def run(label: str, alpha: float, vector: torch.Tensor | None) -> str:
        context = nullcontext() if vector is None else steer(layer, vector, alpha, args.positions)
        text = generate(model, tokenizer, args.prompt, args.max_new_tokens, context)
        cases.append(
            {
                "condition": label,
                "alpha": alpha,
                "bullet_lines": bullet_lines(text),
                "cjk_ratio": cjk_ratio(text),
                "degenerate": is_degenerate(text),
                "text": text,
            }
        )
        return text

    baseline = run("baseline", 0.0, None)
    zero = run("feature", 0.0, direction)
    alphas = [float(value) for value in args.alphas.split(",") if value]
    for alpha in alphas:
        run("feature", alpha, direction)

    stable_cases = [c for c in cases if c["condition"] == "feature" and c["alpha"] > 0 and not c["degenerate"]]
    if not stable_cases:
        raise RuntimeError("no non-degenerate steering strength found")
    selected = max(stable_cases, key=lambda case: (case["cjk_ratio"], case["alpha"]))
    selected_alpha = selected["alpha"]
    selected_text = selected["text"]
    random_direction = matched_random_direction(direction, args.seed)
    run("random", selected_alpha, random_direction)
    rerun = run("feature_rerun", selected_alpha, direction)

    result = {
        "schema": 1,
        "trial_id": args.trial_id,
        "model_path": args.model_path,
        "feature": provenance,
        "generation": {
            "prompt": args.prompt,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "positions": args.positions,
        },
        "cases": cases,
        "checks": {
            "zero_matches_baseline": zero == baseline,
            "selected_alpha": selected_alpha,
            "selected_rerun_matches": rerun == selected_text,
            "selected_changes_output": selected_text != baseline,
            "selected_is_nondegenerate": not is_degenerate(selected_text),
            "selected_cjk_delta": cjk_ratio(selected_text) - cjk_ratio(baseline),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["checks"], ensure_ascii=False))
    for case in cases:
        print(case["condition"], case["alpha"], "bullet_lines=", case["bullet_lines"])


if __name__ == "__main__":
    main()
