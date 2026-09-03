#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sae_scientist.admission import AdmissionThresholds, admission_failures
from sae_scientist.suites import judge_system_prompt, load_suite, steering_sets


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluated_case_count(result: dict[str, Any]) -> int:
    return len(result["steering"]["cases"])


def effective_rerun_agreement(
    repeats: int, judge_agreement: list[bool], raw_rerun_agreement: float
) -> float:
    return mean(judge_agreement) if repeats >= 2 else float(raw_rerun_agreement)


def thresholds_for_stage(stage: str) -> AdmissionThresholds:
    if stage == "formal_steering":
        return AdmissionThresholds()
    if stage == "steering_screen":
        return AdmissionThresholds(
            min_heldout_prompts=8,
            min_target_delta_over_baseline=0.15,
            min_target_delta_over_random=0.15,
            min_success_rate=0.50,
            min_usable_target_rate=0.0,
            min_nondegenerate_rate=0.875,
            min_rerun_agreement=0.80,
        )
    raise ValueError(f"unsupported stage: {stage}")


def causal_thresholds_for_stage(stage: str) -> AdmissionThresholds:
    if stage == "steering_screen":
        return thresholds_for_stage(stage)
    if stage == "formal_steering":
        return AdmissionThresholds(
            min_heldout_prompts=20,
            min_target_delta_over_baseline=0.20,
            min_target_delta_over_random=0.20,
            min_success_rate=0.70,
            min_usable_target_rate=0.0,
            min_nondegenerate_rate=0.50,
            min_rerun_agreement=0.80,
        )
    raise ValueError(f"unsupported stage: {stage}")


def create_judge_client(
    provider: str,
    *,
    model_name: str,
    api_key_env: str,
    azure_endpoint_env: str,
    azure_api_version_env: str,
):
    if provider == "azure-openai":
        from openai import AzureOpenAI

        api_key = os.environ.get(api_key_env) or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get(azure_endpoint_env) or os.environ.get(
            "AZURE_OPENAI_ENDPOINT"
        )
        api_version = os.environ.get(azure_api_version_env) or os.environ.get(
            "OPENAI_API_VERSION"
        )
        missing = [
            name
            for name, value in (
                (api_key_env, api_key),
                (azure_endpoint_env, endpoint),
                (azure_api_version_env, api_version),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Azure OpenAI configuration missing: {', '.join(missing)}")
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
    raise ValueError(f"unsupported judge provider: {provider}")


def parse_response(text: str, labels: set[str]) -> dict[str, dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    ratings = payload.get("ratings", [])
    if {row.get("label") for row in ratings} != labels:
        raise ValueError("judge labels do not match the request")
    parsed = {}
    for row in ratings:
        target = int(row["target_relevance"])
        task = int(row["task_preservation"])
        if not 0 <= target <= 4 or not 0 <= task <= 4:
            raise ValueError("judge score is outside [0, 4]")
        parsed[row["label"]] = {
            "target_relevance": target,
            "task_preservation": task,
            "degenerate": bool(row["degenerate"]),
            "reason": str(row.get("reason", ""))[:200],
        }
    return parsed


def load_resumable_rows(
    path: Path, expected_keys: set[tuple[str, int]]
) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            key = (str(row["case_id"]), int(row["repeat"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if key in expected_keys and "ratings" in row:
            rows[key] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--concept-id")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--provider", choices=("azure-openai",), default="azure-openai"
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key-env", default="AZURE_OPENAI_API_KEY")
    parser.add_argument(
        "--azure-endpoint-env",
        default="AZURE_OPENAI_ENDPOINT",
    )
    parser.add_argument(
        "--azure-api-version-env",
        default="OPENAI_API_VERSION",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--stage",
        choices=("formal_steering", "steering_screen"),
        default="formal_steering",
    )
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    suite = load_suite(args.suite, args.concept_id)
    _, steering_evaluation = steering_sets(suite, args.suite)
    prompts = {row["id"]: row["prompt"] for row in steering_evaluation}
    system_prompt = judge_system_prompt(suite)
    client = create_judge_client(
        args.provider,
        model_name=args.model_name,
        api_key_env=args.api_key_env,
        azure_endpoint_env=args.azure_endpoint_env,
        azure_api_version_env=args.azure_api_version_env,
    )

    jobs = []
    for repeat in range(args.repeats):
        for index, case in enumerate(result["steering"]["cases"]):
            conditions = ["baseline", "feature", "random"]
            random.Random(args.seed + repeat * 1000 + index).shuffle(conditions)
            labels = {chr(ord("A") + offset): condition for offset, condition in enumerate(conditions)}
            blocks = [f"User task:\n{prompts[case['id']]}\n"]
            for label, condition in labels.items():
                blocks.append(f"Output {label}:\n{case[condition]['text']}\n")
            blocks.append(
                "Evaluate every output above exactly once. Return exactly three ratings "
                "with labels A, B, and C; do not omit, rename, or add labels."
            )
            jobs.append((repeat, case["id"], labels, "\n".join(blocks)))

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_prefix.with_suffix(".jsonl")
    expected_keys = {(case_id, repeat) for repeat, case_id, _, _ in jobs}
    resumed = load_resumable_rows(jsonl_path, expected_keys) if args.resume else {}
    pending_jobs = [
        job for job in jobs if (job[1], job[0]) not in resumed
    ]

    def judge(job):
        repeat, case_id, labels, prompt = job
        error = None
        for attempt in range(4):
            try:
                request = {
                    "model": args.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 700,
                    "timeout": args.request_timeout,
                }
                if args.provider == "azure-openai":
                    request["response_format"] = {"type": "json_object"}
                completion = client.chat.completions.create(
                    **request,
                )
                parsed = parse_response(completion.choices[0].message.content, set(labels))
                return {
                    "case_id": case_id,
                    "repeat": repeat,
                    "ratings": {condition: parsed[label] for label, condition in labels.items()},
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    time.sleep(2**attempt)
        return {"case_id": case_id, "repeat": repeat, "error": error}

    rows = list(resumed.values())
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(judge, job) for job in pending_jobs]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["case_id"], row["repeat"]))

    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    valid = [row for row in rows if "ratings" in row]
    condition_summary = {}
    for condition in ("baseline", "feature", "random"):
        ratings = [row["ratings"][condition] for row in valid]
        condition_summary[condition] = {
            "target_relevance": mean([row["target_relevance"] for row in ratings]),
            "target_success_rate": mean([row["target_relevance"] >= 2 for row in ratings]),
            "task_preservation": mean([row["task_preservation"] for row in ratings]),
            "usable_target_rate": mean(
                [
                    row["target_relevance"] >= 2
                    and row["task_preservation"] >= 2
                    and not row["degenerate"]
                    for row in ratings
                ]
            ),
            "degenerate_rate": mean([row["degenerate"] for row in ratings]),
        }

    by_key = {(row["case_id"], row["repeat"]): row for row in valid}
    agreement = []
    if args.repeats >= 2:
        for case_id in prompts:
            for condition in ("baseline", "feature", "random"):
                scores = [
                    by_key[(case_id, repeat)]["ratings"][condition]["target_relevance"]
                    for repeat in range(args.repeats)
                    if (case_id, repeat) in by_key
                ]
                if len(scores) == args.repeats:
                    agreement.append(max(scores) - min(scores) <= 1)
    raw_rerun_agreement = result.get("steering", {}).get("summary", {}).get(
        "rerun_agreement", 0.0
    )
    admission_evaluation = {
        "heldout_prompts": evaluated_case_count(result),
        "baseline_target_score": condition_summary["baseline"]["target_relevance"] / 4,
        "feature_target_score": condition_summary["feature"]["target_relevance"] / 4,
        "random_target_score": condition_summary["random"]["target_relevance"] / 4,
        "feature_success_rate": condition_summary["feature"]["target_success_rate"],
        "usable_target_rate": condition_summary["feature"]["usable_target_rate"],
        "random_success_rate": condition_summary["random"]["target_success_rate"],
        "nondegenerate_rate": 1 - condition_summary["feature"]["degenerate_rate"],
        "rerun_agreement": effective_rerun_agreement(
            args.repeats, agreement, raw_rerun_agreement
        ),
        "scorer": f"blinded PE judge {args.model_name}, target_relevance / 4",
    }
    full_failures = admission_failures(
        {"source": result["feature"], "evaluation": admission_evaluation},
        thresholds_for_stage(args.stage),
    )
    causal_failures = admission_failures(
        {"source": result["feature"], "evaluation": admission_evaluation},
        causal_thresholds_for_stage(args.stage),
    )
    activation_failures = result.get("quality", {}).get("activation_failures", [])
    summary = {
        "schema": 1,
        "judge_provider": args.provider,
        "judge_model": args.model_name,
        "stage": args.stage,
        "result": str(args.result),
        "repeats": args.repeats,
        "expected_rows": len(jobs),
        "valid_rows": len(valid),
        "error_rows": len(rows) - len(valid),
        "resumed_rows": len(resumed),
        "repeat_agreement_within_one_point": mean(agreement),
        "conditions": condition_summary,
        "admission_evaluation": admission_evaluation,
        "quality": {
            "stable": not activation_failures and not full_failures,
            "causal_stable": not activation_failures and not causal_failures,
            "usable_steering": not activation_failures and not full_failures,
            "activation_failures": activation_failures,
            "causal_failures": causal_failures,
            "usability_failures": [
                failure for failure in full_failures if failure not in causal_failures
            ],
        },
    }
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
