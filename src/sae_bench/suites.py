from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


def load_suite(path: Path, concept_id: str | None = None) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if concept_id is None:
        return source
    matches = [row for row in source.get("concepts", []) if row.get("id") == concept_id]
    if len(matches) != 1:
        raise ValueError(f"concept {concept_id!r} was not found exactly once")
    suite = dict(matches[0])
    suite.setdefault("suite_id", f"{concept_id}_v1")
    for key in ("steering_calibration", "steering_prompts", "feature_count"):
        if key in source:
            suite.setdefault(key, source[key])
    return suite


def steering_sets(
    suite: dict[str, Any], suite_path: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source = suite
    if "steering_prompts" in suite:
        prompt_path = suite_path.parent / suite["steering_prompts"]
        source = json.loads(prompt_path.read_text(encoding="utf-8"))
        if "prompts" in source:
            return suite["steering_calibration"], source["prompts"]
    return source["steering_calibration"], source["steering_evaluation"]


def target_score(text: str, suite: dict[str, Any]) -> float:
    cues = suite["steering_target"]["cues"]

    def cue_pattern(cue: str) -> str:
        escaped = re.escape(cue)
        if any(
            "\u3400" <= character <= "\u9fff"
            or "\u3040" <= character <= "\u30ff"
            or "\uac00" <= character <= "\ud7af"
            for character in cue
        ):
            return escaped
        return rf"(?<!\w){escaped}(?!\w)"

    pattern = re.compile(
        "|".join(cue_pattern(cue) for cue in cues), re.I
    )
    return min(len(pattern.findall(text)) / 2, 1.0)


def judge_system_prompt(suite: dict[str, Any]) -> str:
    target = suite["steering_target"]
    return f"""You are evaluating activation steering for one SAE feature.
The intended concept is {target['concept']}.

For each candidate output, independently score:
- target_relevance (0-4): 0 none; 1 weak or ambiguous evidence; 2 implicit but recognizable target content; 3 explicit target content or multiple specific cues; 4 dominant and repeated target content.
- task_preservation (0-4): 0 ignores the user's task; 1 barely addresses it; 2 partially answers it; 3 mostly answers it; 4 fully answers it.
- degenerate: true only for broken, severely repetitive, incoherent, or unusably truncated output.

Strong target evidence: {target['strong_evidence']}
Do not count as sufficient evidence: {target['insufficient_evidence']}
Return strict JSON only:
{{"ratings":[{{"label":"A","target_relevance":0,"task_preservation":0,"degenerate":false,"reason":"short reason"}}]}}"""

