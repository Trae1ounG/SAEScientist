"""SAEScientist-Bench evaluation primitives."""

from .admission import admission_failures, is_admitted
from .sources import QWEN3_8B_BASE_L0_50, require_official_source

__all__ = [
    "QWEN3_8B_BASE_L0_50",
    "admission_failures",
    "is_admitted",
    "matched_random_direction",
    "require_official_source",
    "steering_hook",
]


def __getattr__(name: str):
    if name in {"matched_random_direction", "steering_hook"}:
        from .steering import matched_random_direction, steering_hook

        return {
            "matched_random_direction": matched_random_direction,
            "steering_hook": steering_hook,
        }[name]
    raise AttributeError(name)
