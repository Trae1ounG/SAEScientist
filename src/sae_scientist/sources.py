from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OfficialSAESource:
    publisher: str
    repo: str
    base_model: str
    d_model: int
    d_sae: int | None
    top_k: int | None
    num_layers: int
    hook_point: str


QWEN3_8B_BASE_L0_50 = OfficialSAESource(
    publisher="Qwen",
    repo="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
    base_model="Qwen/Qwen3-8B-Base",
    d_model=4096,
    d_sae=65536,
    top_k=50,
    num_layers=36,
    hook_point="resid_post",
)

GEMMA_SCOPE_9B_IT_RES = OfficialSAESource(
    publisher="Google DeepMind",
    repo="google/gemma-scope-9b-it-res",
    base_model="google/gemma-2-9b-it",
    d_model=3584,
    d_sae=None,
    top_k=None,
    num_layers=42,
    hook_point="resid_post",
)

OFFICIAL_SAE_SOURCES = {
    source.repo: source
    for source in (QWEN3_8B_BASE_L0_50, GEMMA_SCOPE_9B_IT_RES)
}


def require_official_source(repo: str) -> OfficialSAESource:
    try:
        return OFFICIAL_SAE_SOURCES[repo]
    except KeyError as error:
        raise ValueError(f"SAE repository is not in the official allowlist: {repo}") from error


def validate_checkpoint_state(state: dict, source: OfficialSAESource) -> None:
    expected = {
        "W_enc": (source.d_sae, source.d_model),
        "W_dec": (source.d_model, source.d_sae),
        "b_enc": (source.d_sae,),
        "b_dec": (source.d_model,),
    }
    mismatches = [
        key
        for key, shape in expected.items()
        if key not in state or tuple(state[key].shape) != shape
    ]
    if mismatches:
        raise ValueError(f"official SAE tensor mismatch: {', '.join(mismatches)}")

