#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sae_bench.sources import GEMMA_SCOPE_9B_IT_RES


WIDTHS = {"16k": 16384, "131k": 131072}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--feature-id", type=int, action="append", required=True)
    parser.add_argument("--width", choices=WIDTHS, required=True)
    parser.add_argument("--average-l0", type=int, required=True)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--resolved-revision")
    parser.add_argument("--label")
    parser.add_argument("--neuronpedia")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    source = GEMMA_SCOPE_9B_IT_RES
    d_sae = WIDTHS[args.width]
    if not 0 <= args.layer < source.num_layers:
        raise ValueError(f"layer must be in [0, {source.num_layers})")
    if len(set(args.feature_id)) != len(args.feature_id):
        raise ValueError("feature ids must be unique")
    if any(not 0 <= feature_id < d_sae for feature_id in args.feature_id):
        raise ValueError(f"feature id must be in [0, {d_sae})")

    checkpoint = (
        f"layer_{args.layer}/width_{args.width}/average_l0_{args.average_l0}/params.npz"
    )
    if args.params:
        if not args.resolved_revision:
            raise ValueError("--resolved-revision is required with --params")
        params_path = args.params
        resolved_revision = args.resolved_revision
    else:
        from huggingface_hub import HfApi, hf_hub_download

        resolved_revision = HfApi().model_info(source.repo, revision=args.revision).sha
        params_path = Path(
            hf_hub_download(source.repo, checkpoint, revision=resolved_revision)
        )

    with np.load(params_path) as params:
        expected = {
            "W_dec": (d_sae, source.d_model),
            "W_enc": (source.d_model, d_sae),
            "b_dec": (source.d_model,),
            "b_enc": (d_sae,),
            "threshold": (d_sae,),
        }
        mismatches = [
            key
            for key, shape in expected.items()
            if key not in params or params[key].shape != shape
        ]
        if mismatches:
            raise ValueError(f"official SAE tensor mismatch: {', '.join(mismatches)}")
        w_dec = params["W_dec"]
        w_enc = params["W_enc"]
        b_dec = params["b_dec"].astype(np.float32)
        b_enc = params["b_enc"]
        threshold = params["threshold"]
        features = {
            feature_id: {
                "decoder": w_dec[feature_id].astype(np.float32),
                "encoder": w_enc[:, feature_id].astype(np.float32),
                "b_dec": b_dec,
                "b_enc": np.float32(b_enc[feature_id]),
                "threshold": np.float32(threshold[feature_id]),
            }
            for feature_id in args.feature_id
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for feature_id, feature in features.items():
        stem = args.output_dir / (
            f"gemma2_9b_it_l{args.layer}_w{args.width}_feature_{feature_id}"
        )
        np.savez(stem.with_suffix(".npz"), **feature)
        provenance = {
            "publisher": source.publisher,
            "official_source": True,
            "repo": source.repo,
            "resolved_revision": resolved_revision,
            "base_model": source.base_model,
            "checkpoint": checkpoint,
            "hookpoint": source.hook_point,
            "layer": args.layer,
            "feature_id": feature_id,
            "label": args.label,
            "label_source": "Neuronpedia auto-interpretation" if args.label else None,
            "neuronpedia": args.neuronpedia,
        }
        stem.with_suffix(".json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(stem.with_suffix(".npz"))


if __name__ == "__main__":
    main()

