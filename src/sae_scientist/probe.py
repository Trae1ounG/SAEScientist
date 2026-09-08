from __future__ import annotations

import os
from typing import Any


MAX_TEXTS_PER_REQUEST = 64
MAX_TEXT_CHARS = 4096


def validate_request(value: Any, default_top_k: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("each request must be a JSON object")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise ValueError("request id must be a non-empty string")
    texts = value.get("texts")
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_TEXTS_PER_REQUEST:
        raise ValueError(f"texts must contain 1-{MAX_TEXTS_PER_REQUEST} strings")
    if any(not isinstance(text, str) or not text or len(text) > MAX_TEXT_CHARS for text in texts):
        raise ValueError(f"each text must contain 1-{MAX_TEXT_CHARS} characters")
    top_k = value.get("top_k", default_top_k)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 1024:
        raise ValueError("top_k must be an integer in [1, 1024]")
    feature_ids = value.get("feature_ids", [])
    if not isinstance(feature_ids, list) or any(
        isinstance(feature_id, bool) or not isinstance(feature_id, int)
        for feature_id in feature_ids
    ):
        raise ValueError("feature_ids must be a list of integers")
    return {
        "id": value["id"],
        "texts": texts,
        "top_k": top_k,
        "feature_ids": feature_ids,
    }


def make_worker(ray, *, model_path: str, sae_path: str, layer_index: int):
    @ray.remote(num_gpus=1)
    class ProbeWorker:
        def __init__(self) -> None:
            import numpy as np
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            ).eval()
            self.layer = self.model.model.layers[layer_index]
            with np.load(sae_path) as data:
                self.w_enc = torch.from_numpy(data["W_enc"].copy()).to(
                    device=self.model.device, dtype=torch.bfloat16
                )
                self.b_dec = torch.from_numpy(data["b_dec"].copy()).to(
                    device=self.model.device, dtype=torch.bfloat16
                )
                self.b_enc = torch.from_numpy(data["b_enc"].copy()).to(
                    device=self.model.device, dtype=torch.bfloat16
                )
                self.threshold = torch.from_numpy(data["threshold"].copy()).to(
                    device=self.model.device, dtype=torch.bfloat16
                )
            self.feature_count = self.w_enc.shape[1]
            self.visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            self.ray_gpu_ids = ray.get_runtime_context().get_accelerator_ids().get(
                "GPU", []
            )

        def probe(self, request: dict[str, Any]) -> dict[str, Any]:
            rows = [self._probe_text(text, request) for text in request["texts"]]
            return {
                "id": request["id"],
                "worker": {
                    "cuda_visible_devices": self.visible_devices,
                    "ray_gpu_ids": self.ray_gpu_ids,
                },
                "results": rows,
            }

        def _probe_text(self, text: str, request: dict[str, Any]) -> dict[str, Any]:
            torch = self.torch
            inputs = self.tokenizer(
                text,
                add_special_tokens=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            ).to(self.model.device)
            captured = {}

            def capture(_module, _inputs, output):
                captured["hidden"] = output if torch.is_tensor(output) else output[0]

            handle = self.layer.register_forward_hook(capture)
            try:
                with torch.inference_mode():
                    self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        use_cache=False,
                    )
            finally:
                handle.remove()

            hidden = captured["hidden"][0]
            hidden = hidden[inputs.special_tokens_mask[0] == 0]
            with torch.inference_mode():
                preactivation = (hidden.to(torch.bfloat16) - self.b_dec) @ self.w_enc
                preactivation += self.b_enc
                activation = torch.where(
                    preactivation > self.threshold,
                    preactivation,
                    torch.zeros_like(preactivation),
                )
                pooled = activation.topk(min(3, activation.shape[0]), dim=0).values
                pooled = pooled.float().mean(dim=0)
                top = pooled.topk(request["top_k"])

            row = {
                "text": text,
                "top_features": [
                    {"feature_id": int(feature_id), "activation": float(score)}
                    for score, feature_id in zip(top.values, top.indices)
                ],
            }
            if request["feature_ids"]:
                selected = []
                for feature_id in request["feature_ids"]:
                    if not 0 <= feature_id < self.feature_count:
                        raise ValueError(f"feature id out of range: {feature_id}")
                    value = float(pooled[feature_id])
                    rank = int((pooled > value).sum()) + 1 if value > 0 else self.feature_count
                    selected.append(
                        {"feature_id": feature_id, "activation": value, "rank": rank}
                    )
                row["selected_features"] = selected
            return row

    return ProbeWorker

