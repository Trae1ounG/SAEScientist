from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def steering_hook(direction: torch.Tensor, alpha: float, positions: str = "all"):
    """Add one SAE decoder direction to all positions or the last position."""

    if positions not in {"all", "last"}:
        raise ValueError("positions must be 'all' or 'last'")

    def hook(_module, _inputs, output):
        hidden = output if torch.is_tensor(output) else output[0]
        steered = hidden.clone()
        delta = direction.to(device=hidden.device, dtype=hidden.dtype) * alpha
        if positions == "all":
            steered += delta
        else:
            steered[:, -1, :] += delta
        if torch.is_tensor(output):
            return steered
        return (steered,) + output[1:]

    return hook


@contextmanager
def steer(layer, direction: torch.Tensor, alpha: float, positions: str = "all") -> Iterator[None]:
    handle = layer.register_forward_hook(steering_hook(direction, alpha, positions))
    try:
        yield
    finally:
        handle.remove()


def matched_random_direction(direction: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_direction = torch.randn(direction.shape, generator=generator)
    return random_direction * (direction.float().norm().cpu() / random_direction.norm())

