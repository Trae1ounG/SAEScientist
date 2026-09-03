#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CONFIGURATION_PALETTE = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
    "#79706e",
)


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def configuration_name(row: dict[str, Any]) -> str:
    harness = row.get("harness") or "unknown"
    model = row.get("model") or "unknown"
    effort = row.get("reasoning_effort")
    base = f"{harness}/{model}"
    return f"{base} ({effort})" if effort else base


def configuration_color(configuration: str) -> str:
    digest = hashlib.sha256(configuration.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 2**32
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.78)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def configuration_colors(configurations: list[str]) -> dict[str, str]:
    return {
        configuration: (
            CONFIGURATION_PALETTE[index]
            if index < len(CONFIGURATION_PALETTE)
            else configuration_color(configuration)
        )
        for index, configuration in enumerate(sorted(set(configurations)))
    }


def collect_points(payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    discovery: list[dict] = []
    behavior: list[dict] = []
    for row in payload.get("runs", []):
        if not isinstance(row, dict):
            continue
        common = {
            "configuration": configuration_name(row),
            "exact_match": bool(row.get("exact_match", False)),
            "usable_steering": bool(row.get("usable_steering", False)),
        }
        gt_score = finite_number(row.get("gt_normalized_activation"))
        target_effect = finite_number(row.get("steering_effect"))
        if gt_score is not None and target_effect is not None:
            discovery.append({**common, "x": gt_score, "y": target_effect})

        target_relevance = finite_number(row.get("pe_target_relevance"))
        task_preservation = finite_number(row.get("pe_task_preservation"))
        if target_relevance is not None and task_preservation is not None:
            behavior.append(
                {**common, "x": target_relevance, "y": task_preservation}
            )
    return discovery, behavior


def _legend(ax, colors: dict[str, str], line_2d, marker_handles: list) -> None:
    handles = [
        line_2d(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            label=configuration,
        )
        for configuration, color in colors.items()
    ]
    ax.legend(
        handles=handles + marker_handles,
        title="Configuration and marker",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )


def _save(fig, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")


def render_plots(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_dir.mkdir(parents=True, exist_ok=True)
    discovery, behavior = collect_points(payload)
    configurations = sorted(
        {point["configuration"] for point in discovery + behavior}
    )
    colors = configuration_colors(configurations)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for point in discovery:
        ax.scatter(
            point["x"],
            point["y"],
            color=colors[point["configuration"]],
            marker="^" if point["exact_match"] else "o",
            edgecolors="black",
            linewidths=0.45,
            s=48,
        )
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set(xlabel="GT-normalized activation", ylabel="Causal target effect")
    ax.set_title("Discovery quality versus causal steering")
    ax.grid(alpha=0.2)
    if not discovery:
        ax.text(0.5, 0.5, "No complete discovery/steering pairs", ha="center")
    _legend(
        ax,
        colors,
        Line2D,
        [
            Line2D([0], [0], marker="^", color="black", linestyle="", label="Exact match"),
            Line2D([0], [0], marker="o", color="black", linestyle="", label="Alternative feature"),
        ],
    )
    _save(fig, output_dir, "discovery_vs_causal")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for point in behavior:
        ax.scatter(
            point["x"],
            point["y"],
            color=colors[point["configuration"]],
            marker="o" if point["usable_steering"] else "X",
            edgecolors="black",
            linewidths=0.45,
            s=52,
        )
    ax.axvline(2, color="#777777", linewidth=0.8, linestyle="--")
    ax.axhline(2, color="#777777", linewidth=0.8, linestyle="--")
    ax.set(
        xlabel="PE target relevance (0–4)",
        ylabel="PE task preservation (0–4)",
        xlim=(-0.1, 4.1),
        ylim=(-0.1, 4.1),
    )
    ax.set_title("Target induction versus task preservation")
    ax.grid(alpha=0.2)
    if not behavior:
        ax.text(2, 2, "No complete PE behavior pairs", ha="center")
    _legend(
        ax,
        colors,
        Line2D,
        [
            Line2D([0], [0], marker="o", color="black", linestyle="", label="Usable gate passed"),
            Line2D([0], [0], marker="X", color="black", linestyle="", label="Usable gate not passed"),
        ],
    )
    _save(fig, output_dir, "relevance_vs_preservation")
    plt.close(fig)

    return [
        output_dir / f"{stem}.{suffix}"
        for stem in ("discovery_vs_causal", "relevance_vs_preservation")
        for suffix in ("png", "svg")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot discovery and steering behavior from a formal leaderboard."
    )
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.leaderboard.read_text(encoding="utf-8"))
    outputs = render_plots(payload, args.output_dir)
    print(json.dumps({"outputs": [str(path) for path in outputs]}))


if __name__ == "__main__":
    main()

