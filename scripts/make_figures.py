#!/usr/bin/env python3
"""Draw README figures from files in results/. Each figure gets a light and a dark variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "first": "#2a78d6",
        "second": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#333331",
        "first": "#3987e5",
        "second": "#d95926",
    },
}
BUCKETS = ("tiny", "small", "medium", "large")


def style(ax, theme, value_axis="y") -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], length=0, labelsize=10)
    ax.grid(axis=value_axis, color=theme["grid"], linewidth=1)
    ax.set_axisbelow(True)


def figure(theme, width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_facecolor(theme["surface"])
    style(ax, theme)
    return fig, ax


def title(ax, theme, main, sub) -> None:
    ax.set_title(main, loc="left", color=theme["text"], fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=theme["muted"], fontsize=9.5, va="bottom")


def stride_ablation(results: Path, out: Path, name: str, theme) -> None:
    p2 = json.loads((results / "eval_p2.json").read_text())["overall"]
    p3 = json.loads((results / "eval_p3.json").read_text())["overall"]
    fig, ax = figure(theme, 7.2, 3.6)
    width = 0.34
    for offset, data, colour, label in (
        (-width / 2 - 0.01, p2, theme["first"], "Stride 4 (P2 to P5)"),
        (width / 2 + 0.01, p3, theme["second"], "Stride 8 (P3 to P5)"),
    ):
        values = [data[f"AP_{b}"] for b in BUCKETS]
        positions = [i + offset for i in range(len(BUCKETS))]
        bars = ax.bar(positions, values, width, color=colour, label=label)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.004,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                color=theme["muted"],
                fontsize=8.5,
            )
    labels = ("tiny\narea < 16²", "small\n< 32²", "medium\n32² to 96²", "large\n> 96²")
    ax.set_xticks(range(len(BUCKETS)), labels)
    ax.set_ylim(0, max(p2["AP_large"], p3["AP_large"], p2["AP_medium"], p3["AP_medium"]) * 1.18)
    legend = ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    title(
        ax,
        theme,
        "AP by object size",
        "AP@[.5:.95], VisDrone val at 1536 px, same 7,500-step schedule",
    )
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", facecolor=theme["surface"])
    plt.close(fig)


def per_class(results: Path, out: Path, name: str, theme) -> None:
    data = json.loads((results / "eval_p2.json").read_text())["per_class"]
    rows = sorted(data.items(), key=lambda kv: kv[1]["AP50"])
    fig, ax = figure(theme, 7.2, 3.8)
    style(ax, theme, "x")
    values = [v["AP50"] for _, v in rows]
    ax.barh([k for k, _ in rows], values, height=0.56, color=theme["first"])
    for i, value in enumerate(values):
        ax.text(value + 0.006, i, f"{value:.3f}", va="center", color=theme["muted"], fontsize=9)
    ax.set_xlim(0, max(values) * 1.12)
    title(ax, theme, "AP50 by class", "Stride 4 model, VisDrone val")
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", facecolor=theme["surface"])
    plt.close(fig)


def training_curve(results: Path, out: Path, name: str, theme) -> None:
    fig, ax = figure(theme, 7.2, 3.4)
    for run, colour, label in (
        ("p2", theme["first"], "Stride 4"),
        ("p3", theme["second"], "Stride 8"),
    ):
        rows = [
            json.loads(line) for line in (results / f"train_{run}.jsonl").read_text().splitlines()
        ]
        points = [(r["step"], r["val"]["AP"]) for r in rows if "val" in r]
        xs, ys = zip(*points, strict=True)
        ax.plot(xs, ys, color=colour, linewidth=2, marker="o", markersize=5, label=label)
        ax.text(xs[-1] + 80, ys[-1], f"{ys[-1]:.3f}", va="center", color=theme["muted"], fontsize=9)
    ax.set_xlim(0, xs[-1] * 1.08)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Training step", color=theme["muted"], fontsize=10)
    legend = ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    title(
        ax, theme, "Validation AP during training", "Every fifth val image at 1024 px, EMA weights"
    )
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", facecolor=theme["surface"])
    plt.close(fig)


def resolution(results: Path, out: Path, name: str, theme) -> None:
    """Two panels with one measure each: overall AP and tiny-object AP against input size."""
    sides = (960, 1280, 1536, 1920, 2400)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=160)
    fig.patch.set_facecolor(theme["surface"])
    panels = (("AP", "All objects"), ("AP_tiny", "Tiny objects, area < 16²"))
    for ax, (metric, heading) in zip(axes, panels, strict=True):
        style(ax, theme)
        for run, colour, label in (
            ("p2", theme["first"], "Stride 4"),
            ("p3", theme["second"], "Stride 8"),
        ):
            values = [
                json.loads((results / f"resolution_{run}_{side}.json").read_text())["overall"][
                    metric
                ]
                for side in sides
            ]
            ax.plot(
                range(len(sides)),
                values,
                color=colour,
                linewidth=2,
                marker="o",
                markersize=5,
                label=label,
            )
        ax.set_xticks(range(len(sides)), [str(side) for side in sides])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Input long side, px", color=theme["muted"], fontsize=9.5)
        ax.set_title(heading, loc="left", color=theme["text"], fontsize=11, fontweight="bold")
    legend = axes[0].legend(frameon=False, loc="lower right", fontsize=9.5)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", facecolor=theme["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("docs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        stride_ablation(args.results, args.output, f"stride_ablation_{mode}", theme)
        per_class(args.results, args.output, f"per_class_{mode}", theme)
        training_curve(args.results, args.output, f"training_{mode}", theme)
        resolution(args.results, args.output, f"resolution_{mode}", theme)


if __name__ == "__main__":
    main()
