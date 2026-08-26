"""Generate final-width revised experiment figures from canonical results only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "ucf": "UCF",
    "hybrid_cf": "Hybrid-CF",
    "f1": "F1",
    "f2": "F2",
}
COLORS = {
    "ucf": "#0072B2",
    "hybrid_cf": "#666666",
    "f1": "#009E73",
    "f2": "#D55E00",
}
MARKERS = {"ucf": "o", "hybrid_cf": "s", "f1": "^", "f2": "D"}


def _configure() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _series(
    summaries: dict, sigmas: Iterable[float], method: str, metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, lower, upper = [], [], []
    for sigma in sigmas:
        summary = summaries[str(float(sigma))][method][metric]
        means.append(summary["mean"])
        lower.append(summary["ci_lower"])
        upper.append(summary["ci_upper"])
    return np.asarray(means), np.asarray(lower), np.asarray(upper)


def _plot_metric(
    summaries: dict,
    sigmas: list[float],
    methods: list[str],
    metric: str,
    ylabel: str,
    output_stem: Path,
) -> Dict[str, Path]:
    _configure()
    fig, ax = plt.subplots(figsize=(6.3, 3.7))
    x = np.asarray(sigmas, dtype=np.float64)
    for method in methods:
        mean, lower, upper = _series(summaries, sigmas, method, metric)
        ax.plot(
            x,
            mean,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.8,
            markersize=5.5,
        )
        ax.fill_between(x, lower, upper, color=COLORS[method], alpha=0.14, linewidth=0)
    ax.set_xlabel("Feedback noise standard deviation ($\\sigma$)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=len(methods), loc="best")
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def generate_figures(result_path: Path, output_dir: Path) -> Dict[str, Dict[str, Path]]:
    with result_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    payload = document["scientific_payload"]
    summaries = payload["aggregate_summaries"]
    sigmas = [float(value) for value in payload["config"]["feedback_sigmas"]]
    return {
        "figure_a": _plot_metric(
            summaries,
            sigmas,
            ["hybrid_cf", "f1", "f2"],
            "ndcg@10",
            "NDCG@10",
            output_dir / "figure_a_ndcg10_vs_feedback_noise",
        ),
        "figure_b": _plot_metric(
            summaries,
            sigmas,
            ["ucf", "hybrid_cf", "f2"],
            "rmse",
            "RMSE (0–10 scale)",
            output_dir / "figure_b_rmse_vs_feedback_noise",
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results/revised_experiment/canonical_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/revised_experiment/figures"),
    )
    args = parser.parse_args()
    outputs = generate_figures(args.result, args.output_dir)
    for figure, paths in outputs.items():
        print(f"{figure}: {paths['pdf']} | {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
