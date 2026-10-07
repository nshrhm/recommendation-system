"""Generate final-width figures from canonical revised-experiment results."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Literal, Tuple

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
MONOCHROME_LINES = {
    "ucf": {"color": "0.2", "linestyle": ":", "markerfacecolor": "white"},
    "hybrid_cf": {"color": "0.35", "linestyle": "-"},
    "f1": {"color": "0.1", "linestyle": "--", "markerfacecolor": "white"},
    "f2": {"color": "0.0", "linestyle": "-."},
}
LANGUAGE_LABELS = {
    "en": {
        "xlabel": "Feedback noise standard deviation ($\\sigma$)",
        "ndcg_ylabel": "NDCG@10",
        "rmse_ylabel": "RMSE (0–10 scale)",
        "suffix": "",
    },
    "ja": {
        "xlabel": "フィードバックノイズの標準偏差 ($\\sigma$)",
        "ndcg_ylabel": "NDCG@10",
        "rmse_ylabel": "RMSE（0–10尺度）",
        "suffix": "_ja",
    },
}
Language = Literal["en", "ja"]
Style = Literal["color", "monochrome"]


def _reference_sizes(fig) -> tuple[tuple[float, float], tuple[int, int]]:
    """Measure the unchanged color export in memory, without touching its files.

    PDF and Agg have slightly different tight bounding boxes. Read each
    backend's actual output to preserve both page size and PNG pixel size.
    The MediaBox pattern applies to PDFs written here by Matplotlib only.
    """
    pdf = BytesIO()
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    match = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", pdf.getvalue()
    )
    if match is None:
        raise ValueError("Cannot determine reference PDF page dimensions")
    pdf_inches = tuple(float(value) / 72 for value in match.groups())
    png = BytesIO()
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    png.seek(0)
    height, width = plt.imread(png).shape[:2]
    return pdf_inches, (width, height)


def _apply_monochrome(fig, ax, methods: list[str]) -> None:
    # Freeze the color plot's limits and tick locations before changing layout.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    for method, line in zip(methods, ax.lines):
        line.set(**MONOCHROME_LINES[method])
        line.set_linewidth(2.0)
        line.set_markersize(8.0 if method == "f1" else 7.0)
        line.set_markeredgewidth(1.2)
        line.set_zorder(3)
    for band in ax.collections:
        band.set_facecolor("0.5")
        band.set_alpha(0.14)
        band.set_zorder(1)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linestyle=":", linewidth=0.6, alpha=1)
    ax.xaxis.label.set_size(12.5)
    ax.yaxis.label.set_size(12.5)
    ax.tick_params(axis="both", labelsize=12.5)
    ax.legend(
        frameon=False,
        ncol=len(methods),
        loc="best",
        fontsize=12.5,
        handlelength=4.2,
        handletextpad=0.45,
        columnspacing=0.9,
    )


def _configure(language: Language) -> None:
    font_family = "Noto Sans CJK JP" if language == "ja" else "sans-serif"
    plt.rcParams.update(
        {
            "font.family": font_family,
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
    language: Language,
    style: Style = "color",
) -> Dict[str, Path]:
    _configure(language)
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
        ax.fill_between(
            x,
            lower,
            upper,
            color=COLORS[method],
            alpha=0.14,
            linewidth=0,
        )
    ax.set_xlabel(LANGUAGE_LABELS[language]["xlabel"])
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncol=len(methods), loc="best")
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    if style == "monochrome":
        pdf_inches, png_pixels = _reference_sizes(fig)
        _apply_monochrome(fig, ax, methods)
        fig.set_size_inches(*pdf_inches)
        fig.tight_layout(pad=0.6)
        fig.savefig(pdf_path, format="pdf", bbox_inches=None)
        # Tiny backend rounding differences must not change the PNG dimensions.
        fig.set_size_inches(*(np.nextafter(n / 300, np.inf) for n in png_pixels))
        fig.savefig(png_path, format="png", dpi=300, bbox_inches=None)
    else:
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def generate_figures(
    result_path: Path,
    output_dir: Path,
    language: Language = "en",
    style: Style = "color",
) -> Dict[str, Dict[str, Path]]:
    if style not in ("color", "monochrome"):
        raise ValueError(f"Unknown figure style: {style}")
    with result_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    payload = document["scientific_payload"]
    summaries = payload["aggregate_summaries"]
    sigmas = [float(value) for value in payload["config"]["feedback_sigmas"]]
    labels = LANGUAGE_LABELS[language]
    suffix = labels["suffix"]
    if style == "monochrome":
        suffix += "_monochrome"
    return {
        "figure_a": _plot_metric(
            summaries,
            sigmas,
            ["hybrid_cf", "f1", "f2"],
            "ndcg@10",
            labels["ndcg_ylabel"],
            output_dir / f"figure_a_ndcg10_vs_feedback_noise{suffix}",
            language,
            style,
        ),
        "figure_b": _plot_metric(
            summaries,
            sigmas,
            ["ucf", "hybrid_cf", "f2"],
            "rmse",
            labels["rmse_ylabel"],
            output_dir / f"figure_b_rmse_vs_feedback_noise{suffix}",
            language,
            style,
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
    parser.add_argument(
        "--language",
        choices=tuple(LANGUAGE_LABELS),
        default="en",
        help=("Figure-label language. Japanese outputs use " "a _ja filename suffix."),
    )
    parser.add_argument(
        "--style",
        choices=("color", "monochrome"),
        default="color",
        help="Monochrome outputs use a separate _monochrome filename suffix.",
    )
    args = parser.parse_args()
    outputs = generate_figures(
        args.result, args.output_dir, language=args.language, style=args.style
    )
    for figure, paths in outputs.items():
        print(f"{figure}: {paths['pdf']} | {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
