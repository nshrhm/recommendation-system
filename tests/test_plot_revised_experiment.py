"""Tests for language-specific revised-experiment figure generation."""

import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from src.visualization import plot_revised_experiment as plotting


@pytest.fixture()
def minimal_result(tmp_path: Path) -> Path:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "scientific_payload": {
                    "aggregate_summaries": {},
                    "config": {"feedback_sigmas": [0.0, 0.5, 1.0, 2.0]},
                }
            }
        ),
        encoding="utf-8",
    )
    return result_path


@pytest.mark.parametrize(
    ("language", "suffix", "xlabel", "rmse_ylabel"),
    [
        (
            "en",
            "",
            "Feedback noise standard deviation ($\\sigma$)",
            "RMSE (0–10 scale)",
        ),
        (
            "ja",
            "_ja",
            "フィードバックノイズの標準偏差 ($\\sigma$)",
            "RMSE（0–10尺度）",
        ),
    ],
)
def test_language_controls_labels_and_output_names(
    monkeypatch,
    minimal_result: Path,
    tmp_path: Path,
    language: str,
    suffix: str,
    xlabel: str,
    rmse_ylabel: str,
):
    calls = []

    def fake_plot_metric(*args):
        calls.append(args)
        output_stem = args[5]
        return {
            "pdf": output_stem.with_suffix(".pdf"),
            "png": output_stem.with_suffix(".png"),
        }

    monkeypatch.setattr(plotting, "_plot_metric", fake_plot_metric)
    outputs = plotting.generate_figures(
        minimal_result, tmp_path / "figures", language=language
    )

    assert calls[0][6] == language
    assert calls[1][6] == language
    assert plotting.LANGUAGE_LABELS[language]["xlabel"] == xlabel
    assert calls[1][4] == rmse_ylabel
    assert outputs["figure_a"]["pdf"].name == (
        f"figure_a_ndcg10_vs_feedback_noise{suffix}.pdf"
    )
    assert outputs["figure_b"]["pdf"].name == (
        f"figure_b_rmse_vs_feedback_noise{suffix}.pdf"
    )


def test_default_language_remains_english(monkeypatch, minimal_result, tmp_path):
    calls = []

    def fake_plot_metric(*args):
        calls.append(args)
        output_stem = args[5]
        return {"pdf": output_stem.with_suffix(".pdf")}

    monkeypatch.setattr(plotting, "_plot_metric", fake_plot_metric)
    plotting.generate_figures(minimal_result, tmp_path)
    assert all(call[6] == "en" for call in calls)
    assert calls[0][5].name == "figure_a_ndcg10_vs_feedback_noise"
    assert all(call[7] == "color" for call in calls)


@pytest.mark.parametrize("language", ["en", "ja"])
def test_monochrome_preserves_science_dimensions_and_color_files(
    monkeypatch, tmp_path, language
):
    """Compare actual artists and exports, including overlapping confidence bands."""
    result = Path(__file__).resolve().parents[1] / (
        "results/revised_experiment/canonical_results.json"
    )
    source_bytes = result.read_bytes()
    captured = {}
    savefig = Figure.savefig

    def capture(fig, filename, *args, **kwargs):
        if isinstance(filename, Path) and filename.suffix == ".pdf":
            ax = fig.axes[0]
            captured[filename.stem] = {
                "limits": (ax.get_xlim(), ax.get_ylim()),
                "ticks": (ax.get_xticks().copy(), ax.get_yticks().copy()),
                "labels": (ax.get_xlabel(), ax.get_ylabel()),
                "legend": [t.get_text() for t in ax.get_legend().get_texts()],
                "lines": [line.get_xydata().copy() for line in ax.lines],
                "bands": [
                    p.vertices.copy()
                    for band in ax.collections
                    for p in band.get_paths()
                ],
                "styles": [line.get_linestyle() for line in ax.lines],
                "markers": [line.get_marker() for line in ax.lines],
                "text_sizes": [t.get_fontsize() for t in ax.get_legend().get_texts()],
            }
        return savefig(fig, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    with plt.rc_context():
        color = plotting.generate_figures(result, tmp_path, language=language)
        before = {p: p.read_bytes() for paths in color.values() for p in paths.values()}
        mono = plotting.generate_figures(
            result, tmp_path, language=language, style="monochrome"
        )

    expected_styles = {"figure_a": ["-", "--", "-."], "figure_b": [":", "-", "-."]}
    expected_markers = {"figure_a": ["s", "^", "D"], "figure_b": ["o", "s", "D"]}
    for name in color:
        original, changed = color[name], mono[name]
        assert changed["pdf"].stem == original["pdf"].stem + "_monochrome"
        a, b = (captured[paths["pdf"].stem] for paths in [original, changed])
        for key in ["limits", "labels", "legend"]:
            assert a[key] == b[key]
        for key in ["ticks", "lines", "bands"]:
            assert len(a[key]) == len(b[key])
            for old, new in zip(a[key], b[key]):
                np.testing.assert_array_equal(old, new)
        assert b["styles"] == expected_styles[name]
        assert b["markers"] == expected_markers[name]
        assert all(size == 12.5 for size in b["text_sizes"])
        boxes = [
            re.search(rb"/MediaBox\s*\[([^]]+)\]", p["pdf"].read_bytes())
            for p in [original, changed]
        ]
        np.testing.assert_allclose(
            [float(v) for v in boxes[0][1].split()],
            [float(v) for v in boxes[1][1].split()],
            rtol=0,
            atol=1e-8,
        )
        with Image.open(original["png"]) as old, Image.open(changed["png"]) as new:
            assert old.size == new.size
            rgb = np.asarray(new.convert("RGB"))
            np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
            np.testing.assert_array_equal(rgb[:, :, 1], rgb[:, :, 2])
    assert result.read_bytes() == source_bytes
    assert all(p.read_bytes() == content for p, content in before.items())
