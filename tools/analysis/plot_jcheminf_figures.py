"""Draw the data figures of the J Cheminform manuscript (Figures 2--4).

Every plotted value is read from the committed revision-supplement summaries
under ``outputs/revision_supplement_20261111/``; nothing is typed in by hand.
Run from the repository root with the WSL plotting environment:

    MPLBACKEND=Agg MPLCONFIGDIR="$PWD/outputs/plotting_cache" \
        environments/wsl-figures/.venv/bin/python tools/analysis/plot_jcheminf_figures.py

Figures are written as vector PDFs to
``manuscript/ms2_molecule_retrieval_jcheminf/formal_article/figures/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
import numpy as np  # noqa: E402

SUPPLEMENT = Path("outputs/revision_supplement_20261111")
OUT_DIR = Path("manuscript/ms2_molecule_retrieval_jcheminf/formal_article/figures")

# Single-column sn-jnl text width is 31 pc.
TEXT_WIDTH_IN = 31 * 12 / 72.27

# Categorical slots 1--3 of the validated reference palette (all-pairs PASS);
# the aqua slot is below 3:1 contrast, so bars carry visible value labels.
METHOD_COLOURS = {
    "sccf_ensemble": "#2a78d6",
    "msalign_ensemble": "#eb6834",
    "selected": "#1baf7a",
}
METHOD_LABELS = {
    "sccf_ensemble": "Stage 1 ensemble",
    "msalign_ensemble": r"MSAlign ensemble$^\dagger$",
    "embcos_ensemble": r"EmbCos ensemble$^\dagger$",
    "selected": "Final system",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e4e3df"
NEUTRAL_LIGHT = "#c9c8c2"
NEUTRAL_MID = "#8a8983"
ACCENT = "#e34948"
WEIGHTINGS = (("query_weighted", "Query-weighted"),
              ("connectivity_balanced", "Connectivity-balanced"))


def load(relative: str) -> dict:
    with open(SUPPLEMENT / relative / "summary.json") as handle:
        return json.load(handle)


PREVIEW_DIR: Path | None = None


def save(fig, path: Path) -> None:
    fig.savefig(path)
    if PREVIEW_DIR is not None:
        fig.savefig(PREVIEW_DIR / path.with_suffix(".png").name, dpi=200)
    plt.close(fig)


def style(font_dir: Path | None = None) -> None:
    if font_dir is not None:
        for path in sorted(font_dir.glob("*.ttf")):
            font_manager.fontManager.addfont(path)
    # Refuse a silent font substitution in publication figures.
    for weight in ("normal", "bold"):
        font_manager.findfont(
            font_manager.FontProperties(family="Arial", weight=weight),
            fallback_to_default=False,
        )
    plt.rcParams.update({
        "font.family": "Arial",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.cal": "Arial",
        "mathtext.fallback": None,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 7,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "text.color": INK,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def value_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="right", va="bottom", color=INK)


def figure_reranking(path: Path) -> None:
    """Official-test R@1/R@5/R@20 of Stage 1 ensemble, MSAlign ensemble, final."""
    rows = load("R3/test")["rows"]
    methods = ("sccf_ensemble", "msalign_ensemble", "selected")
    metrics = (("R1", "R@1"), ("R5", "R@5"), ("R20", "R@20"))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.35), sharey=True)
    width = 0.26
    for letter, ax, (weighting, title) in zip("ab", axes, WEIGHTINGS):
        for offset, method in zip((-width, 0.0, width), methods):
            points, lower, upper = [], [], []
            for key, _ in metrics:
                cell = rows[method][weighting][key]
                points.append(cell["point"])
                lower.append(cell["point"] - cell["ci95"][0])
                upper.append(cell["ci95"][1] - cell["point"])
            x = np.arange(len(metrics)) + offset
            ax.bar(x, points, width, color=METHOD_COLOURS[method],
                   edgecolor="white", linewidth=0.8, label=METHOD_LABELS[method])
            ax.errorbar(x, points, yerr=[lower, upper], fmt="none",
                        ecolor=INK_SECONDARY, elinewidth=0.6, capsize=1.2)
            for xi, value, up in zip(x, points, upper):
                ax.text(xi, value + up + 1.2, f"{value:.1f}", ha="center",
                        va="bottom", fontsize=6.3, color=INK_SECONDARY)
        ax.set_xticks(np.arange(len(metrics)), [label for _, label in metrics])
        ax.set_ylim(0, 92)
        ax.set_title(title)
        value_grid(ax)
        panel_letter(ax, letter)
    axes[0].set_ylabel("Official test (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.4)
    fig.tight_layout(w_pad=1.2)
    save(fig, path)


def figure_spectrum_controls(path: Path) -> None:
    """Matched-slice R@1 under real, wrong, training-mean and zero spectra."""
    data = load("R2/test")["populations"]["strict_common"]
    rows = data["rows"]
    methods = ("sccf_ensemble", "msalign_ensemble", "embcos_ensemble", "selected")
    wrong_keys = ("wrong_seed20260820", "wrong_seed20260821", "wrong_seed20260822")

    def r1(method: str, control: str, weighting: str) -> float:
        return rows[method][control]["metrics"][weighting]["R1"]["point"]

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.15), sharey=True)
    y = np.arange(len(methods))[::-1]
    for letter, ax, (weighting, title) in zip("ab", axes, WEIGHTINGS):
        candidate_only = r1(methods[0], "candidate_prior", weighting)
        random_rank = r1(methods[0], "exact_random", weighting)
        ax.axvline(candidate_only, color=INK_MUTED, linewidth=0.8,
                   linestyle=(0, (4, 2)), zorder=1, label="Candidate-only baseline")
        ax.axvline(random_rank, color=INK_MUTED, linewidth=0.8,
                   linestyle=(0, (1, 1.5)), zorder=1, label="Random ranking")
        for yi, method in zip(y, methods):
            real = r1(method, "real", weighting)
            wrong = float(np.mean([r1(method, key, weighting) for key in wrong_keys]))
            ax.plot([wrong, real], [yi, yi], color=INK_MUTED, linewidth=1.2, zorder=2)
            ax.scatter(r1(method, "zero", weighting), yi, marker="s", s=14,
                       color=NEUTRAL_LIGHT, edgecolor=INK_MUTED, linewidth=0.5,
                       zorder=3, label="Zero spectrum")
            ax.scatter(r1(method, "train_mean", weighting), yi, marker="^", s=18,
                       color=NEUTRAL_LIGHT, edgecolor=INK_MUTED, linewidth=0.5,
                       zorder=3, label="Training-mean spectrum")
            ax.scatter(wrong, yi, s=26, facecolor="white", edgecolor=INK,
                       linewidth=0.9, zorder=4, label="Matched wrong spectrum")
            ax.scatter(real, yi, s=26, color=INK, edgecolor="white",
                       linewidth=0.6, zorder=5, label="Real spectrum")
        ax.set_yticks(y, [METHOD_LABELS[m] + (r"$^\dagger$" if m == "sccf_ensemble" else "")
                          for m in methods])
        ax.set_xlim(0, 31)
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.set_xlabel("R@1 on matched slice (%)")
        ax.set_title(title)
        value_grid(ax, "x")
        panel_letter(ax, letter)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    order = ("Real spectrum", "Matched wrong spectrum", "Candidate-only baseline",
             "Training-mean spectrum", "Zero spectrum", "Random ranking")
    fig.legend([unique[k] for k in order], order, loc="lower center", ncol=3,
               bbox_to_anchor=(0.55, 1.0), handletextpad=0.4, columnspacing=1.4,
               handlelength=1.6)
    fig.tight_layout(w_pad=1.0)
    save(fig, path)


def figure_molecule_coverage(path: Path) -> None:
    """All-wrong clustering, proximity-binned reachability, coverage intervention."""
    clustering = load("R1a/test")["rows"]
    proximity = load("R1b/test")["rows"]["sccf_ensemble"]["bins"]
    intervention = load("R6_lite/validation/analysis")

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 4.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05), width_ratios=(1.15, 1.0),
                            hspace=0.62, wspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    # (a) Molecules wrong for every spectrum, observed versus permutation null.
    groups = (("Stage 1 ensemble", ("sccf_ensemble",)),
              ("MSAlign, 3 seeds", tuple(f"msalign_seed2026062{i}" for i in (1, 2, 3))),
              ("EmbCos, 3 seeds", tuple(f"embcos_seed2026062{i}" for i in (1, 2, 3))),
              ("Final system", ("selected",)))
    y = np.arange(len(groups))[::-1]
    for yi, (_, keys) in zip(y, groups):
        jitter = np.linspace(-0.12, 0.12, len(keys)) if len(keys) > 1 else [0.0]
        for dy, key in zip(jitter, keys):
            cell = clustering[key]["R1"]
            observed = 100 * cell["minimum4"]["values"]["all_wrong_molecule_share"]
            null = cell["null_minimum4"]
            null_mean = 100 * null["all_wrong_share_mean"]
            null_lo, null_hi = (100 * v for v in null["all_wrong_share_reference95"])
            ax_a.plot([null_lo, null_hi], [yi + dy] * 2, color=INK_MUTED, linewidth=1.0)
            ax_a.plot([null_mean, observed], [yi + dy] * 2, color=GRID, linewidth=1.0,
                      zorder=1)
            ax_a.scatter(null_mean, yi + dy, s=16, facecolor="white", edgecolor=INK_MUTED,
                         linewidth=0.8, zorder=3, label="Permutation null")
            ax_a.scatter(observed, yi + dy, s=16, color=INK, zorder=4, label="Observed")
    ax_a.set_yticks(y, [name for name, _ in groups])
    ax_a.set_xlim(0, 75)
    ax_a.set_ylim(-0.6, len(groups) + 0.35)
    ax_a.set_xlabel("Molecules wrong for every spectrum (%)")
    ax_a.set_title("Errors cluster by molecule (test)")
    value_grid(ax_a, "x")
    handles, labels = ax_a.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax_a.legend([unique["Observed"], unique["Permutation null"]],
                ["Observed", "Permutation null"], loc="upper left",
                ncol=2, handletextpad=0.2, columnspacing=0.8, borderaxespad=0.1)
    panel_letter(ax_a, "a")

    # (b) Stage 1 ensemble R@20 by training proximity, official test.
    bins = ("<0.60", "[0.60,0.70)", "[0.70,0.80)", "[0.80,0.90)", ">=0.90")
    tick_labels = ("<0.6", "0.6–0.7", "0.7–0.8", "0.8–0.9", "≥0.9")
    points, lower, upper = [], [], []
    for key in bins:
        metrics = proximity[key]["metrics"]
        cell = metrics["query_weighted"]["R20"]
        points.append(cell["point"])
        lower.append(cell["point"] - cell["ci95"][0])
        upper.append(cell["ci95"][1] - cell["point"])
    x = np.arange(len(bins))
    ax_b.bar(x, points, 0.66, color=METHOD_COLOURS["sccf_ensemble"],
             edgecolor="white", linewidth=0.8)
    ax_b.errorbar(x, points, yerr=[lower, upper], fmt="none", ecolor=INK_SECONDARY,
                  elinewidth=0.6, capsize=1.2)
    ax_b.set_xticks(x, tick_labels, rotation=30, ha="right", rotation_mode="anchor")
    ax_b.set_ylim(0, 105)
    ax_b.set_xlabel("Training proximity of query molecule")
    ax_b.set_ylabel("Stage 1 R@20 (%)")
    ax_b.set_title("Reachability by proximity (test)")
    value_grid(ax_b)
    panel_letter(ax_b, "b")

    # (c) Coverage intervention: R@20 of cohort A and untouched molecules.
    cohorts = (("A", "Selected well-covered\nmolecules (n = 200)"),
               ("B_clean", "Molecules with untouched\nneighbours (n = 2,691)"))
    centres = (0.0, 1.15)
    arms = (("full", "Full training set", NEUTRAL_LIGHT),
            ("size_matched", "Random removal of distant molecules", NEUTRAL_MID),
            ("targeted", "Targeted removal of close neighbours", ACCENT))
    seeds = ("seed20260621", "seed20260622")
    width = 0.24
    for ci, (cohort, _) in enumerate(cohorts):
        for ai, (arm, label, colour) in enumerate(arms):
            if arm == "full":
                reference = intervention["existing_full_train_reference"]
                mean = reference["two_seed_mean"][cohort]["recall@20"]
                per_seed = [reference[s][cohort]["recall@20"] for s in seeds]
            else:
                block = intervention["metrics"][arm]
                mean = block["two_seed_mean"][cohort]["recall@20"]
                per_seed = [block[s][cohort]["recall@20"] for s in seeds]
            xi = centres[ci] + (ai - 1) * width
            ax_c.bar(xi, mean, width, color=colour, edgecolor="white", linewidth=0.8,
                     label=label if ci == 0 else None)
            ax_c.scatter([xi - 0.04, xi + 0.04], per_seed, s=7, color=INK, zorder=3,
                         label="Individual seeds" if (ci == 0 and ai == 0) else None)
            ax_c.text(xi, max(mean, *per_seed) + 1.5, f"{mean:.1f}", ha="center",
                      va="bottom", fontsize=6.3, color=INK_SECONDARY)
    did = intervention["contrasts"]["difference_in_differences"]["two_seed_mean"]
    did_q = did["query_weighted"]["R@20"]
    lo, hi = did_q["ci95_pp"]
    did_text = f"{did_q['estimate_pp']:.1f} points [{lo:.1f}, {hi:.1f}]".replace("-", "\u2212")
    ax_c.text(1.68, 50, "Difference-in-differences, R@20:\n" + did_text,
              ha="left", va="center", fontsize=6.3, color=INK)
    ax_c.set_xticks(centres, [label for _, label in cohorts])
    ax_c.set_xlim(-0.5, 2.55)
    ax_c.set_ylim(0, 108)
    ax_c.set_ylabel("Stage 1 R@20 (%)")
    ax_c.set_title("Coverage intervention (official validation, Stage 1, two seeds per arm)")
    value_grid(ax_c)
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=2,
                handlelength=1.0, columnspacing=1.2)
    panel_letter(ax_c, "c")

    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--preview-dir", type=Path, default=None,
                        help="also write PNG previews here (not for the manuscript)")
    parser.add_argument("--font-dir", type=Path, default=None,
                        help="directory containing licensed Arial TTF faces, if not installed")
    args = parser.parse_args()
    global PREVIEW_DIR
    PREVIEW_DIR = args.preview_dir
    if PREVIEW_DIR is not None:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style(args.font_dir)
    figure_reranking(args.out_dir / "fig2_reranking_recall.pdf")
    figure_spectrum_controls(args.out_dir / "fig3_spectrum_controls.pdf")
    figure_molecule_coverage(args.out_dir / "fig4_molecule_coverage.pdf")


if __name__ == "__main__":
    main()
