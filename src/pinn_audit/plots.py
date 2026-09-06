"""Publication-ready figures for the benchmark audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import compare_sources, metric_winners, rank_switch_summary, rerun_winner_probabilities

INK = "#17212b"
ACCENT = "#20a486"
ALERT = "#d95f59"
MUTED = "#91a3b0"


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_source_conflicts(data: pd.DataFrame, path: Path) -> None:
    conflicts = compare_sources(data)
    conflicts = conflicts[~conflicts["consistent"]].head(12).copy()
    conflicts["label"] = conflicts["family"] + " " + conflicts["case"] + " · " + conflicts["method"]
    y = np.arange(len(conflicts))
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.52 * len(conflicts))))
    ax.hlines(y, conflicts["main_table"], conflicts["appendix_table"], color=MUTED, lw=2)
    ax.scatter(conflicts["main_table"], y, color=ACCENT, s=55, label="Main table", zorder=3)
    ax.scatter(conflicts["appendix_table"], y, color=ALERT, s=55, label="Appendix", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, conflicts["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Reported L2 relative error · log scale")
    ax.set_title("Same claim, different number", loc="left", weight="bold", color=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    ax.legend(frameon=False)
    _finish(fig, path)


def plot_winner_stability(data: pd.DataFrame, path: Path) -> None:
    stability = rerun_winner_probabilities(data)
    leaders = stability.groupby(["family", "case"], sort=False).head(1).copy()
    leaders = leaders.sort_values("winner_probability").head(14)
    labels = leaders["family"] + " " + leaders["case"] + " · " + leaders["method"]
    colors = np.where(leaders["winner_probability"] < 0.8, ALERT, ACCENT)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(labels, leaders["winner_probability"], color=colors)
    ax.axvline(0.8, color=INK, lw=1, ls=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability of remaining the winner on a fresh run")
    ax.set_title("A leaderboard rank can be a low-confidence event", loc="left", weight="bold", color=INK)
    ax.bar_label(bars, labels=[f"{p:.0%}" for p in leaders["winner_probability"]], padding=4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    _finish(fig, path)


def plot_metric_sensitivity(data: pd.DataFrame, path: Path) -> None:
    winners = metric_winners(data)
    switches = rank_switch_summary(data)
    sensitive = switches[switches["metric_sensitive"]][["family", "case"]]
    selected = winners.merge(sensitive, on=["family", "case"])
    selected["case_label"] = selected["family"] + " " + selected["case"]
    selected["metric"] = pd.Categorical(
        selected["metric"], categories=["L1RE", "L2RE", "mERR"], ordered=True
    )
    selected = selected.sort_values(["case_label", "metric"])
    cases = list(dict.fromkeys(selected["case_label"]))
    methods = sorted(selected["method"].unique())
    palette = plt.get_cmap("tab20")
    colors = {method: palette(i % 20) for i, method in enumerate(methods)}
    fig, ax = plt.subplots(figsize=(9, max(5.5, 0.48 * len(cases))))
    for yi, case in enumerate(cases):
        case_rows = selected[selected["case_label"].eq(case)]
        for row in case_rows.itertuples():
            xi = ["L1RE", "L2RE", "mERR"].index(str(row.metric))
            ax.text(
                xi,
                yi,
                row.method,
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                bbox={"boxstyle": "round,pad=0.28", "fc": colors[row.method], "ec": "none"},
            )
    ax.set_xticks(range(3), ["L1 relative error", "L2 relative error", "Maximum error"])
    ax.set_yticks(range(len(cases)), cases)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(len(cases) - 0.4, -0.6)
    ax.set_title("Change the question, change the winner", loc="left", weight="bold", color=INK)
    ax.tick_params(axis="both", length=0)
    ax.spines[:].set_visible(False)
    ax.grid(axis="y", alpha=0.12)
    _finish(fig, path)


def plot_collocation_sensitivity(data: pd.DataFrame, path: Path) -> None:
    cases = list(data["case"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, case in zip(axes.flat, cases):
        case_data = data[data["case"].eq(case)]
        for method, group in case_data.groupby("method", sort=False):
            group = group.sort_values("collocation_points")
            ax.errorbar(
                group["collocation_points"],
                group["mean"],
                yerr=group["std"],
                marker="o",
                lw=2,
                capsize=3,
                label=method,
                color=ACCENT if method == "PINN" else ALERT,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(400, 42_000)
        ax.set_xticks([512, 2048, 8192, 32768], ["512", "2k", "8k", "32k"])
        ax.set_title(case, loc="left", weight="bold")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    fig.supxlabel("Collocation points")
    fig.supylabel("Reported L2 relative error · log scale")
    fig.suptitle("Method advantage is conditional on sampling budget", x=0.08, ha="left", weight="bold", color=INK)
    fig.tight_layout()
    _finish(fig, path)
