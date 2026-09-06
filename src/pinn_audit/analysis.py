"""Reliability diagnostics for benchmark tables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compare_sources(data: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    """Compare duplicate metric claims across the main text and appendix."""
    subset = data[data["metric"].eq("L2RE")].dropna(subset=["mean"])
    pivot = subset.pivot_table(
        index=["family", "case", "method"], columns="source", values="mean", aggfunc="first"
    ).dropna()
    pivot = pivot.reset_index()
    pivot["absolute_gap"] = (pivot["main_table"] - pivot["appendix_table"]).abs()
    denominator = pivot[["main_table", "appendix_table"]].abs().min(axis=1).replace(0, np.nan)
    pivot["relative_gap"] = pivot["absolute_gap"] / denominator
    pivot["consistent"] = pivot["absolute_gap"] <= tolerance
    return pivot.sort_values(["consistent", "relative_gap"], ascending=[True, False])


def _lognormal_parameters(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    variance_ratio = np.square(std / mean)
    sigma2 = np.log1p(variance_ratio)
    return np.log(mean) - sigma2 / 2.0, np.sqrt(sigma2)


def rerun_winner_probabilities(
    data: pd.DataFrame,
    metric: str = "L2RE",
    draws: int = 100_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Estimate which method wins a fresh run using reported mean and std.

    Positive errors are modeled with moment-matched log-normal distributions.
    This is a predictive sensitivity analysis, not a posterior probability.
    """
    rng = np.random.default_rng(seed)
    appendix = data[
        data["source"].eq("appendix_table")
        & data["metric"].eq(metric)
        & data["mean"].gt(0)
        & data["std"].notna()
    ]
    records: list[dict[str, object]] = []
    for (family, case), group in appendix.groupby(["family", "case"], sort=False):
        means = group["mean"].to_numpy(float)
        stds = group["std"].to_numpy(float)
        methods = group["method"].to_numpy(str)
        safe_stds = np.maximum(stds, means * 1e-12)
        mu, sigma = _lognormal_parameters(means, safe_stds)
        samples = rng.lognormal(mu[:, None], sigma[:, None], size=(len(group), draws))
        winners = samples.argmin(axis=0)
        counts = np.bincount(winners, minlength=len(group)) / draws
        entropy = -float(np.sum(counts[counts > 0] * np.log(counts[counts > 0])))
        normalized_entropy = entropy / math.log(len(group)) if len(group) > 1 else 0.0
        for method, mean, std, probability in zip(methods, means, stds, counts):
            records.append(
                {
                    "family": family,
                    "case": case,
                    "metric": metric,
                    "method": method,
                    "reported_mean": mean,
                    "reported_std": std,
                    "winner_probability": probability,
                    "rank_entropy": normalized_entropy,
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["family", "case", "winner_probability"], ascending=[True, True, False]
    )


def metric_winners(data: pd.DataFrame) -> pd.DataFrame:
    """Return the reported winner under each available error definition."""
    appendix = data[data["source"].eq("appendix_table")].dropna(subset=["mean"])
    indices = appendix.groupby(["family", "case", "metric"])["mean"].idxmin()
    return appendix.loc[indices, ["family", "case", "metric", "method", "mean"]].sort_values(
        ["family", "case", "metric"]
    )


def rank_switch_summary(data: pd.DataFrame) -> pd.DataFrame:
    winners = metric_winners(data)
    summary = winners.groupby(["family", "case"]).agg(
        metrics=("metric", "nunique"),
        distinct_winners=("method", "nunique"),
        winners=("method", lambda values: ", ".join(sorted(set(values)))),
    )
    summary["metric_sensitive"] = summary["distinct_winners"] > 1
    return summary.reset_index().sort_values(
        ["metric_sensitive", "distinct_winners"], ascending=[False, False]
    )


def collocation_winners(data: pd.DataFrame) -> pd.DataFrame:
    """Find the best method at each published collocation budget."""
    indices = data.groupby(["case", "collocation_points"])["mean"].idxmin()
    return data.loc[indices, ["case", "collocation_points", "method", "mean"]].sort_values(
        ["case", "collocation_points"]
    )


def collocation_switch_summary(data: pd.DataFrame) -> pd.DataFrame:
    winners = collocation_winners(data)
    return (
        winners.groupby("case")
        .agg(
            budgets=("collocation_points", "nunique"),
            distinct_winners=("method", "nunique"),
            winner_sequence=("method", lambda values: " → ".join(values)),
        )
        .reset_index()
        .assign(budget_sensitive=lambda frame: frame["distinct_winners"] > 1)
    )
