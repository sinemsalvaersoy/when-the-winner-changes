"""Build a concise, evidence-linked audit report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import (
    collocation_switch_summary,
    compare_sources,
    distribution_sensitivity,
    rank_switch_summary,
    rerun_winner_probabilities,
)


def build_report(data: pd.DataFrame, output: Path, collocation: pd.DataFrame | None = None) -> str:
    consistency = compare_sources(data)
    conflicts = consistency[~consistency["consistent"]]
    stability = rerun_winner_probabilities(data)
    assumptions = distribution_sensitivity(data)
    top = stability.groupby(["family", "case"], sort=False).head(1)
    fragile = top[top["winner_probability"] < 0.80]
    switches = rank_switch_summary(data)
    switched = switches[switches["metric_sensitive"]]
    budget_switches = collocation_switch_summary(collocation) if collocation is not None else pd.DataFrame()

    hinv = conflicts[(conflicts["case"] == "HInv") & (conflicts["method"] == "vPINN")]
    if hinv.empty:
        headline = "No HInv vPINN conflict was detected."
    else:
        row = hinv.iloc[0]
        factor = max(row["main_table"], row["appendix_table"]) / min(
            row["main_table"], row["appendix_table"]
        )
        headline = (
            f"The reported HInv vPINN L2RE is {row['main_table']:.3g} in the main table "
            f"and {row['appendix_table']:.3g} in the detailed appendix, a {factor:.1f}× difference."
        )

    lines = [
        "# Audit report",
        "",
        "## Executive finding",
        "",
        headline,
        "This discrepancy reverses the HInv winner for that table and changes which result supports the associated comparison. It is a reporting inconsistency, not evidence that either number is the correct experimental result.",
        "",
        "## Audit counts",
        "",
        f"* {len(consistency)} shared L2RE claims compared across main and appendix tables",
        f"* {len(conflicts)} numerical source conflicts above exact transcription tolerance",
        f"* {len(fragile)} of {len(top)} cases where the conditional log-normal estimate for the most likely fresh-run winner remains below the diagnostic 80% threshold",
        f"* {len(switched)} of {len(switches)} cases where the winning method changes across L2RE, L1RE, and maximum error",
        f"* {int(budget_switches['budget_sensitive'].sum()) if not budget_switches.empty else 0} of {len(budget_switches)} collocation ablations where the winning method changes with sampling budget",
        "",
        "## Interpretation",
        "",
        "A leaderboard winner is not automatically a stable property of a method. The audit separates source inconsistency, conditional rerun stability, and metric dependence. These diagnostics support different conclusions and should not be collapsed into one accuracy number.",
        "",
        "The rerun analysis uses moment-matched distributions derived from the reported three-run mean and standard deviation. Its probabilities are conditional simulation estimates, not measured frequencies, posterior probabilities, or substitutes for raw seeds. Independence between methods is assumed because run-level pairing is unavailable.",
        "",
        "Across log-normal, nonnegative-normal, and gamma models, the identity of the most likely winner is unchanged in "
        f"{int(assumptions['winner_consistent_across_models'].sum())} of {len(assumptions)} cases. The estimated winning probability can nevertheless move by as much as "
        f"{assumptions['probability_range'].max():.1%}, so probability magnitudes should not be read as distribution-free facts.",
        "",
        "## Most fragile fresh-run leaders",
        "",
        "| PDE case | Most probable method | Conditional log-normal estimate | Rank entropy |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in top.sort_values("winner_probability").head(8).itertuples():
        lines.append(
            f"| {row.family} {row.case} | {row.method} | {row.winner_probability:.1%} | {row.rank_entropy:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Distribution-model sensitivity",
            "",
        "The full comparison is recorded in `distribution_sensitivity.csv`. These alternative models are robustness checks, not candidates for the true run distribution.",
        "Maximum-error results are unavailable for some methods. Winner comparisons use the methods reported for each metric and should therefore be read together with coverage differences.",
            "",
            "## Metric-sensitive cases",
            "",
            "| PDE case | Number of winners | Winning methods |",
            "| --- | ---: | --- |",
        ]
    )
    for row in switched.itertuples():
        lines.append(f"| {row.family} {row.case} | {row.distinct_winners} | {row.winners} |")
    if not budget_switches.empty:
        lines.extend(
            [
                "",
                "## Collocation-budget sensitivity",
                "",
                "| PDE case | Winner sequence from 512 to 32768 points |",
                "| --- | --- |",
            ]
        )
        for row in budget_switches.itertuples():
            lines.append(f"| {row.case} | {row.winner_sequence} |")
    lines.extend(
        [
            "",
            "## Provenance and limits",
            "",
            "All numerical claims are extracted from the versioned arXiv source for PINNacle v2. The project does not claim misconduct or determine which conflicting entry is correct. Definitive resolution requires raw per-seed outputs or confirmation from the benchmark authors.",
            "",
            "Source: [PINNacle, arXiv:2306.08827v2](https://arxiv.org/abs/2306.08827)",
        ]
    )
    text = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text
