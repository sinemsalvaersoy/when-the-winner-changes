"""Generate the preregistered mechanism-audit extension."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import mechanism_evidence_matrix


def build_mechanism_report(
    data: pd.DataFrame, collocation: pd.DataFrame, output: Path
) -> str:
    evidence = mechanism_evidence_matrix(data, collocation)
    lines = [
        "# Mechanism audit",
        "",
        "## Question",
        "",
        "Why does a benchmark winner change?",
        "",
        "The first audit establishes dependence on source, metric, rerun model, and collocation budget. None of those observations alone proves collocation overfitting, optimization-basin dependence, or failure to capture invariant physics. This extension turns those explanations into separate, falsifiable tests.",
        "",
        "## What the current evidence can support",
        "",
        "| Question | Status | Current evidence | Causal claim supported? | Data needed next |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in evidence.itertuples(index=False):
        supported = "yes" if row.causal_conclusion_supported else "no"
        lines.append(
            f"| {row.question} | {row.current_status} | {row.current_evidence} | "
            f"{supported} | {row.required_next_data} |"
        )

    lines.extend(
        [
            "",
            "## Preregistered diagnostic experiments",
            "",
            "### 1. Optimization-basin sensitivity",
            "",
            "Train every method from the same paired set of at least 30 initialization seeds while fixing architecture, optimizer schedule, training budget, collocation set, and evaluation grid. Preserve checkpoints and loss trajectories. Compare rank-switch frequency across seeds and decompose within-method and between-method variation.",
            "",
            "A basin-sensitivity explanation is supported only if winner identity changes across paired initializations under otherwise fixed conditions and the between-seed variation is large relative to the reported method advantage.",
            "",
            "### 2. Collocation generalization",
            "",
            "For every trained model, evaluate PDE residuals on both the training collocation points and a much denser independently sampled holdout grid. Repeat across paired collocation draws at fixed point count and across point-count budgets at fixed compute.",
            "",
            "A collocation-overfitting explanation is supported only if low training residual coexists with systematically worse holdout residual and that generalization gap predicts rank reversals.",
            "",
            "### 3. Physical-invariant preservation",
            "",
            "Define diagnostics before training for each PDE family, including applicable conservation integrals, boundary and initial-condition residuals, symmetry constraints, and long-horizon drift. Evaluate them on common grids without selecting the diagnostic after seeing the leaderboard.",
            "",
            "Failure to capture invariant structure is supported only by direct, PDE-specific invariant violations. A change in L1, L2, or maximum-error rank is not sufficient evidence.",
            "",
            "### 4. Causal attribution",
            "",
            "Use a crossed design over method, initialization seed, collocation draw, and collocation budget. Keep evaluation points and compute accounting common. Estimate variance components and bootstrap the complete ranking procedure, not isolated model scores.",
            "",
            "The output should attribute observed rank variance to initialization, sampling, budget, and their interactions. If the design cannot separate these factors, the result remains a fragility observation rather than a mechanism claim.",
            "",
            "## Interpretation rule",
            "",
            "Detecting a rank change and explaining a rank change are different scientific tasks. The mechanism audit reports `not identifiable` whenever the required intervention or diagnostic is absent. This is an outcome, not missing decoration: it prevents a benchmark audit from claiming more than its evidence can carry.",
        ]
    )
    text = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return text
