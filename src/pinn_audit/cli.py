"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis import (
    collocation_switch_summary,
    compare_sources,
    rank_switch_summary,
    rerun_winner_probabilities,
)
from .extract import download_tex, extract_collocation_ablation, extract_pinnacle_tables
from .plots import (
    plot_collocation_sensitivity,
    plot_metric_sensitivity,
    plot_source_conflicts,
    plot_winner_stability,
)
from .report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit scientific benchmark claims")
    parser.add_argument("--input", type=Path, default=Path("data/pinnacle_published_metrics.csv"))
    parser.add_argument("--collocation-input", type=Path, default=Path("data/pinnacle_collocation_ablation.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--extract", action="store_true", help="Download and re-extract PINNacle v2 tables")
    args = parser.parse_args()

    if args.extract:
        tex = download_tex()
        data = extract_pinnacle_tables(tex)
        collocation = extract_collocation_ablation(tex)
        args.input.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.input, index=False)
        collocation.to_csv(args.collocation_input, index=False)
    else:
        data = pd.read_csv(args.input)
        collocation = pd.read_csv(args.collocation_input)
    args.output.mkdir(parents=True, exist_ok=True)
    compare_sources(data).to_csv(args.output / "source_consistency.csv", index=False)
    rerun_winner_probabilities(data).to_csv(args.output / "winner_probabilities.csv", index=False)
    rank_switch_summary(data).to_csv(args.output / "metric_sensitivity.csv", index=False)
    collocation_switch_summary(collocation).to_csv(
        args.output / "collocation_sensitivity.csv", index=False
    )
    build_report(data, args.output / "AUDIT_REPORT.md", collocation)
    plot_source_conflicts(data, args.output / "source_conflicts.png")
    plot_winner_stability(data, args.output / "winner_stability.png")
    plot_metric_sensitivity(data, args.output / "metric_sensitivity.png")
    plot_collocation_sensitivity(collocation, args.output / "collocation_sensitivity.png")


if __name__ == "__main__":
    main()
