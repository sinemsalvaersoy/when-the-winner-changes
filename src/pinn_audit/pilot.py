"""Validation and diagnostics for paired PINN rerun experiments."""

from __future__ import annotations

import pandas as pd


PILOT_COLUMNS = (
    "case",
    "method",
    "seed",
    "collocation_draw",
    "collocation_points",
    "train_residual",
    "holdout_residual",
    "l2re",
    "runtime_seconds",
    "status",
)

DESIGN_KEYS = ("case", "seed", "collocation_draw", "collocation_points")


def validate_pilot_results(data: pd.DataFrame, expected_methods: tuple[str, ...]) -> None:
    """Reject incomplete, duplicated, or unpaired rerun results."""
    missing = sorted(set(PILOT_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"missing pilot columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("pilot results are empty")
    if data[list(DESIGN_KEYS) + ["method"]].duplicated().any():
        raise ValueError("duplicate method result within a paired design block")
    if not data["status"].eq("completed").all():
        raise ValueError("pilot contains runs that did not complete")
    if (data[["train_residual", "holdout_residual", "l2re"]] < 0).any().any():
        raise ValueError("error and residual values must be nonnegative")

    observed = data.groupby(list(DESIGN_KEYS))["method"].agg(lambda x: frozenset(x))
    expected = frozenset(expected_methods)
    if not observed.map(lambda methods: methods == expected).all():
        raise ValueError("methods are not paired within every design block")


def pilot_mechanism_summary(
    data: pd.DataFrame, expected_methods: tuple[str, ...] = ("adam", "lra")
) -> pd.DataFrame:
    """Summarize rank switches and residual generalization gaps by PDE case."""
    validate_pilot_results(data, expected_methods)
    frame = data.copy()
    frame["generalization_gap"] = frame["holdout_residual"] - frame["train_residual"]

    winners = frame.loc[
        frame.groupby(list(DESIGN_KEYS))["l2re"].idxmin(),
        list(DESIGN_KEYS) + ["method"],
    ]
    records: list[dict[str, object]] = []
    for case, group in frame.groupby("case", sort=True):
        case_winners = winners[winners["case"].eq(case)]
        counts = case_winners["method"].value_counts(normalize=True)
        modal_share = float(counts.iloc[0])
        method_gaps = group.groupby("method")["generalization_gap"].mean()
        records.append(
            {
                "case": case,
                "paired_blocks": len(case_winners),
                "distinct_winners": int(case_winners["method"].nunique()),
                "winner_switch_observed": bool(case_winners["method"].nunique() > 1),
                "modal_winner": str(counts.index[0]),
                "modal_winner_share": modal_share,
                "mean_generalization_gap": float(group["generalization_gap"].mean()),
                "largest_method_gap": str(method_gaps.idxmax()),
                "largest_mean_generalization_gap": float(method_gaps.max()),
            }
        )
    return pd.DataFrame.from_records(records)
