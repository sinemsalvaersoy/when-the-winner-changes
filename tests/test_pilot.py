import pandas as pd
import pytest

from pinn_audit.pilot import pilot_mechanism_summary, validate_pilot_results


def _paired_results() -> pd.DataFrame:
    rows = []
    for seed, scores in ((1, {"adam": 0.10, "lra": 0.20}), (2, {"adam": 0.30, "lra": 0.15})):
        for method, score in scores.items():
            rows.append(
                {
                    "case": "C",
                    "method": method,
                    "seed": seed,
                    "collocation_draw": 1,
                    "collocation_points": 512,
                    "train_residual": score / 4,
                    "holdout_residual": score / 2,
                    "l2re": score,
                    "runtime_seconds": 1.0,
                    "status": "completed",
                }
            )
    return pd.DataFrame(rows)


def test_pilot_detects_winner_switch_across_paired_seeds():
    result = pilot_mechanism_summary(_paired_results())
    assert bool(result.iloc[0]["winner_switch_observed"])
    assert result.iloc[0]["distinct_winners"] == 2
    assert result.iloc[0]["paired_blocks"] == 2


def test_pilot_rejects_unpaired_method_results():
    data = _paired_results().iloc[:-1]
    with pytest.raises(ValueError, match="not paired"):
        validate_pilot_results(data, ("adam", "lra"))


def test_pilot_rejects_failed_runs():
    data = _paired_results()
    data.loc[0, "status"] = "failed"
    with pytest.raises(ValueError, match="did not complete"):
        validate_pilot_results(data, ("adam", "lra"))
