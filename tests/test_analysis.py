import pandas as pd

from pinn_audit.analysis import (
    compare_sources,
    distribution_sensitivity,
    mechanism_evidence_matrix,
    rank_switch_summary,
    rerun_winner_probabilities,
)


def test_source_conflict_is_detected():
    data = pd.DataFrame(
        [
            {"source": "main_table", "metric": "L2RE", "family": "Heat", "case": "HInv", "method": "vPINN", "mean": 0.0119, "std": None},
            {"source": "appendix_table", "metric": "L2RE", "family": "Heat", "case": "HInv", "method": "vPINN", "mean": 0.456, "std": 0.013},
        ]
    )
    result = compare_sources(data)
    assert len(result) == 1
    assert not bool(result.iloc[0]["consistent"])
    assert result.iloc[0]["relative_gap"] > 30


def test_clear_winner_gets_high_probability():
    data = pd.DataFrame(
        [
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "A", "mean": 0.1, "std": 0.005},
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "B", "mean": 0.3, "std": 0.005},
        ]
    )
    result = rerun_winner_probabilities(data, draws=20_000)
    assert result.iloc[0]["method"] == "A"
    assert result.iloc[0]["winner_probability"] > 0.99


def test_metric_switch_is_detected():
    data = pd.DataFrame(
        [
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "A", "mean": 0.1},
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "B", "mean": 0.2},
            {"source": "appendix_table", "metric": "mERR", "family": "P", "case": "C", "method": "A", "mean": 0.7},
            {"source": "appendix_table", "metric": "mERR", "family": "P", "case": "C", "method": "B", "mean": 0.3},
        ]
    )
    result = rank_switch_summary(data)
    assert bool(result.iloc[0]["metric_sensitive"])
    assert result.iloc[0]["distinct_winners"] == 2


def test_distribution_sensitivity_exposes_model_dependence():
    data = pd.DataFrame(
        [
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "A", "mean": 0.10, "std": 0.04},
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "B", "mean": 0.13, "std": 0.04},
        ]
    )
    result = distribution_sensitivity(data, draws=30_000)
    assert result.iloc[0]["winner_consistent_across_models"]
    assert result.iloc[0]["probability_range"] > 0


def test_missing_results_do_not_enter_fresh_run_simulation():
    data = pd.DataFrame(
        [
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "A", "mean": 0.10, "std": 0.01},
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "B", "mean": None, "std": None},
        ]
    )
    result = rerun_winner_probabilities(data, draws=1_000)
    assert list(result["method"]) == ["A"]
    assert result.iloc[0]["winner_probability"] == 1.0


def test_mechanism_matrix_does_not_infer_causality_from_aggregate_tables():
    data = pd.DataFrame(
        [
            {"source": "main_table", "metric": "L2RE", "family": "P", "case": "C", "method": "A", "mean": 0.10, "std": None},
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "A", "mean": 0.11, "std": 0.01},
            {"source": "appendix_table", "metric": "L2RE", "family": "P", "case": "C", "method": "B", "mean": 0.20, "std": 0.01},
            {"source": "appendix_table", "metric": "mERR", "family": "P", "case": "C", "method": "A", "mean": 0.40, "std": 0.02},
            {"source": "appendix_table", "metric": "mERR", "family": "P", "case": "C", "method": "B", "mean": 0.30, "std": 0.02},
        ]
    )
    collocation = pd.DataFrame(
        [
            {"case": "C", "collocation_points": 10, "method": "A", "mean": 0.1},
            {"case": "C", "collocation_points": 10, "method": "B", "mean": 0.2},
            {"case": "C", "collocation_points": 20, "method": "A", "mean": 0.2},
            {"case": "C", "collocation_points": 20, "method": "B", "mean": 0.1},
        ]
    )
    result = mechanism_evidence_matrix(data, collocation)
    assert not result["causal_conclusion_supported"].any()
    assert result.loc[
        result["question"].eq("optimization-basin sensitivity"), "current_status"
    ].item() == "not identifiable"
    assert result.loc[
        result["question"].eq("collocation overfitting"), "current_status"
    ].item() == "not identifiable"
