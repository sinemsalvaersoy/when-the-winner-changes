import pandas as pd

from pinn_audit.analysis import compare_sources, rank_switch_summary, rerun_winner_probabilities


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

