from pathlib import Path

import pandas as pd

from pinn_audit.analysis import (
    collocation_switch_summary,
    compare_sources,
    distribution_sensitivity,
    mechanism_evidence_matrix,
    rank_switch_summary,
    rerun_winner_probabilities,
)


ROOT = Path(__file__).parents[1]


def test_versioned_case_study_counts_and_headline_conflict():
    data = pd.read_csv(ROOT / "data" / "pinnacle_published_metrics.csv")
    collocation = pd.read_csv(ROOT / "data" / "pinnacle_collocation_ablation.csv")
    consistency = compare_sources(data)
    conflicts = consistency[~consistency["consistent"]]
    stability = rerun_winner_probabilities(data, draws=20_000)
    leaders = stability.groupby(["family", "case"], sort=False).head(1)
    switches = rank_switch_summary(data)
    budgets = collocation_switch_summary(collocation)
    assumptions = distribution_sensitivity(data, draws=20_000)
    mechanisms = mechanism_evidence_matrix(data, collocation)

    assert len(consistency) == 235
    assert len(conflicts) == 2
    hinv = conflicts[(conflicts["case"] == "HInv") & (conflicts["method"] == "vPINN")].iloc[0]
    assert hinv["main_table"] == 0.0119
    assert hinv["appendix_table"] == 0.456
    assert (leaders["winner_probability"] < 0.80).sum() == 12
    assert switches["metric_sensitive"].sum() == 13
    assert budgets["budget_sensitive"].sum() == 4
    assert assumptions["winner_consistent_across_models"].all()
    assert mechanisms.loc[
        mechanisms["question"].eq("collocation-budget dependence"), "current_status"
    ].item() == "observed fragility"
    assert not mechanisms["causal_conclusion_supported"].any()
