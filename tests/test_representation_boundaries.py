import numpy as np
import pandas as pd

from pinn_audit.representation_boundaries import (
    BoundaryConfig,
    BoundaryThresholds,
    _posterior,
    extract_boundaries,
    run_boundary_audit,
    summarize_boundaries,
)


def test_posterior_is_normalized_and_prefers_the_nearest_candidate():
    candidates = np.asarray([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])
    posterior = _posterior(np.asarray([0.9, 1.1]), candidates, sigma=0.3)
    assert np.isclose(posterior.sum(), 1.0)
    assert posterior.argmax() == 1


def test_stable_boundary_does_not_confuse_a_temporary_pass_with_recovery():
    rows = pd.DataFrame(
        {
            "pde": ["test"] * 4,
            "case": [0] * 4,
            "pod_modes": [2, 4, 8, 16],
            "state_relative_l2": [0.10, 0.04, 0.08, 0.03],
            "dynamics_relative_l2": [0.3, 0.2, 0.1, 0.05],
            "posterior_total_variation": [0.3, 0.2, 0.08, 0.04],
            "action_disagreement": [0.3, 0.2, 0.04, 0.01],
            "false_safe_rate": [0.2, 0.1, 0.04, 0.01],
        }
    )
    boundaries = extract_boundaries(rows)
    state = boundaries[boundaries["preservation"].eq("state")].iloc[0]
    assert state["first_passing_rank"] == 4
    assert state["stable_boundary_rank"] == 16
    assert state["boundary_width_steps"] == 2
    assert state["order_violations"] == 1


def test_small_cross_pde_audit_is_paired_and_complete():
    config = BoundaryConfig(
        grid_points=16,
        pod_modes=(2, 4, 8, 16),
        calibration_cases=3,
        evaluation_cases=2,
        noise_draws=3,
        wave_spatial_modes=8,
    )
    results, noise = run_boundary_audit(config)
    assert set(results["pde"]) == {"burgers", "damped_wave"}
    assert len(results) == 2 * 2 * 4
    assert len(noise) == 2 * 2 * 4 * 3
    paired = noise.groupby(["pde", "case", "pod_modes"])["noise_draw"].nunique()
    assert (paired == 3).all()
    assert np.isfinite(
        results[
            [
                "state_relative_l2",
                "dynamics_relative_l2",
                "posterior_total_variation",
                "action_disagreement",
            ]
        ]
    ).all().all()


def test_full_rank_is_an_exact_state_negative_control():
    config = BoundaryConfig(
        grid_points=16,
        pod_modes=(4, 8, 16),
        calibration_cases=4,
        evaluation_cases=2,
        noise_draws=2,
        wave_spatial_modes=8,
    )
    results, _ = run_boundary_audit(config)
    full_rank = results[results["pod_modes"].eq(16)]
    assert (full_rank["state_relative_l2"] < 1e-10).all()
    assert (full_rank["dynamics_relative_l2"] < 1e-9).all()


def test_boundary_summary_retains_each_preservation_axis():
    rows = pd.DataFrame(
        {
            "pde": ["x"] * 4,
            "case": [0] * 4,
            "preservation": ["state", "dynamics", "inference", "decision"],
            "stable_boundary_rank": [4, 8, 16, np.nan],
            "resolved": [True, True, True, False],
            "order_violations": [0, 0, 1, 0],
        }
    )
    summary = summarize_boundaries(rows)
    assert set(summary["preservation"]) == {
        "state",
        "dynamics",
        "inference",
        "decision",
    }
    assert summary.loc[summary["preservation"].eq("decision"), "resolved_fraction"].iloc[0] == 0


def test_headline_burgers_case_passes_posterior_rule_but_changes_decision():
    """Reproduce README case 7 with the versioned cases and paired noise."""
    # Keep the full audit: both systems consume RNG draws before evaluation.
    config = BoundaryConfig(
        seed=20261005,
        grid_points=64,
        pod_modes=(2, 4, 8, 16, 32, 64),
        calibration_cases=8,
        evaluation_cases=10,
        noise_draws=30,
    )
    thresholds = BoundaryThresholds()
    results, noise = run_boundary_audit(config)
    case = results[
        results["pde"].eq("burgers")
        & results["case"].eq(7)
        & results["pod_modes"].eq(16)
    ]
    assert len(case) == 1
    row = case.iloc[0]
    paired = noise[
        noise["pde"].eq("burgers")
        & noise["case"].eq(7)
        & noise["pod_modes"].eq(16)
    ]
    assert len(paired) == config.noise_draws
    assert paired["noise_draw"].nunique() == config.noise_draws
    disagreements = paired["grid_action"] != paired["representation_action"]
    assert int(disagreements.sum()) == 2
    assert np.isclose(
        row["posterior_total_variation"], 0.0689994583503128,
        rtol=0, atol=1e-8,
    )
    assert np.isclose(
        row["posterior_total_variation"],
        paired["posterior_total_variation"].mean(),
    )
    assert np.isclose(row["action_disagreement"], 2 / 30)
    assert row["posterior_total_variation"] <= thresholds.posterior_total_variation
    assert row["action_disagreement"] > thresholds.action_disagreement


