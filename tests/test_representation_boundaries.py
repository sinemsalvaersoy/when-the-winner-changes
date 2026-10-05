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
