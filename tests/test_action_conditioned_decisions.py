import numpy as np
import pandas as pd

from pinn_audit.action_conditioned_decisions import (
    ActionConfig,
    _controlled_burgers,
    _controlled_wave,
    _wave_state,
    extract_action_boundaries,
    run_action_audit,
)
from pinn_audit.representation_boundaries import (
    _burgers_initial,
    _burgers_trajectory,
    _wave_initial,
    _wave_trajectory,
)


def test_zero_burgers_control_matches_unforced_continuation():
    rng = np.random.default_rng(4)
    initial = _burgers_initial(rng, 32)
    state = _burgers_trajectory(initial, 0.03, (0.12,), 0.002)[0]
    controlled = _controlled_burgers(state, 0.03, 0.12, 0.002, 0.0)
    direct = _burgers_trajectory(initial, 0.03, (0.24,), 0.002)[0]
    assert np.allclose(controlled, direct, atol=1e-11)


def test_zero_wave_control_matches_unforced_continuation():
    rng = np.random.default_rng(7)
    initial, velocity = _wave_initial(rng, 32, 12)
    state, state_velocity = _wave_state(initial, velocity, 0.12, 0.30, 12)
    controlled = _controlled_wave(state, state_velocity, 0.12, 0.50, 12, 0.0)
    direct = _wave_trajectory(initial, velocity, 0.12, (0.80,), 12)[0]
    assert np.allclose(controlled, direct, atol=1e-10)


def test_small_action_audit_is_paired_and_separates_pathways():
    config = ActionConfig(
        grid_points=16,
        pod_modes=(2, 4, 8, 16),
        calibration_cases=3,
        evaluation_cases=2,
        noise_draws=3,
        wave_spatial_modes=8,
    )
    results, paired = run_action_audit(config)
    assert set(results["pde"]) == {"burgers", "damped_wave"}
    assert set(results["pathway"]) == {
        "inference_only",
        "response_only",
        "combined",
    }
    assert len(results) == 2 * 2 * 4 * 3
    assert len(paired) == 2 * 2 * 4 * 3 * 3
    draws = paired.groupby(["pde", "case", "pod_modes", "pathway"])[
        "noise_draw"
    ].nunique()
    assert (draws == 3).all()


def test_full_rank_is_an_exact_action_negative_control():
    config = ActionConfig(
        grid_points=16,
        pod_modes=(4, 8, 16),
        calibration_cases=4,
        evaluation_cases=2,
        noise_draws=2,
        wave_spatial_modes=8,
    )
    results, _ = run_action_audit(config)
    full_rank = results[results["pod_modes"].eq(16)]
    assert (full_rank["action_disagreement"] == 0).all()
    assert (full_rank["under_control_rate"] == 0).all()
    assert (full_rank["mean_normalized_regret"] < 1e-12).all()
    assert (full_rank["action_qoi_relative_error"] < 1e-10).all()


def test_action_boundary_keeps_temporary_pass_visible():
    rows = []
    for rank, disagreement in zip((2, 4, 8, 16), (0.2, 0.04, 0.1, 0.0)):
        rows.append(
            {
                "pde": "test",
                "case": 0,
                "pathway": "combined",
                "pod_modes": rank,
                "action_disagreement": disagreement,
                "under_control_rate": disagreement,
                "mean_normalized_regret": 0.0,
            }
        )
    boundary = extract_action_boundaries(pd.DataFrame(rows)).iloc[0]
    assert boundary["first_passing_rank"] == 4
    assert boundary["stable_boundary_rank"] == 16
    assert boundary["order_violations"] == 1
