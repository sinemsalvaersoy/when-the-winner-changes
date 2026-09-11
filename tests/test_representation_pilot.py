import numpy as np

from pinn_audit.representation_pilot import (
    BurgersConfig,
    conditions_table,
    run_representation_pilot,
    solve_burgers,
    summarize_pilot,
)


def test_burgers_solver_preserves_a_constant_state():
    config = BurgersConfig(grid_points=16, time_horizon=0.01, time_step=0.001)
    initial = np.full(config.grid_points, 0.7)
    assert np.allclose(solve_burgers(initial, config), initial)


def test_small_pilot_is_complete_and_paired():
    config = BurgersConfig(
        grid_points=24,
        train_samples=16,
        test_samples=8,
        time_horizon=0.01,
        pod_modes=3,
        seeds=(7,),
    )
    results = run_representation_pilot(config)
    assert len(results) == 4
    assert set(results["representation"]) == {"grid", "pod"}
    assert set(results["model"]) == {"ridge", "knn"}
    assert np.isfinite(
        results[["relative_l2", "steepest_gradient_location_error_radians"]]
    ).all().all()


def test_versioned_pilot_exposes_a_shock_location_ranking_reversal():
    results = run_representation_pilot()
    summary = summarize_pilot(results)
    winners = summary[
        summary["winner_steepest_gradient_location_error_radians"]
    ].set_index("representation")
    assert winners.loc["grid", "model"] == "knn"
    assert winners.loc["pod", "model"] == "ridge"


def test_conditions_identify_the_representation_as_changed():
    conditions = conditions_table()
    changed = conditions[conditions["status"].eq("Changed")]
    assert changed["factor"].tolist() == ["Input and output encoding"]
