import numpy as np

from pinn_audit.representation_pilot import (
    BurgersConfig,
    RobustnessConfig,
    conditions_table,
    run_representation_pilot,
    run_robustness_experiment,
    solve_burgers,
    summarize_pilot,
    summarize_robustness,
    winner_probabilities,
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


def test_small_robustness_sweep_is_paired_and_complete():
    base = BurgersConfig(
        grid_points=16,
        train_samples=12,
        test_samples=6,
        time_horizon=0.005,
        time_step=0.001,
    )
    robustness = RobustnessConfig(
        grid_points=(16,),
        pod_modes=(2, 4),
        seeds=(3, 5),
        bootstrap_draws=100,
    )
    results = run_robustness_experiment(robustness, base)
    assert len(results) == 2 * 2 * 3 * 2
    assert set(results["representation"]) == {"grid", "pod_l2", "pod_gradient"}
    assert set(results["model"]) == {"ridge", "knn"}
    assert np.isfinite(
        results[
            [
                "relative_l2",
                "steepest_gradient_location_error_radians",
                "output_l2_fraction_retained",
                "output_gradient_fraction_retained",
            ]
        ]
    ).all().all()
    assert (summarize_robustness(results)["trials"] == 2).all()


def test_winner_probabilities_partition_each_condition():
    base = BurgersConfig(
        grid_points=16,
        train_samples=12,
        test_samples=6,
        time_horizon=0.005,
        time_step=0.001,
    )
    robustness = RobustnessConfig(
        grid_points=(16,),
        pod_modes=(2,),
        seeds=(3, 5),
        bootstrap_draws=100,
    )
    probabilities = winner_probabilities(
        run_robustness_experiment(robustness, base),
        robustness,
    )
    totals = probabilities.groupby(
        ["grid_points", "pod_modes", "representation", "metric"]
    )["winner_probability"].sum()
    assert np.allclose(totals, 1.0)
    assert probabilities["bootstrap_ci_low"].between(0, 1).all()
    assert probabilities["bootstrap_ci_high"].between(0, 1).all()
    assert probabilities["wilson_ci_low"].between(0, 1).all()
    assert probabilities["wilson_ci_high"].between(0, 1).all()
