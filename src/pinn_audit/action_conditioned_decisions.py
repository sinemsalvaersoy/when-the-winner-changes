"""Action-conditioned decision preservation under controlled PDE forcing.

The experiment separates three routes by which a representation can alter a
decision: compression of the parameter posterior, compression of the predicted
controlled response, and their combination.  Burgers receives proportional
body-force feedback and the damped wave receives active velocity damping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .representation_boundaries import (
    _burgers_initial,
    _burgers_qoi,
    _burgers_rhs,
    _burgers_trajectory,
    _case_parameters,
    _pod_fit,
    _pod_coordinates,
    _pod_reconstruct,
    _posterior,
    _wave_initial,
    _wave_qoi,
    _wave_trajectory,
)


@dataclass(frozen=True)
class ActionConfig:
    grid_points: int = 64
    pod_modes: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    calibration_cases: int = 8
    evaluation_cases: int = 10
    noise_draws: int = 30
    seed: int = 20261007
    burgers_dt: float = 0.002
    burgers_observation_times: tuple[float, ...] = (0.06, 0.12)
    burgers_decision_time: float = 0.24
    burgers_viscosities: tuple[float, ...] = (0.018, 0.024, 0.030, 0.036, 0.044, 0.052, 0.062)
    burgers_noise_sigma: float = 0.025
    burgers_control_gains: tuple[float, ...] = (0.0, 2.0, 5.0)
    wave_observation_times: tuple[float, ...] = (0.16, 0.30)
    wave_decision_time: float = 0.80
    wave_dampings: tuple[float, ...] = (0.04, 0.08, 0.12, 0.17, 0.23, 0.30, 0.38)
    wave_noise_sigma: float = 0.020
    wave_spatial_modes: int = 20
    wave_control_gains: tuple[float, ...] = (0.0, 0.25, 0.75)
    burgers_action_costs: tuple[float, ...] = (0.0, 0.08, 0.24)
    wave_action_costs: tuple[float, ...] = (0.0, 0.01, 0.08)
    burgers_hazard_quantile: float = 0.50
    wave_hazard_quantile: float = 0.25


@dataclass(frozen=True)
class ActionThresholds:
    action_disagreement: float = 0.05
    under_control_rate: float = 0.05
    normalized_regret: float = 0.02


def _controlled_burgers(
    initial: np.ndarray,
    viscosity: float,
    horizon: float,
    dt: float,
    control_gain: float,
) -> np.ndarray:
    """Integrate Burgers with proportional body force f=-gain*(u-mean(u))."""
    points = initial.size
    wave_numbers = np.fft.fftfreq(points, d=1 / points)
    steps = int(round(horizon / dt))
    if not np.isclose(steps * dt, horizon):
        raise ValueError("Burgers control horizon must be an integer multiple of dt")

    def rhs(field: np.ndarray) -> np.ndarray:
        feedback = -control_gain * (field - field.mean())
        return _burgers_rhs(field, viscosity, wave_numbers) + feedback

    field = initial.copy()
    for _ in range(steps):
        a = rhs(field)
        b = rhs(field + 0.5 * dt * a)
        c = rhs(field + 0.5 * dt * b)
        d = rhs(field + dt * c)
        field += dt * (a + 2 * b + 2 * c + d) / 6
    return field


def _wave_state(
    initial: np.ndarray,
    velocity: np.ndarray,
    damping: float,
    time: float,
    spatial_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return displacement and velocity for the unforced damped wave."""
    points = initial.size
    x = np.linspace(0, 1, points + 2)[1:-1]
    indices = np.arange(1, spatial_modes + 1)
    sines = np.sin(np.pi * indices[:, None] * x[None, :])
    normalization = 2.0 / (points + 1)
    q0 = normalization * (sines @ initial)
    v0 = normalization * (sines @ velocity)
    omega = np.sqrt(np.maximum((np.pi * indices) ** 2 - damping**2, 1e-12))
    cosine = np.cos(omega * time)
    sine = np.sin(omega * time)
    shifted = (v0 + damping * q0) / omega
    decay = np.exp(-damping * time)
    q = decay * (q0 * cosine + shifted * sine)
    qdot = decay * (
        -damping * (q0 * cosine + shifted * sine)
        - q0 * omega * sine
        + shifted * omega * cosine
    )
    return q @ sines, qdot @ sines


def _controlled_wave(
    initial: np.ndarray,
    velocity: np.ndarray,
    damping: float,
    horizon: float,
    spatial_modes: int,
    control_gain: float,
) -> np.ndarray:
    """Apply active velocity feedback, f=-2*gain*u_t, over the horizon."""
    return _wave_trajectory(
        initial,
        velocity,
        damping + control_gain,
        (horizon,),
        spatial_modes,
    )[0]


def _loss_matrix(
    qois: np.ndarray,
    hazard_threshold: float,
    action_costs: tuple[float, ...],
) -> np.ndarray:
    """Action loss combines normalized exceedance severity and control cost."""
    exceedance = np.maximum(qois / hazard_threshold - 1.0, 0.0)
    return np.square(exceedance) + np.asarray(action_costs)[None, :]


def _choose_action(posterior: np.ndarray, losses: np.ndarray) -> tuple[int, np.ndarray]:
    expected = posterior @ losses
    return int(np.argmin(expected)), expected


def _prepare_systems(config: ActionConfig, rng: np.random.Generator) -> tuple[dict[str, object], ...]:
    total = config.calibration_cases + config.evaluation_cases

    burgers_initials = np.asarray(
        [_burgers_initial(rng, config.grid_points) for _ in range(total)]
    )
    burgers_parameters = _case_parameters(config.burgers_viscosities, total)
    burgers_times = (0.0,) + config.burgers_observation_times
    burgers_libraries = np.asarray(
        [
            [
                _burgers_trajectory(initial, viscosity, burgers_times, config.burgers_dt)
                for viscosity in config.burgers_viscosities
            ]
            for initial in burgers_initials
        ]
    )
    burgers_true_indices = np.asarray(
        [config.burgers_viscosities.index(float(value)) for value in burgers_parameters]
    )
    burgers_truth = burgers_libraries[np.arange(total), burgers_true_indices]
    burgers_mean, burgers_basis = _pod_fit(
        burgers_libraries[: config.calibration_cases].reshape(-1, config.grid_points)
    )
    burgers_horizon = config.burgers_decision_time - config.burgers_observation_times[-1]
    burgers_calibration_qoi = []
    for case in range(config.calibration_cases):
        field = burgers_truth[case, -1]
        burgers_calibration_qoi.append(
            _burgers_qoi(
                _controlled_burgers(
                    field,
                    float(burgers_parameters[case]),
                    burgers_horizon,
                    config.burgers_dt,
                    0.0,
                )
            )
        )

    wave_pairs = [
        _wave_initial(rng, config.grid_points, config.wave_spatial_modes)
        for _ in range(total)
    ]
    wave_initials = np.asarray([pair[0] for pair in wave_pairs])
    wave_velocities = np.asarray([pair[1] for pair in wave_pairs])
    wave_parameters = _case_parameters(config.wave_dampings, total)
    wave_times = (0.0,) + config.wave_observation_times
    wave_libraries = np.asarray(
        [
            [
                _wave_trajectory(
                    initial,
                    velocity,
                    damping,
                    wave_times,
                    config.wave_spatial_modes,
                )
                for damping in config.wave_dampings
            ]
            for initial, velocity in zip(wave_initials, wave_velocities)
        ]
    )
    wave_true_indices = np.asarray(
        [config.wave_dampings.index(float(value)) for value in wave_parameters]
    )
    wave_truth = wave_libraries[np.arange(total), wave_true_indices]
    wave_mean, wave_basis = _pod_fit(
        np.concatenate(
            (
                wave_libraries[: config.calibration_cases].reshape(-1, config.grid_points),
                wave_velocities[: config.calibration_cases],
            ),
            axis=0,
        )
    )
    wave_horizon = config.wave_decision_time - config.wave_observation_times[-1]
    wave_calibration_qoi = []
    for case in range(config.calibration_cases):
        state, velocity = _wave_state(
            wave_initials[case],
            wave_velocities[case],
            float(wave_parameters[case]),
            config.wave_observation_times[-1],
            config.wave_spatial_modes,
        )
        wave_calibration_qoi.append(
            _wave_qoi(
                _controlled_wave(
                    state,
                    velocity,
                    float(wave_parameters[case]),
                    wave_horizon,
                    config.wave_spatial_modes,
                    0.0,
                )
            )
        )

    return (
        {
            "name": "burgers",
            "parameters": burgers_parameters,
            "parameter_grid": config.burgers_viscosities,
            "initials": burgers_initials,
            "velocities": None,
            "libraries": burgers_libraries,
            "truth": burgers_truth,
            "mean": burgers_mean,
            "basis": burgers_basis,
            "hazard_threshold": float(
                np.quantile(burgers_calibration_qoi, config.burgers_hazard_quantile)
            ),
            "noise_sigma": config.burgers_noise_sigma,
            "control_gains": config.burgers_control_gains,
            "action_costs": config.burgers_action_costs,
            "horizon": burgers_horizon,
        },
        {
            "name": "damped_wave",
            "parameters": wave_parameters,
            "parameter_grid": config.wave_dampings,
            "initials": wave_initials,
            "velocities": wave_velocities,
            "libraries": wave_libraries,
            "truth": wave_truth,
            "mean": wave_mean,
            "basis": wave_basis,
            "hazard_threshold": float(
                np.quantile(wave_calibration_qoi, config.wave_hazard_quantile)
            ),
            "noise_sigma": config.wave_noise_sigma,
            "control_gains": config.wave_control_gains,
            "action_costs": config.wave_action_costs,
            "horizon": wave_horizon,
        },
    )


def _action_qois(
    system: dict[str, object],
    case_index: int,
    config: ActionConfig,
    modes: int | None,
) -> np.ndarray:
    """Return parameter-by-action QoIs, optionally after POD reconstruction."""
    qois = []
    mean = system["mean"]
    basis = system["basis"] if modes is None else system["basis"][:modes]
    for parameter_index, parameter in enumerate(system["parameter_grid"]):
        if system["name"] == "burgers":
            state = system["libraries"][case_index, parameter_index, -1]
            if modes is not None:
                state = _pod_reconstruct(state[None, :], mean, basis)[0]
            action_values = [
                _burgers_qoi(
                    _controlled_burgers(
                        state,
                        float(parameter),
                        float(system["horizon"]),
                        config.burgers_dt,
                        float(gain),
                    )
                )
                for gain in system["control_gains"]
            ]
        else:
            state, velocity = _wave_state(
                system["initials"][case_index],
                system["velocities"][case_index],
                float(parameter),
                config.wave_observation_times[-1],
                config.wave_spatial_modes,
            )
            if modes is not None:
                state = _pod_reconstruct(state[None, :], mean, basis)[0]
                velocity = _pod_reconstruct(velocity[None, :], mean, basis)[0]
            action_values = [
                _wave_qoi(
                    _controlled_wave(
                        state,
                        velocity,
                        float(parameter),
                        float(system["horizon"]),
                        config.wave_spatial_modes,
                        float(gain),
                    )
                )
                for gain in system["control_gains"]
            ]
        qois.append(action_values)
    return np.asarray(qois)


def run_action_audit(
    config: ActionConfig = ActionConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired action-conditioned decisions for both PDE families."""
    if len(config.burgers_action_costs) != 3 or len(config.wave_action_costs) != 3:
        raise ValueError("each PDE must define costs for no, moderate, and strong control")
    if max(config.pod_modes) > config.grid_points:
        raise ValueError("POD modes cannot exceed grid_points")
    rng = np.random.default_rng(config.seed)
    aggregate: list[dict[str, object]] = []
    paired: list[dict[str, object]] = []
    pathways = ("inference_only", "response_only", "combined")

    for system in _prepare_systems(config, rng):
        start = config.calibration_cases
        sigma = float(system["noise_sigma"])
        threshold = float(system["hazard_threshold"])
        for local_case, case_index in enumerate(range(start, start + config.evaluation_cases)):
            observation_truth = system["truth"][case_index, 1:]
            candidates = system["libraries"][case_index, :, 1:, :]
            noise = rng.normal(0, sigma, size=(config.noise_draws,) + observation_truth.shape)
            grid_predictions = candidates.reshape(len(candidates), -1)
            grid_posteriors = np.asarray(
                [
                    _posterior((observation_truth + draw).reshape(-1), grid_predictions, sigma)
                    for draw in noise
                ]
            )
            grid_qois = _action_qois(system, case_index, config, modes=None)
            grid_losses = _loss_matrix(grid_qois, threshold, system["action_costs"])
            true_index = list(system["parameter_grid"]).index(
                float(system["parameters"][case_index])
            )
            true_best_action = int(np.argmin(grid_losses[true_index]))
            grid_choices = np.asarray(
                [_choose_action(posterior, grid_losses)[0] for posterior in grid_posteriors]
            )

            for modes in config.pod_modes:
                basis = system["basis"][:modes]
                candidate_coordinates = np.asarray(
                    [_pod_coordinates(fields, system["mean"], basis).reshape(-1) for fields in candidates]
                )
                reduced_posteriors = np.asarray(
                    [
                        _posterior(
                            _pod_coordinates(observation_truth + draw, system["mean"], basis).reshape(-1),
                            candidate_coordinates,
                            sigma,
                        )
                        for draw in noise
                    ]
                )
                reduced_qois = _action_qois(system, case_index, config, modes=modes)
                reduced_losses = _loss_matrix(reduced_qois, threshold, system["action_costs"])
                action_qoi_relative_error = float(
                    np.linalg.norm(reduced_qois - grid_qois) / np.linalg.norm(grid_qois)
                )

                choices_by_pathway: dict[str, np.ndarray] = {}
                for pathway in pathways:
                    posteriors = grid_posteriors if pathway == "response_only" else reduced_posteriors
                    losses = grid_losses if pathway == "inference_only" else reduced_losses
                    choices = []
                    regrets = []
                    for draw_index, posterior in enumerate(posteriors):
                        choice, _ = _choose_action(posterior, losses)
                        grid_expected = grid_posteriors[draw_index] @ grid_losses
                        regret = float(grid_expected[choice] - grid_expected.min())
                        choices.append(choice)
                        regrets.append(max(regret, 0.0))
                        paired.append(
                            {
                                "pde": system["name"],
                                "case": local_case,
                                "pod_modes": modes,
                                "noise_draw": draw_index,
                                "pathway": pathway,
                                "grid_action": int(grid_choices[draw_index]),
                                "representation_action": choice,
                                "under_control": bool(choice < grid_choices[draw_index]),
                                "grid_expected_regret": max(regret, 0.0),
                                "true_best_action": true_best_action,
                                "grid_truth_disagreement": bool(
                                    grid_choices[draw_index] != true_best_action
                                ),
                                "representation_truth_disagreement": bool(
                                    choice != true_best_action
                                ),
                            }
                        )
                    choices_array = np.asarray(choices)
                    choices_by_pathway[pathway] = choices_array
                    normalizer = 1.0 + np.asarray(
                        [(grid_posteriors[index] @ grid_losses).min() for index in range(config.noise_draws)]
                    )
                    normalized_regrets = np.asarray(regrets) / normalizer
                    aggregate.append(
                        {
                            "pde": system["name"],
                            "case": local_case,
                            "true_parameter": float(system["parameters"][case_index]),
                            "pod_modes": modes,
                            "pathway": pathway,
                            "action_disagreement": float(np.mean(choices_array != grid_choices)),
                            "under_control_rate": float(np.mean(choices_array < grid_choices)),
                            "over_control_rate": float(np.mean(choices_array > grid_choices)),
                            "mean_normalized_regret": float(np.mean(normalized_regrets)),
                            "maximum_normalized_regret": float(np.max(normalized_regrets)),
                            "action_qoi_relative_error": action_qoi_relative_error,
                            "hazard_threshold": threshold,
                            "grid_no_control_rate": float(np.mean(grid_choices == 0)),
                            "grid_moderate_control_rate": float(np.mean(grid_choices == 1)),
                            "grid_strong_control_rate": float(np.mean(grid_choices == 2)),
                        }
                    )
    return pd.DataFrame(aggregate), pd.DataFrame(paired)


def extract_action_boundaries(
    results: pd.DataFrame,
    thresholds: ActionThresholds = ActionThresholds(),
) -> pd.DataFrame:
    """Find first and stable decision-preserving rank for each causal pathway."""
    records: list[dict[str, object]] = []
    for (pde, case, pathway), group in results.groupby(["pde", "case", "pathway"], sort=True):
        ordered = group.sort_values("pod_modes")
        ranks = ordered["pod_modes"].to_numpy(dtype=int)
        passes = (
            (ordered["action_disagreement"].to_numpy() <= thresholds.action_disagreement)
            & (ordered["under_control_rate"].to_numpy() <= thresholds.under_control_rate)
            & (ordered["mean_normalized_regret"].to_numpy() <= thresholds.normalized_regret)
        )
        first = int(np.flatnonzero(passes)[0]) if passes.any() else None
        stable_candidates = [index for index in range(len(passes)) if bool(np.all(passes[index:]))]
        stable = stable_candidates[0] if stable_candidates else None
        records.append(
            {
                "pde": pde,
                "case": case,
                "pathway": pathway,
                "first_passing_rank": int(ranks[first]) if first is not None else np.nan,
                "stable_boundary_rank": int(ranks[stable]) if stable is not None else np.nan,
                "boundary_width_steps": stable - first if first is not None and stable is not None else np.nan,
                "order_violations": int(np.sum(passes[:-1] & ~passes[1:])),
                "resolved": stable is not None,
            }
        )
    return pd.DataFrame(records)


def summarize_action_boundaries(boundaries: pd.DataFrame) -> pd.DataFrame:
    return (
        boundaries.groupby(["pde", "pathway"], as_index=False)
        .agg(
            cases=("case", "nunique"),
            resolved_fraction=("resolved", "mean"),
            median_stable_rank=("stable_boundary_rank", "median"),
            maximum_stable_rank=("stable_boundary_rank", "max"),
            cases_with_order_violations=("order_violations", lambda values: int((values > 0).sum())),
        )
    )


def action_specification(
    config: ActionConfig = ActionConfig(),
    thresholds: ActionThresholds = ActionThresholds(),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("actions", "ordered control levels", "none, moderate, strong"),
            ("burgers_control", "body force", "-gain*(u-mean(u))"),
            ("wave_control", "active velocity feedback", "-2*gain*u_t"),
            ("burgers_action_costs", "none, moderate, strong", str(config.burgers_action_costs)),
            ("wave_action_costs", "none, moderate, strong", str(config.wave_action_costs)),
            ("burgers_hazard_threshold", "calibration quantile of uncontrolled QoI", str(config.burgers_hazard_quantile)),
            ("wave_hazard_threshold", "calibration quantile of uncontrolled QoI", str(config.wave_hazard_quantile)),
            ("action_disagreement", "mean disagreement from paired full-grid action", str(thresholds.action_disagreement)),
            ("under_control", "representation selects weaker action than full grid", str(thresholds.under_control_rate)),
            ("normalized_regret", "grid expected regret divided by one plus optimal loss", str(thresholds.normalized_regret)),
        ],
        columns=("criterion", "definition", "value"),
    )


def plot_action_boundaries(boundaries: pd.DataFrame, path: Path) -> None:
    pathways = ("inference_only", "response_only", "combined")
    pdes = ("burgers", "damped_wave")
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.9), sharex=True)
    finite = boundaries["stable_boundary_rank"].dropna()
    maximum = int(finite.max()) if len(finite) else 64
    for ax, pde in zip(axes, pdes):
        subset = boundaries[boundaries["pde"].eq(pde)]
        matrix = subset.pivot(index="pathway", columns="case", values="stable_boundary_rank").reindex(pathways)
        display = matrix.fillna(maximum * 1.25).to_numpy(dtype=float)
        image = ax.imshow(display, aspect="auto", cmap="viridis_r", vmin=2, vmax=maximum * 1.25)
        ax.set_yticks(range(len(pathways)), [value.replace("_", " ").title() for value in pathways])
        ax.set_title(pde.replace("_", " ").title(), loc="left", weight="bold")
        for row in range(display.shape[0]):
            for column in range(display.shape[1]):
                value = matrix.iloc[row, column]
                ax.text(column, row, "NR" if pd.isna(value) else str(int(value)), ha="center", va="center", fontsize=8)
    axes[-1].set_xlabel("Evaluation case")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Stable POD boundary rank")
    fig.suptitle("Decision loss can enter through inference or controlled response", x=0.08, ha="left", weight="bold")
    fig.subplots_adjust(left=0.20, right=0.88, top=0.89, hspace=0.32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_action_outputs(
    output: Path,
    config: ActionConfig = ActionConfig(),
    thresholds: ActionThresholds = ActionThresholds(),
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    results, paired = run_action_audit(config)
    boundaries = extract_action_boundaries(results, thresholds)
    results.to_csv(output / "action_conditioned_runs.csv", index=False)
    paired.to_csv(output / "action_conditioned_noise_draws.csv", index=False)
    boundaries.to_csv(output / "action_conditioned_boundaries.csv", index=False)
    summarize_action_boundaries(boundaries).to_csv(output / "action_conditioned_summary.csv", index=False)
    action_specification(config, thresholds).to_csv(output / "action_conditioned_specification.csv", index=False)
    plot_action_boundaries(boundaries, output / "action_conditioned_boundaries.png")
