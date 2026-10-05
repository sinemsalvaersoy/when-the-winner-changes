"""Case-level preservation boundaries for Burgers and damped-wave systems.

The audit asks how much POD capacity is required before four distinct claims
remain valid: state reconstruction, forward dynamics, parameter inference, and
an operational decision.  All modal ranks share the same cases and full-grid
noise draws.  Decision thresholds are calibrated on separate cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BoundaryThresholds:
    state_relative_l2: float = 0.05
    dynamics_relative_l2: float = 0.15
    posterior_total_variation: float = 0.10
    action_disagreement: float = 0.05
    false_safe_rate: float = 0.05


@dataclass(frozen=True)
class BoundaryConfig:
    grid_points: int = 64
    pod_modes: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    calibration_cases: int = 8
    evaluation_cases: int = 10
    noise_draws: int = 30
    seed: int = 20261005
    burgers_dt: float = 0.002
    burgers_observation_times: tuple[float, ...] = (0.06, 0.12)
    burgers_decision_time: float = 0.24
    burgers_viscosities: tuple[float, ...] = (0.018, 0.024, 0.030, 0.036, 0.044, 0.052, 0.062)
    burgers_noise_sigma: float = 0.025
    wave_observation_times: tuple[float, ...] = (0.16, 0.30)
    wave_decision_time: float = 0.80
    wave_dampings: tuple[float, ...] = (0.04, 0.08, 0.12, 0.17, 0.23, 0.30, 0.38)
    wave_noise_sigma: float = 0.020
    wave_spatial_modes: int = 20


def _relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    denominator = np.linalg.norm(reference.ravel())
    return float(np.linalg.norm((reference - estimate).ravel()) / denominator)


def _pod_fit(snapshots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = snapshots.mean(axis=0)
    basis = np.linalg.svd(snapshots - mean, full_matrices=False)[2]
    return mean, basis


def _pod_coordinates(fields: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (fields - mean) @ basis.T


def _pod_reconstruct(
    fields: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    return _pod_coordinates(fields, mean, basis) @ basis + mean


def _posterior(observation: np.ndarray, predictions: np.ndarray, sigma: float) -> np.ndarray:
    residual = predictions - observation[None, :]
    log_weights = -0.5 * np.square(residual / sigma).sum(axis=1)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    return weights / weights.sum()


def _burgers_rhs(u: np.ndarray, viscosity: float, wave_numbers: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(u)
    gradient = np.fft.ifft(1j * wave_numbers * spectrum).real
    laplacian = np.fft.ifft(-(wave_numbers**2) * spectrum).real
    return -u * gradient + viscosity * laplacian


def _burgers_trajectory(
    initial: np.ndarray,
    viscosity: float,
    times: tuple[float, ...],
    dt: float,
) -> np.ndarray:
    """Integrate once and return fields at prespecified times."""
    points = initial.size
    wave_numbers = np.fft.fftfreq(points, d=1 / points)
    requested = np.asarray(times, dtype=float)
    step_indices = np.rint(requested / dt).astype(int)
    if not np.allclose(step_indices * dt, requested):
        raise ValueError("Burgers output times must be integer multiples of dt")
    output = np.empty((len(times), points))
    u = initial.copy()
    lookup = {step: index for index, step in enumerate(step_indices)}
    if 0 in lookup:
        output[lookup[0]] = u
    for step in range(1, int(step_indices.max()) + 1):
        a = _burgers_rhs(u, viscosity, wave_numbers)
        b = _burgers_rhs(u + 0.5 * dt * a, viscosity, wave_numbers)
        c = _burgers_rhs(u + 0.5 * dt * b, viscosity, wave_numbers)
        d = _burgers_rhs(u + dt * c, viscosity, wave_numbers)
        u += dt * (a + 2 * b + 2 * c + d) / 6
        if step in lookup:
            output[lookup[step]] = u
    return output


def _burgers_initial(rng: np.random.Generator, points: int) -> np.ndarray:
    x = np.linspace(0, 2 * np.pi, points, endpoint=False)
    coefficients = rng.normal(0, (0.85, 0.42, 0.25, 0.16, 0.10, 0.07))
    phases = rng.uniform(0, 2 * np.pi, len(coefficients))
    return sum(
        coefficient * np.sin((index + 1) * x + phase)
        for index, (coefficient, phase) in enumerate(zip(coefficients, phases))
    )


def _burgers_qoi(field: np.ndarray) -> float:
    gradient = 0.5 * (np.roll(field, -1) - np.roll(field, 1))
    return float(-gradient.min())


def _wave_initial(
    rng: np.random.Generator,
    points: int,
    spatial_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, points + 2)[1:-1]
    indices = np.arange(1, spatial_modes + 1)
    amplitudes = rng.normal(0, 0.30 / np.sqrt(indices))
    velocity_amplitudes = rng.normal(0, 0.16 / np.sqrt(indices))
    sines = np.sin(np.pi * indices[:, None] * x[None, :])
    return amplitudes @ sines, velocity_amplitudes @ sines


def _wave_trajectory(
    initial: np.ndarray,
    velocity: np.ndarray,
    damping: float,
    times: tuple[float, ...],
    spatial_modes: int,
) -> np.ndarray:
    points = initial.size
    x = np.linspace(0, 1, points + 2)[1:-1]
    indices = np.arange(1, spatial_modes + 1)
    sines = np.sin(np.pi * indices[:, None] * x[None, :])
    normalization = 2.0 / (points + 1)
    initial_coefficients = normalization * (sines @ initial)
    velocity_coefficients = normalization * (sines @ velocity)
    omega = np.sqrt(np.maximum((np.pi * indices) ** 2 - damping**2, 1e-12))
    output = []
    for time in times:
        coefficients = np.exp(-damping * time) * (
            initial_coefficients * np.cos(omega * time)
            + (velocity_coefficients + damping * initial_coefficients)
            / omega
            * np.sin(omega * time)
        )
        output.append(coefficients @ sines)
    return np.asarray(output)


def _wave_qoi(field: np.ndarray) -> float:
    index = round(0.75 * (field.size + 1)) - 1
    return float(abs(field[index]))


def _case_parameters(values: tuple[float, ...], count: int) -> np.ndarray:
    indices = np.arange(count) % len(values)
    return np.asarray(values)[indices]


def _prepare_burgers(config: BoundaryConfig, rng: np.random.Generator) -> dict[str, object]:
    total = config.calibration_cases + config.evaluation_cases
    initials = [_burgers_initial(rng, config.grid_points) for _ in range(total)]
    parameters = _case_parameters(config.burgers_viscosities, total)
    all_times = (0.0,) + config.burgers_observation_times + (config.burgers_decision_time,)
    libraries = []
    for initial in initials:
        libraries.append(
            np.asarray(
                [
                    _burgers_trajectory(
                        initial,
                        viscosity,
                        all_times,
                        config.burgers_dt,
                    )
                    for viscosity in config.burgers_viscosities
                ]
            )
        )
    libraries_array = np.asarray(libraries)
    true_indices = np.asarray(
        [config.burgers_viscosities.index(float(value)) for value in parameters]
    )
    true_trajectories = libraries_array[np.arange(total), true_indices]
    calibration_qoi = np.asarray(
        [_burgers_qoi(field) for field in true_trajectories[: config.calibration_cases, -1]]
    )
    threshold = float(np.median(calibration_qoi))
    training_snapshots = libraries_array[: config.calibration_cases].reshape(
        -1, config.grid_points
    )
    mean, basis = _pod_fit(training_snapshots)
    return {
        "name": "burgers",
        "initials": np.asarray(initials),
        "velocities": None,
        "parameters": parameters,
        "libraries": libraries_array,
        "truth": true_trajectories,
        "mean": mean,
        "basis": basis,
        "hazard_threshold": threshold,
        "qoi": _burgers_qoi,
        "noise_sigma": config.burgers_noise_sigma,
        "observation_slice": slice(1, 1 + len(config.burgers_observation_times)),
    }


def _prepare_wave(config: BoundaryConfig, rng: np.random.Generator) -> dict[str, object]:
    total = config.calibration_cases + config.evaluation_cases
    pairs = [
        _wave_initial(rng, config.grid_points, config.wave_spatial_modes)
        for _ in range(total)
    ]
    initials = np.asarray([pair[0] for pair in pairs])
    velocities = np.asarray([pair[1] for pair in pairs])
    parameters = _case_parameters(config.wave_dampings, total)
    all_times = (0.0,) + config.wave_observation_times + (config.wave_decision_time,)
    libraries = []
    for initial, velocity in zip(initials, velocities):
        libraries.append(
            np.asarray(
                [
                    _wave_trajectory(
                        initial,
                        velocity,
                        damping,
                        all_times,
                        config.wave_spatial_modes,
                    )
                    for damping in config.wave_dampings
                ]
            )
        )
    libraries_array = np.asarray(libraries)
    true_indices = np.asarray(
        [config.wave_dampings.index(float(value)) for value in parameters]
    )
    true_trajectories = libraries_array[np.arange(total), true_indices]
    calibration_qoi = np.asarray(
        [_wave_qoi(field) for field in true_trajectories[: config.calibration_cases, -1]]
    )
    threshold = float(np.median(calibration_qoi))
    training_snapshots = np.concatenate(
        (
            libraries_array[: config.calibration_cases].reshape(-1, config.grid_points),
            velocities[: config.calibration_cases],
        ),
        axis=0,
    )
    mean, basis = _pod_fit(training_snapshots)
    return {
        "name": "damped_wave",
        "initials": initials,
        "velocities": velocities,
        "parameters": parameters,
        "libraries": libraries_array,
        "truth": true_trajectories,
        "mean": mean,
        "basis": basis,
        "hazard_threshold": threshold,
        "qoi": _wave_qoi,
        "noise_sigma": config.wave_noise_sigma,
        "observation_slice": slice(1, 1 + len(config.wave_observation_times)),
    }


def _compressed_dynamics(
    system: dict[str, object],
    case_index: int,
    modes: int,
    config: BoundaryConfig,
) -> np.ndarray:
    mean = system["mean"]
    basis = system["basis"][:modes]
    initial = system["initials"][case_index]
    reconstructed_initial = _pod_reconstruct(initial[None, :], mean, basis)[0]
    parameter = float(system["parameters"][case_index])
    if system["name"] == "burgers":
        return _burgers_trajectory(
            reconstructed_initial,
            parameter,
            (config.burgers_decision_time,),
            config.burgers_dt,
        )[0]
    velocity = system["velocities"][case_index]
    reconstructed_velocity = _pod_reconstruct(velocity[None, :], mean, basis)[0]
    return _wave_trajectory(
        reconstructed_initial,
        reconstructed_velocity,
        parameter,
        (config.wave_decision_time,),
        config.wave_spatial_modes,
    )[0]


def _evaluate_system(
    system: dict[str, object],
    config: BoundaryConfig,
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    aggregate: list[dict[str, object]] = []
    noise_records: list[dict[str, object]] = []
    start = config.calibration_cases
    observation_slice = system["observation_slice"]
    mean = system["mean"]
    qoi = system["qoi"]
    threshold = float(system["hazard_threshold"])
    sigma = float(system["noise_sigma"])
    for local_case, case_index in enumerate(range(start, start + config.evaluation_cases)):
        truth = system["truth"][case_index]
        observation_truth = truth[observation_slice]
        future_truth = truth[-1]
        candidate_observations = system["libraries"][case_index, :, observation_slice, :]
        candidate_future = system["libraries"][case_index, :, -1, :]
        candidate_hazards = np.asarray([qoi(field) > threshold for field in candidate_future])
        true_hazard = bool(qoi(future_truth) > threshold)
        full_noise = rng.normal(
            0,
            sigma,
            size=(config.noise_draws,) + observation_truth.shape,
        )
        grid_predictions = candidate_observations.reshape(len(candidate_observations), -1)
        grid_posteriors = np.asarray(
            [
                _posterior(
                    (observation_truth + noise).reshape(-1),
                    grid_predictions,
                    sigma,
                )
                for noise in full_noise
            ]
        )
        grid_actions = (grid_posteriors @ candidate_hazards.astype(float)) > 0.5
        for modes in config.pod_modes:
            basis = system["basis"][:modes]
            reconstructed_observation = _pod_reconstruct(
                observation_truth,
                mean,
                basis,
            )
            state_error = _relative_l2(observation_truth, reconstructed_observation)
            predicted_future = _compressed_dynamics(system, case_index, modes, config)
            dynamics_error = _relative_l2(future_truth, predicted_future)
            candidate_coordinates = np.asarray(
                [
                    _pod_coordinates(fields, mean, basis).reshape(-1)
                    for fields in candidate_observations
                ]
            )
            posterior_tvs = []
            actions = []
            for draw, noise in enumerate(full_noise):
                noisy_coordinates = _pod_coordinates(
                    observation_truth + noise,
                    mean,
                    basis,
                ).reshape(-1)
                posterior = _posterior(noisy_coordinates, candidate_coordinates, sigma)
                posterior_tv = 0.5 * np.abs(posterior - grid_posteriors[draw]).sum()
                action = bool((posterior @ candidate_hazards.astype(float)) > 0.5)
                posterior_tvs.append(posterior_tv)
                actions.append(action)
                noise_records.append(
                    {
                        "pde": system["name"],
                        "case": local_case,
                        "pod_modes": modes,
                        "noise_draw": draw,
                        "posterior_total_variation": posterior_tv,
                        "grid_action": bool(grid_actions[draw]),
                        "representation_action": action,
                        "true_hazard": true_hazard,
                    }
                )
            actions_array = np.asarray(actions, dtype=bool)
            aggregate.append(
                {
                    "pde": system["name"],
                    "case": local_case,
                    "true_parameter": float(system["parameters"][case_index]),
                    "pod_modes": modes,
                    "state_relative_l2": state_error,
                    "dynamics_relative_l2": dynamics_error,
                    "posterior_total_variation": float(np.mean(posterior_tvs)),
                    "action_disagreement": float(np.mean(actions_array != grid_actions)),
                    "false_safe_rate": float(
                        np.mean((~actions_array) & grid_actions)
                    ),
                    "false_alarm_rate": float(
                        np.mean(actions_array & (~grid_actions))
                    ),
                    "truth_false_safe_rate": float(
                        np.mean((~actions_array) & true_hazard)
                    ),
                    "truth_false_alarm_rate": float(
                        np.mean(actions_array & (not true_hazard))
                    ),
                    "true_hazard": true_hazard,
                    "hazard_threshold": threshold,
                }
            )
    return aggregate, noise_records


def run_boundary_audit(
    config: BoundaryConfig = BoundaryConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired representation sweeps for both PDE families."""
    if max(config.pod_modes) > config.grid_points:
        raise ValueError("POD modes cannot exceed grid_points")
    rng = np.random.default_rng(config.seed)
    systems = (_prepare_burgers(config, rng), _prepare_wave(config, rng))
    aggregate: list[dict[str, object]] = []
    noise_records: list[dict[str, object]] = []
    for system in systems:
        system_aggregate, system_noise = _evaluate_system(system, config, rng)
        aggregate.extend(system_aggregate)
        noise_records.extend(system_noise)
    return pd.DataFrame(aggregate), pd.DataFrame(noise_records)


def _criterion_passes(
    rows: pd.DataFrame,
    preservation: str,
    thresholds: BoundaryThresholds,
) -> np.ndarray:
    if preservation == "state":
        return rows["state_relative_l2"].to_numpy() <= thresholds.state_relative_l2
    if preservation == "dynamics":
        return rows["dynamics_relative_l2"].to_numpy() <= thresholds.dynamics_relative_l2
    if preservation == "inference":
        return (
            rows["posterior_total_variation"].to_numpy()
            <= thresholds.posterior_total_variation
        )
    if preservation == "decision":
        return (
            (rows["action_disagreement"].to_numpy() <= thresholds.action_disagreement)
            & (rows["false_safe_rate"].to_numpy() <= thresholds.false_safe_rate)
        )
    raise ValueError(f"Unknown preservation criterion: {preservation}")


def extract_boundaries(
    results: pd.DataFrame,
    thresholds: BoundaryThresholds = BoundaryThresholds(),
) -> pd.DataFrame:
    """Extract first and stable passing ranks without assuming monotonicity."""
    records: list[dict[str, object]] = []
    for (pde, case), group in results.groupby(["pde", "case"], sort=True):
        ordered = group.sort_values("pod_modes")
        ranks = ordered["pod_modes"].to_numpy(dtype=int)
        for preservation in ("state", "dynamics", "inference", "decision"):
            passes = _criterion_passes(ordered, preservation, thresholds)
            first_index = int(np.flatnonzero(passes)[0]) if passes.any() else None
            stable_candidates = [
                index for index in range(len(passes)) if bool(np.all(passes[index:]))
            ]
            stable_index = stable_candidates[0] if stable_candidates else None
            violations = int(np.sum(passes[:-1] & ~passes[1:]))
            records.append(
                {
                    "pde": pde,
                    "case": case,
                    "preservation": preservation,
                    "first_passing_rank": (
                        int(ranks[first_index]) if first_index is not None else np.nan
                    ),
                    "stable_boundary_rank": (
                        int(ranks[stable_index]) if stable_index is not None else np.nan
                    ),
                    "boundary_width_steps": (
                        int(stable_index - first_index)
                        if first_index is not None and stable_index is not None
                        else np.nan
                    ),
                    "order_violations": violations,
                    "resolved": stable_index is not None,
                }
            )
    return pd.DataFrame(records)


def summarize_boundaries(boundaries: pd.DataFrame) -> pd.DataFrame:
    """Summarize the distribution of case-level stable boundaries."""
    return (
        boundaries.groupby(["pde", "preservation"], as_index=False)
        .agg(
            cases=("case", "nunique"),
            resolved_fraction=("resolved", "mean"),
            median_stable_rank=("stable_boundary_rank", "median"),
            maximum_stable_rank=("stable_boundary_rank", "max"),
            cases_with_order_violations=("order_violations", lambda values: int((values > 0).sum())),
        )
    )


def boundary_specification(
    config: BoundaryConfig = BoundaryConfig(),
    thresholds: BoundaryThresholds = BoundaryThresholds(),
) -> pd.DataFrame:
    """Return the prespecified rules needed to interpret every boundary."""
    return pd.DataFrame(
        [
            ("state", "mean observation-field relative L2", thresholds.state_relative_l2),
            ("dynamics", "future-field relative L2 after compressed initial state", thresholds.dynamics_relative_l2),
            ("inference", "mean posterior total variation from full-grid oracle", thresholds.posterior_total_variation),
            ("decision", "action disagreement and false-safe rate", thresholds.action_disagreement),
            ("decision_false_safe", "false-safe rate", thresholds.false_safe_rate),
            ("action_probability", "intervene when posterior hazard probability exceeds", 0.5),
            ("hazard_calibration", "median future QoI on separate calibration cases", config.calibration_cases),
        ],
        columns=("criterion", "definition", "threshold"),
    )


def plot_boundaries(boundaries: pd.DataFrame, path: Path) -> None:
    """Plot case-level stable ranks for both PDE families."""
    preservations = ("state", "dynamics", "inference", "decision")
    pdes = ("burgers", "damped_wave")
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), sharex=True)
    finite = boundaries["stable_boundary_rank"].dropna()
    maximum = int(finite.max()) if len(finite) else 64
    for ax, pde in zip(axes, pdes):
        subset = boundaries[boundaries["pde"].eq(pde)]
        matrix = subset.pivot(
            index="preservation",
            columns="case",
            values="stable_boundary_rank",
        ).reindex(preservations)
        display = matrix.fillna(maximum * 1.25).to_numpy(dtype=float)
        image = ax.imshow(display, aspect="auto", cmap="magma_r", vmin=2, vmax=maximum * 1.25)
        ax.set_yticks(range(len(preservations)), [value.title() for value in preservations])
        ax.set_title(pde.replace("_", " ").title(), loc="left", weight="bold")
        for row in range(display.shape[0]):
            for column in range(display.shape[1]):
                value = matrix.iloc[row, column]
                label = "NR" if pd.isna(value) else str(int(value))
                ax.text(column, row, label, ha="center", va="center", fontsize=8)
    axes[-1].set_xlabel("Evaluation case")
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Stable POD boundary rank")
    fig.suptitle(
        "One physical case can have four representation boundaries",
        x=0.08,
        ha="left",
        weight="bold",
    )
    fig.subplots_adjust(left=0.16, right=0.88, top=0.90, hspace=0.30)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_boundary_outputs(
    output: Path,
    config: BoundaryConfig = BoundaryConfig(),
    thresholds: BoundaryThresholds = BoundaryThresholds(),
) -> None:
    """Run the audit and write the complete, paired evidence trail."""
    output.mkdir(parents=True, exist_ok=True)
    results, noise_draws = run_boundary_audit(config)
    boundaries = extract_boundaries(results, thresholds)
    results.to_csv(output / "representation_boundary_runs.csv", index=False)
    noise_draws.to_csv(output / "representation_boundary_noise_draws.csv", index=False)
    boundaries.to_csv(output / "representation_boundaries.csv", index=False)
    summarize_boundaries(boundaries).to_csv(
        output / "representation_boundary_summary.csv",
        index=False,
    )
    boundary_specification(config, thresholds).to_csv(
        output / "representation_boundary_specification.csv",
        index=False,
    )
    plot_boundaries(boundaries, output / "representation_boundaries.png")
