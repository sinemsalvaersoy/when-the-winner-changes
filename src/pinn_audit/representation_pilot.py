"""Controlled representation experiment for the viscous Burgers equation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BurgersConfig:
    grid_points: int = 64
    train_samples: int = 140
    test_samples: int = 80
    viscosity: float = 0.03
    time_horizon: float = 0.30
    time_step: float = 0.001
    pod_modes: int = 4
    neighbours: int = 3
    ridge_penalty: float = 1e-5
    seeds: tuple[int, ...] = (11, 17, 23)


@dataclass(frozen=True)
class RobustnessConfig:
    """Axes for testing whether the representation result survives perturbations."""

    grid_points: tuple[int, ...] = (64, 128, 256)
    pod_modes: tuple[int, ...] = (2, 4, 8, 16)
    seeds: tuple[int, ...] = (11, 17, 23, 29, 31, 37, 41, 43, 47, 53)
    gradient_weight: float = 1.0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 20260919


def _rhs(u: np.ndarray, viscosity: float, wave_numbers: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(u)
    gradient = np.fft.ifft(1j * wave_numbers * spectrum).real
    laplacian = np.fft.ifft(-(wave_numbers**2) * spectrum).real
    return -u * gradient + viscosity * laplacian


def solve_burgers(initial: np.ndarray, config: BurgersConfig) -> np.ndarray:
    """Integrate periodic viscous Burgers dynamics with fourth order Runge Kutta."""
    u = initial.copy()
    k = np.fft.fftfreq(config.grid_points, d=1 / config.grid_points)
    steps = round(config.time_horizon / config.time_step)
    for _ in range(steps):
        a = _rhs(u, config.viscosity, k)
        b = _rhs(u + 0.5 * config.time_step * a, config.viscosity, k)
        c = _rhs(u + 0.5 * config.time_step * b, config.viscosity, k)
        d = _rhs(u + config.time_step * c, config.viscosity, k)
        u += config.time_step * (a + 2 * b + 2 * c + d) / 6
    return u


def generate_dataset(seed: int, config: BurgersConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate smooth initial states and their evolved Burgers solutions."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, config.grid_points, endpoint=False)
    total = config.train_samples + config.test_samples
    inputs = np.empty((total, config.grid_points))
    targets = np.empty_like(inputs)
    for index in range(total):
        amplitude = rng.uniform(0.6, 1.4)
        phase = rng.uniform(0, 2 * np.pi)
        harmonic = rng.uniform(-0.35, 0.35)
        state = amplitude * np.sin(x + phase) + harmonic * np.sin(2 * x - 0.3 * phase)
        inputs[index] = state
        targets[index] = solve_burgers(state, config)
    return inputs, targets


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, penalty: float) -> np.ndarray:
    design = np.column_stack((np.ones(len(train_x)), train_x))
    test_design = np.column_stack((np.ones(len(test_x)), test_x))
    regularizer = penalty * np.eye(design.shape[1])
    regularizer[0, 0] = 0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ train_y)
    return test_design @ weights


def _knn_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, neighbours: int) -> np.ndarray:
    distances = ((test_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
    indices = np.argpartition(distances, neighbours, axis=1)[:, :neighbours]
    return train_y[indices].mean(axis=1)


def _scores(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    relative_l2 = np.mean(
        np.linalg.norm(target - prediction, axis=1) / np.linalg.norm(target, axis=1)
    )
    target_gradient = np.roll(target, -1, axis=1) - np.roll(target, 1, axis=1)
    prediction_gradient = np.roll(prediction, -1, axis=1) - np.roll(prediction, 1, axis=1)
    target_location = np.argmin(target_gradient, axis=1)
    prediction_location = np.argmin(prediction_gradient, axis=1)
    separation = np.abs(target_location - prediction_location)
    circular_separation = np.minimum(separation, target.shape[1] - separation)
    location_error = np.mean(circular_separation * 2 * np.pi / target.shape[1])
    return float(relative_l2), float(location_error)


def _gradient(values: np.ndarray) -> np.ndarray:
    """Return the periodic central difference used by the local observable."""
    return 0.5 * (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1))


def _gradient_aware_basis(
    centered_snapshots: np.ndarray,
    modes: int,
    gradient_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a POD basis in an L2 plus periodic-gradient inner product.

    The returned coordinates and decoder minimize snapshot error after whitening
    by the chosen inner product. This makes the intervention explicit instead of
    changing the learner or its training data.
    """
    points = centered_snapshots.shape[1]
    difference = np.zeros((points, points))
    indices = np.arange(points)
    difference[indices, (indices + 1) % points] = 0.5
    difference[indices, (indices - 1) % points] = -0.5
    metric = np.eye(points) + gradient_weight * difference @ difference.T
    cholesky = np.linalg.cholesky(metric)
    transformed = centered_snapshots @ cholesky
    right_vectors = np.linalg.svd(transformed, full_matrices=False)[2][:modes]
    coordinates = transformed @ right_vectors.T
    decoder = np.linalg.solve(cholesky.T, right_vectors.T).T
    return coordinates, decoder


def _retained_fraction(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    denominator = np.square(reference).sum()
    if denominator == 0:
        return 1.0
    return float(1.0 - np.square(reference - reconstruction).sum() / denominator)


def run_representation_pilot(config: BurgersConfig = BurgersConfig()) -> pd.DataFrame:
    """Compare identical learners under grid and POD encodings across paired seeds."""
    records: list[dict[str, object]] = []
    for seed in config.seeds:
        inputs, targets = generate_dataset(seed, config)
        split = config.train_samples
        train_x, test_x = inputs[:split], inputs[split:]
        train_y, test_y = targets[:split], targets[split:]

        input_mean = train_x.mean(axis=0)
        output_mean = train_y.mean(axis=0)
        input_basis = np.linalg.svd(train_x - input_mean, full_matrices=False)[2][: config.pod_modes]
        output_basis = np.linalg.svd(train_y - output_mean, full_matrices=False)[2][: config.pod_modes]
        input_singular_values = np.linalg.svd(train_x - input_mean, compute_uv=False)
        output_singular_values = np.linalg.svd(train_y - output_mean, compute_uv=False)
        input_energy = float(
            (input_singular_values[: config.pod_modes] ** 2).sum()
            / (input_singular_values**2).sum()
        )
        output_energy = float(
            (output_singular_values[: config.pod_modes] ** 2).sum()
            / (output_singular_values**2).sum()
        )

        settings = {
            "grid": (train_x, test_x, train_y, lambda values: values),
            "pod": (
                (train_x - input_mean) @ input_basis.T,
                (test_x - input_mean) @ input_basis.T,
                (train_y - output_mean) @ output_basis.T,
                lambda values: values @ output_basis + output_mean,
            ),
        }
        for representation, (encoded_train_x, encoded_test_x, encoded_train_y, decode) in settings.items():
            predictions = {
                "ridge": _ridge_predict(
                    encoded_train_x, encoded_train_y, encoded_test_x, config.ridge_penalty
                ),
                "knn": _knn_predict(
                    encoded_train_x, encoded_train_y, encoded_test_x, config.neighbours
                ),
            }
            for model, encoded_prediction in predictions.items():
                relative_l2, location_error = _scores(test_y, decode(encoded_prediction))
                records.append(
                    {
                        "seed": seed,
                        "representation": representation,
                        "model": model,
                        "relative_l2": relative_l2,
                        "steepest_gradient_location_error_radians": location_error,
                        "pod_input_energy_retained": input_energy,
                        "pod_output_energy_retained": output_energy,
                    }
                )
    return pd.DataFrame.from_records(records)


def summarize_pilot(results: pd.DataFrame) -> pd.DataFrame:
    """Return mean scores and the winner under each representation and metric."""
    summary = (
        results.groupby(["representation", "model"], as_index=False)
        .agg(
            relative_l2=("relative_l2", "mean"),
            relative_l2_std=("relative_l2", "std"),
            steepest_gradient_location_error_radians=(
                "steepest_gradient_location_error_radians",
                "mean",
            ),
            steepest_gradient_location_error_std=(
                "steepest_gradient_location_error_radians",
                "std",
            ),
        )
    )
    for metric in ("relative_l2", "steepest_gradient_location_error_radians"):
        summary[f"winner_{metric}"] = False
        winners = summary.groupby("representation")[metric].idxmin()
        summary.loc[winners, f"winner_{metric}"] = True
    return summary


def conditions_table(config: BurgersConfig = BurgersConfig()) -> pd.DataFrame:
    """Record controlled and changed factors for the representation intervention."""
    values = asdict(config)
    rows = [
        ("Physical equation", "Fixed", "Periodic viscous Burgers equation"),
        ("Viscosity", "Fixed", str(values["viscosity"])),
        ("Time horizon", "Fixed", str(values["time_horizon"])),
        ("Spatial resolution", "Fixed", str(values["grid_points"])),
        ("Train and test samples", "Fixed", f'{values["train_samples"]} and {values["test_samples"]}'),
        ("Data seeds", "Paired", ", ".join(map(str, values["seeds"]))),
        ("Learners", "Fixed", "Linear ridge and 3 nearest neighbours"),
        ("Input and output encoding", "Changed", f'Full grid or {values["pod_modes"]} mode POD'),
        ("Evaluation", "Fixed", "Relative L2 and steepest gradient location error"),
    ]
    return pd.DataFrame(rows, columns=["factor", "status", "value"])


def plot_pilot(summary: pd.DataFrame, path: Path) -> None:
    """Plot paired model comparisons and expose representation dependent winners."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    metrics = [
        ("relative_l2", "Relative L2 error"),
        (
            "steepest_gradient_location_error_radians",
            "Steepest gradient location error · radians",
        ),
    ]
    colors = {"ridge": "#20a486", "knn": "#d95f59"}
    for ax, (metric, label) in zip(axes, metrics):
        for model in ("ridge", "knn"):
            subset = summary[summary["model"].eq(model)].set_index("representation")
            ax.plot(
                ["Grid", "POD"],
                [subset.loc["grid", metric], subset.loc["pod", metric]],
                marker="o",
                linewidth=2.4,
                markersize=7,
                color=colors[model],
                label=model.upper() if model == "knn" else "Ridge",
            )
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Field error", loc="left", weight="bold")
    axes[1].set_title("Physical observable", loc="left", weight="bold")
    axes[0].legend(frameon=False)
    fig.suptitle("The winner depends on what the representation preserves", x=0.07, ha="left", weight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_pilot_outputs(output: Path, config: BurgersConfig = BurgersConfig()) -> None:
    """Run the experiment and write its auditable data products."""
    output.mkdir(parents=True, exist_ok=True)
    raw = run_representation_pilot(config)
    summary = summarize_pilot(raw)
    raw.to_csv(output / "burgers_representation_runs.csv", index=False)
    summary.to_csv(output / "burgers_representation_summary.csv", index=False)
    conditions_table(config).to_csv(output / "burgers_conditions.csv", index=False)
    plot_pilot(summary, output / "burgers_representation_reversal.png")


def run_robustness_experiment(
    robustness: RobustnessConfig = RobustnessConfig(),
    base: BurgersConfig = BurgersConfig(),
) -> pd.DataFrame:
    """Run a paired seed, resolution, modal-budget, and basis-objective sweep."""
    records: list[dict[str, object]] = []
    for grid_points in robustness.grid_points:
        config = replace(base, grid_points=grid_points)
        for seed in robustness.seeds:
            inputs, targets = generate_dataset(seed, config)
            split = config.train_samples
            train_x, test_x = inputs[:split], inputs[split:]
            train_y, test_y = targets[:split], targets[split:]
            input_mean = train_x.mean(axis=0)
            output_mean = train_y.mean(axis=0)
            centered_x = train_x - input_mean
            centered_y = train_y - output_mean
            input_right = np.linalg.svd(centered_x, full_matrices=False)[2]
            output_right = np.linalg.svd(centered_y, full_matrices=False)[2]

            grid_predictions = {
                "ridge": _ridge_predict(train_x, train_y, test_x, config.ridge_penalty),
                "knn": _knn_predict(train_x, train_y, test_x, config.neighbours),
            }

            for modes in robustness.pod_modes:
                if modes > min(centered_x.shape):
                    raise ValueError(
                        f"POD modes ({modes}) exceed the available rank for grid {grid_points}"
                    )
                input_basis = input_right[:modes]
                encoded_train_x = centered_x @ input_basis.T
                encoded_test_x = (test_x - input_mean) @ input_basis.T

                l2_basis = output_right[:modes]
                l2_train_y = centered_y @ l2_basis.T
                gradient_train_y, gradient_decoder = _gradient_aware_basis(
                    centered_y,
                    modes,
                    robustness.gradient_weight,
                )

                representations = {
                    "grid": (
                        grid_predictions,
                        1.0,
                        1.0,
                    ),
                    "pod_l2": (
                        {
                            "ridge": _ridge_predict(
                                encoded_train_x,
                                l2_train_y,
                                encoded_test_x,
                                config.ridge_penalty,
                            )
                            @ l2_basis
                            + output_mean,
                            "knn": _knn_predict(
                                encoded_train_x,
                                l2_train_y,
                                encoded_test_x,
                                config.neighbours,
                            )
                            @ l2_basis
                            + output_mean,
                        },
                        _retained_fraction(centered_y, l2_train_y @ l2_basis),
                        _retained_fraction(_gradient(centered_y), _gradient(l2_train_y @ l2_basis)),
                    ),
                    "pod_gradient": (
                        {
                            "ridge": _ridge_predict(
                                encoded_train_x,
                                gradient_train_y,
                                encoded_test_x,
                                config.ridge_penalty,
                            )
                            @ gradient_decoder
                            + output_mean,
                            "knn": _knn_predict(
                                encoded_train_x,
                                gradient_train_y,
                                encoded_test_x,
                                config.neighbours,
                            )
                            @ gradient_decoder
                            + output_mean,
                        },
                        _retained_fraction(
                            centered_y,
                            gradient_train_y @ gradient_decoder,
                        ),
                        _retained_fraction(
                            _gradient(centered_y),
                            _gradient(gradient_train_y @ gradient_decoder),
                        ),
                    ),
                }

                for representation, (
                    predictions,
                    l2_retained,
                    gradient_retained,
                ) in representations.items():
                    for model, prediction in predictions.items():
                        relative_l2, location_error = _scores(test_y, prediction)
                        records.append(
                            {
                                "seed": seed,
                                "grid_points": grid_points,
                                "pod_modes": modes,
                                "representation": representation,
                                "model": model,
                                "relative_l2": relative_l2,
                                "steepest_gradient_location_error_radians": location_error,
                                "output_l2_fraction_retained": l2_retained,
                                "output_gradient_fraction_retained": gradient_retained,
                            }
                        )
    return pd.DataFrame.from_records(records)


def summarize_robustness(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate paired scores without treating modal budgets as extra grid trials."""
    return (
        results.groupby(
            ["grid_points", "pod_modes", "representation", "model"],
            as_index=False,
        )
        .agg(
            trials=("seed", "nunique"),
            relative_l2=("relative_l2", "mean"),
            relative_l2_std=("relative_l2", "std"),
            steepest_gradient_location_error_radians=(
                "steepest_gradient_location_error_radians",
                "mean",
            ),
            steepest_gradient_location_error_std=(
                "steepest_gradient_location_error_radians",
                "std",
            ),
            output_l2_fraction_retained=("output_l2_fraction_retained", "mean"),
            output_gradient_fraction_retained=(
                "output_gradient_fraction_retained",
                "mean",
            ),
        )
    )


def winner_probabilities(
    results: pd.DataFrame,
    robustness: RobustnessConfig = RobustnessConfig(),
) -> pd.DataFrame:
    """Estimate model win probabilities and paired-seed bootstrap intervals."""
    metrics = ("relative_l2", "steepest_gradient_location_error_radians")
    block = ["seed", "grid_points", "pod_modes", "representation"]
    winners: list[pd.DataFrame] = []
    for metric in metrics:
        indices = results.groupby(block, sort=True)[metric].idxmin()
        selected = results.loc[indices, block + ["model"]].copy()
        selected["metric"] = metric
        winners.append(selected)
    winner_rows = pd.concat(winners, ignore_index=True)

    rng = np.random.default_rng(robustness.bootstrap_seed)
    records: list[dict[str, object]] = []
    group_columns = ["grid_points", "pod_modes", "representation", "metric"]
    for keys, group in winner_rows.groupby(group_columns, sort=True):
        labels = group.sort_values("seed")["model"].to_numpy()
        for model in sorted(results["model"].unique()):
            indicators = labels == model
            trials = len(indicators)
            probability = float(indicators.mean())
            z_value = 1.959963984540054
            denominator = 1 + z_value**2 / trials
            centre = (probability + z_value**2 / (2 * trials)) / denominator
            half_width = (
                z_value
                * np.sqrt(
                    probability * (1 - probability) / trials
                    + z_value**2 / (4 * trials**2)
                )
                / denominator
            )
            draws = rng.choice(
                indicators,
                size=(robustness.bootstrap_draws, trials),
                replace=True,
            ).mean(axis=1)
            records.append(
                {
                    **dict(zip(group_columns, keys)),
                    "model": model,
                    "wins": int(indicators.sum()),
                    "trials": trials,
                    "winner_probability": probability,
                    "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                    "wilson_ci_low": float(centre - half_width),
                    "wilson_ci_high": float(centre + half_width),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_robustness(summary: pd.DataFrame, path: Path) -> None:
    """Plot retained information beside the local-observable error."""
    pod = summary[summary["representation"].ne("grid")]
    pod = (
        pod.groupby(
            ["grid_points", "pod_modes", "representation"],
            as_index=False,
        )
        .agg(
            output_l2_fraction_retained=("output_l2_fraction_retained", "mean"),
            location_error=("steepest_gradient_location_error_radians", "min"),
        )
        .groupby(["pod_modes", "representation"], as_index=False)
        .agg(
            output_l2_fraction_retained=("output_l2_fraction_retained", "mean"),
            location_error=("location_error", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = {"pod_l2": "L2 POD", "pod_gradient": "Gradient-aware POD"}
    colors = {"pod_l2": "#3b6ea8", "pod_gradient": "#d46a3a"}
    styles = {
        "pod_l2": {"linestyle": "--", "marker": "o", "markerfacecolor": "white", "zorder": 3},
        "pod_gradient": {"linestyle": "-", "marker": "x", "zorder": 2},
    }
    for representation in ("pod_gradient", "pod_l2"):
        subset = pod[pod["representation"].eq(representation)]
        axes[0].plot(
            subset["pod_modes"],
            100 * subset["output_l2_fraction_retained"],
            color=colors[representation],
            label=labels[representation],
            **styles[representation],
        )
        axes[1].plot(
            subset["pod_modes"],
            subset["location_error"],
            color=colors[representation],
            label=labels[representation],
            **styles[representation],
        )
    axes[0].set_ylabel("Output L2 variance retained · %")
    axes[1].set_ylabel("Best location error · radians")
    for ax in axes:
        ax.set_xlabel("Modal budget")
        ax.set_xticks(sorted(pod["pod_modes"].unique()))
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    axes[0].set_title("Compression", loc="left", weight="bold")
    axes[1].set_title("Physical observable", loc="left", weight="bold")
    fig.suptitle(
        "Retained variance and observable fidelity are separate tests",
        x=0.07,
        ha="left",
        weight="bold",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_winner_probabilities(probabilities: pd.DataFrame, path: Path) -> None:
    """Plot how often each learner wins the local observable across paired seeds."""
    data = probabilities[
        probabilities["metric"].eq("steepest_gradient_location_error_radians")
        & probabilities["model"].eq("ridge")
    ]
    data = (
        data.groupby(["pod_modes", "representation"], as_index=False)
        ["winner_probability"]
        .mean()
    )
    labels = {
        "grid": "Grid",
        "pod_l2": "L2 POD",
        "pod_gradient": "Gradient-aware POD",
    }
    colors = {"grid": "#666666", "pod_l2": "#3b6ea8", "pod_gradient": "#d46a3a"}
    styles = {
        "grid": {"linestyle": ":", "marker": "s"},
        "pod_l2": {"linestyle": "--", "marker": "o", "markerfacecolor": "white", "zorder": 3},
        "pod_gradient": {"linestyle": "-", "marker": "x", "zorder": 2},
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for representation in ("grid", "pod_gradient", "pod_l2"):
        subset = data[data["representation"].eq(representation)]
        ax.plot(
            subset["pod_modes"],
            subset["winner_probability"],
            linewidth=2.2,
            label=labels[representation],
            color=colors[representation],
            **styles[representation],
        )
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1)
    ax.set_xlabel("Modal budget")
    ax.set_ylabel("Probability that ridge wins")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(sorted(data["pod_modes"].unique()))
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title("Representation changes the winning probability", loc="left", weight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_robustness_outputs(
    output: Path,
    robustness: RobustnessConfig = RobustnessConfig(),
    base: BurgersConfig = BurgersConfig(),
) -> None:
    """Run the robustness sweep and persist its complete evidence trail."""
    output.mkdir(parents=True, exist_ok=True)
    raw = run_robustness_experiment(robustness, base)
    summary = summarize_robustness(raw)
    probabilities = winner_probabilities(raw, robustness)
    raw.to_csv(output / "burgers_robustness_runs.csv", index=False)
    summary.to_csv(output / "burgers_robustness_summary.csv", index=False)
    probabilities.to_csv(output / "burgers_robustness_winner_probabilities.csv", index=False)
    plot_robustness(summary, output / "burgers_robustness_variance_observable.png")
    plot_winner_probabilities(
        probabilities,
        output / "burgers_robustness_winner_probability.png",
    )
