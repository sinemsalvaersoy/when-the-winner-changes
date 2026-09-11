"""Controlled representation experiment for the viscous Burgers equation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
