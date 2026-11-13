"""Oracle diagnostic: restore one local gradient functional outside rank-16 POD.

Directions are selected from the reduced state before propagation, never from
decision outcomes. Controls are gradient-orthogonal, NOT asserted to be irrelevant
to the nonlinear future decision. All pathways use the same full-grid posterior.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.signal import resample

from .action_conditioned_decisions import (
    ActionConfig, _prepare_systems, _controlled_burgers, _loss_matrix,
)
from .representation_boundaries import _pod_reconstruct, _posterior, _burgers_qoi


def slope_vector(points: int, index: int) -> np.ndarray:
    """Descending centered difference, in the benchmark's frozen units."""
    vector = np.zeros(points)
    vector[(index - 1) % points] = 0.5
    vector[(index + 1) % points] = -0.5
    return vector


def restore_local_gradient(full, mean, basis, control_vectors):
    reduced = _pod_reconstruct(full[None], mean, basis)[0]
    index = int(np.argmin(np.roll(reduced, -1) - np.roll(reduced, 1)))
    functional = slope_vector(full.size, index)
    # Enforce both frozen POD coordinates and spatial mean.
    constraints = np.vstack((basis, np.ones(full.size) / np.sqrt(full.size)))
    _, singular, vt = np.linalg.svd(constraints, full_matrices=False)
    rows = vt[singular > 1e-10]

    def residual(vector):
        return vector - (vector @ rows.T) @ rows

    direction = residual(functional)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("Selected gradient is already spanned by the representation")
    direction /= norm
    coefficient = float((full - reduced) @ functional / (direction @ functional))
    correction = coefficient * direction
    controls = []
    for raw in control_vectors:
        vector = residual(raw)
        vector -= (vector @ direction) * direction
        length = np.linalg.norm(vector)
        if length < 1e-12:
            raise ValueError("Degenerate gradient-orthogonal control")
        controls.append(reduced + abs(coefficient) * vector / length)
    return reduced, reduced + correction, np.asarray(controls), {
        "gradient_index": index,
        "coefficient": coefficient,
        "added_energy": float(correction @ correction),
        "full_gradient_at_selected_site": float(full @ functional),
        "rank16_gradient_at_selected_site": float(reduced @ functional),
        "restored_gradient_at_selected_site": float((reduced + correction) @ functional),
        "pod_coordinate_change": float(np.linalg.norm(correction @ basis.T)),
        "mean_change": float(correction.mean()),
        "control_energy_error": float(max(abs(np.sum((c - reduced)**2) - correction @ correction) for c in controls)),
        "control_gradient_change": float(max(abs((c - reduced) @ functional) for c in controls)),
    }


def independent_response(states, viscosities, gains, horizon, points, rtol=1e-9):
    """Independent MOL: fourth-order finite differences + adaptive DOP853.

    Resample the same starting fields, propagate, then sample onto the original
    grid. The QoI remains the original centered difference (no changed dx factor).
    This checks numerical continuation, not model discrepancy or initial truth.
    """
    initial = resample(states, points, axis=-1)
    shape = initial.shape
    dx = 2 * np.pi / points
    nu = np.asarray(viscosities)[:, None, None]
    gain = np.asarray(gains)[None, :, None]

    def rhs(_, flat):
        u = flat.reshape(shape)
        plus1, minus1 = np.roll(u, -1, axis=-1), np.roll(u, 1, axis=-1)
        plus2, minus2 = np.roll(u, -2, axis=-1), np.roll(u, 2, axis=-1)
        derivative = (-plus2 + 8 * plus1 - 8 * minus1 + minus2) / (12 * dx)
        laplacian = (-plus2 + 16 * plus1 - 30 * u + 16 * minus1 - minus2) / (12 * dx**2)
        return (-u * derivative + nu * laplacian - gain * (u - u.mean(axis=-1, keepdims=True))).ravel()

    result = solve_ivp(rhs, (0.0, horizon), initial.ravel(), method="DOP853",
                       rtol=rtol, atol=rtol * 0.01, t_eval=[horizon])
    if not result.success:
        raise RuntimeError(result.message)
    field = result.y[:, -1].reshape(shape)
    if points % states.shape[-1]:
        raise ValueError("Reference grid must be a multiple of the decision grid")
    return field[..., ::points // states.shape[-1]]


def qois(fields):
    return -np.min(0.5 * (np.roll(fields, -1, axis=-1) - np.roll(fields, 1, axis=-1)), axis=-1)


def run(output: Path, draws: int = 2000, controls: int = 20):
    if draws < 1 or controls < 1:
        raise ValueError("Positive draw and control counts required")
    config = ActionConfig()
    rng = np.random.default_rng(config.seed)
    system = _prepare_systems(config, rng)[0]  # prepare BOTH systems to preserve legacy RNG
    control_seed, new_noise_seed = 2026111301, 2026111302
    control_vectors = np.random.default_rng(control_seed).normal(size=(controls, config.grid_points))
    new_rng = np.random.default_rng(new_noise_seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "upstream_commit": "cbe947d1564248ad44459912badb2b76699e890e",
        "config": asdict(config), "new_noise_draws_per_case": draws,
        "control_count": controls, "control_seed": control_seed, "noise_seed": new_noise_seed,
        "candidate": "one descending centered-gradient functional at rank16 steepest site, before control",
        "selection": "all ten original Burgers action cases; no outcome-based case filtering",
        "posterior": "full grid, frozen identically across representation arms",
        "reference": "fourth-order finite differences with DOP853, grids 128/256/512, same initial field",
        "qoi_convergence_relative_tolerance": 0.001,
        "expected_loss_convergence_absolute_tolerance": 0.0001,
        "reference_action_stability": "fine-grid margin > 2 * maximum per-action fine/medium expected-loss difference",
        "scope": "oracle information-restoration diagnostic, not a deployable reduced solver",
    }
    # Write the specification before any outcome is evaluated.
    (output / "specification.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records, features, convergence, physics = [], [], [], []
    for local_case in range(config.evaluation_cases):
        ci = config.calibration_cases + local_case
        states = system["libraries"][ci, :, -1]
        observation = system["truth"][ci, 1:]
        candidates = system["libraries"][ci, :, 1:].reshape(len(states), -1)
        sigma = float(system["noise_sigma"])
        legacy = rng.normal(0, sigma, size=(30,) + observation.shape)
        fresh = new_rng.normal(0, sigma, size=(draws,) + observation.shape)
        variants = {"full_grid": states, "rank32": _pod_reconstruct(states, system["mean"], system["basis"][:32])}
        reduced, restored, matched = [], [], []
        for pi, state in enumerate(states):
            r, s, c, diagnostics = restore_local_gradient(state, system["mean"], system["basis"][:16], control_vectors)
            reduced.append(r); restored.append(s); matched.append(c)
            features.append({"case": local_case, "parameter_index": pi, **diagnostics})
        variants["rank16"] = np.asarray(reduced)
        variants["gradient_restored"] = np.asarray(restored)
        for index in range(controls):
            variants[f"control_{index:02d}"] = np.asarray(matched)[:, index]
        losses = {}
        for label, starting in variants.items():
            response = np.asarray([[_controlled_burgers(state, float(nu), float(system["horizon"]), config.burgers_dt, gain)
                                     for gain in config.burgers_control_gains]
                                    for state, nu in zip(starting, config.burgers_viscosities)])
            values = qois(response)
            losses[label] = _loss_matrix(values, system["hazard_threshold"], system["action_costs"])
            for pi, row in enumerate(values):
                for ai, value in enumerate(row):
                    physics.append({"case": local_case, "arm": label, "parameter_index": pi, "action": ai, "qoi": value})
        initial = np.repeat(states[:, None], 3, axis=1)
        references = {n: qois(independent_response(initial, config.burgers_viscosities,
                        config.burgers_control_gains, float(system["horizon"]), n)) for n in (128, 256, 512)}
        ref_losses = {n: _loss_matrix(q, system["hazard_threshold"], system["action_costs"]) for n, q in references.items()}
        ref_error = float(np.max(np.abs(references[512] - references[256]) / np.maximum(np.abs(references[512]), 1e-12)))
        convergence.append({"case": local_case, "qoi_max_relative_128_256": float(np.max(np.abs(references[128] - references[256]) / np.maximum(np.abs(references[256]), 1e-12))),
                            "qoi_max_relative_256_512": ref_error, "qoi_converged": ref_error < 0.001,
                            "loss_max_absolute_256_512": float(np.max(np.abs(ref_losses[512] - ref_losses[256]))),
                            "hazard_threshold": float(system["hazard_threshold"])})
        true_index = list(config.burgers_viscosities).index(float(system["parameters"][ci]))
        for batch, noise in (("legacy", legacy), ("fresh", fresh)):
            posteriors = np.asarray([_posterior((observation + d).ravel(), candidates, sigma) for d in noise])
            posterior_hash = hashlib.sha256(posteriors.tobytes()).hexdigest()
            full_expected = posteriors @ losses["full_grid"]
            full_choices = full_expected.argmin(axis=1)
            expected = posteriors @ ref_losses[512]
            coarse_expected = posteriors @ ref_losses[256]
            errors = np.max(np.abs(expected - coarse_expected), axis=1)
            best = expected.argmin(axis=1)
            margin = np.diff(np.sort(expected, axis=1)[:, :2], axis=1)[:, 0]
            stable = (margin > 2 * errors) & (errors < 0.0001) & (ref_error < 0.001)
            for label, arm_losses in losses.items():
                choices = (posteriors @ arm_losses).argmin(axis=1)
                for di, action in enumerate(choices):
                    records.append({"case": local_case, "batch": batch, "draw": di, "arm": label,
                        "posterior_hash": posterior_hash, "action": int(action), "full_grid_action": int(full_choices[di]),
                        "reference_action": int(best[di]), "reference_stable": bool(stable[di]),
                        "disagree_full_grid": bool(action != full_choices[di]), "disagree_reference": bool(action != best[di]),
                        "reference_regret": float(expected[di, action] - expected[di].min()),
                        "extra_loss_vs_full_grid": float(expected[di, action] - expected[di, full_choices[di]]),
                        "true_parameter_regret": float(ref_losses[512][true_index, action] - ref_losses[512][true_index].min()),
                        "reference_loss_numerical_difference": float(errors[di])})
        print(f"case {local_case}: reference QoI difference {ref_error:.3g}", flush=True)
    paired = pd.DataFrame(records)
    paired.to_csv(output / "paired.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(features).to_csv(output / "restoration_checks.csv", index=False)
    pd.DataFrame(convergence).to_csv(output / "reference_convergence.csv", index=False)
    pd.DataFrame(physics).to_csv(output / "response_qois.csv", index=False)
    summary = paired.groupby(["case", "batch", "arm"], as_index=False).agg(
        draws=("draw", "size"), disagreements=("disagree_full_grid", "sum"),
        reference_disagreements=("disagree_reference", "sum"),
        reference_stable_fraction=("reference_stable", "mean"),
        mean_reference_regret=("reference_regret", "mean"), maximum_reference_regret=("reference_regret", "max"),
        mean_extra_loss_vs_full_grid=("extra_loss_vs_full_grid", "mean"),
        mean_true_parameter_regret=("true_parameter_regret", "mean"))
    summary.to_csv(output / "summary.csv", index=False)
    comparisons = []
    bootstrap = np.random.default_rng(2026111303)
    for case, group in paired[paired.batch.eq("fresh")].groupby("case"):
        base = group[group.arm.eq("rank16")].sort_values("draw")
        target = group[group.arm.eq("gradient_restored")].sort_values("draw")
        delta = base.reference_regret.to_numpy() - target.reference_regret.to_numpy()
        indices = bootstrap.integers(0, len(delta), size=(2000, len(delta)))
        lo, hi = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
        control_summary = summary[(summary.case.eq(case)) & summary.batch.eq("fresh") & summary.arm.str.startswith("control_")]
        comparisons.append({"case": case, "draws": len(delta),
            "rank16_disagreements": int(base.disagree_full_grid.sum()),
            "restored_disagreements": int(target.disagree_full_grid.sum()),
            "rescued_full_grid_actions": int((base.disagree_full_grid.to_numpy() & ~target.disagree_full_grid.to_numpy()).sum()),
            "new_full_grid_disagreements": int((~base.disagree_full_grid.to_numpy() & target.disagree_full_grid.to_numpy()).sum()),
            "mean_reference_regret_reduction": float(delta.mean()),
            "paired_bootstrap_95_low": float(lo), "paired_bootstrap_95_high": float(hi),
            "control_disagreements_min": int(control_summary.disagreements.min()),
            "control_disagreements_median": float(control_summary.disagreements.median()),
            "control_disagreements_max": int(control_summary.disagreements.max()),
            "controls_with_regret_at_most_target": int((control_summary.mean_reference_regret <= target.reference_regret.mean()).sum())})
    pd.DataFrame(comparisons).to_csv(output / "paired_comparison.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/selective_restoration"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--controls", type=int, default=20)
    args = parser.parse_args()
    run(args.output, args.draws, args.controls)
