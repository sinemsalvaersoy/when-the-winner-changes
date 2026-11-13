"""Decision-response sensitivity restoration with the initial local slope fixed.

The target is selected from reduced-state derivatives of ALL three controlled
QoIs, before seeing full-field responses or decision outcomes. Case 5 is retained
as the primary previously failed case. This is a sequential exploratory audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .action_conditioned_decisions import _controlled_burgers
from .representation_boundaries import _burgers_rhs, _pod_reconstruct
from .selective_restoration import run, slope_vector, qois, restore_local_gradient


def adjoint_qoi(initial, viscosity, horizon, dt, gain):
    """Exact discrete adjoint of the RK4 map on a fixed maximum-gradient branch.

    At ties the max is not differentiable; return the active-site gap so this
    limitation remains auditable. No full-grid target response is used here.
    """
    n = initial.size
    k = np.fft.fftfreq(n, d=1 / n)
    steps = int(round(horizon / dt))
    if not np.isclose(steps * dt, horizon):
        raise ValueError("Horizon must be an integer multiple of dt")

    def rhs(u):
        return _burgers_rhs(u, viscosity, k) - gain * (u - u.mean())

    def jt(u, v):
        du = np.fft.ifft(1j * k * np.fft.fft(u)).real
        product_derivative = np.fft.ifft(1j * k * np.fft.fft(u * v)).real
        lap_v = np.fft.ifft(-k**2 * np.fft.fft(v)).real
        return -du * v + product_derivative + viscosity * lap_v - gain * (v - v.mean())

    u = initial.copy()
    tape = []
    for _ in range(steps):
        a = rhs(u)
        u2 = u + dt / 2 * a
        b = rhs(u2)
        u3 = u + dt / 2 * b
        c = rhs(u3)
        u4 = u + dt * c
        d = rhs(u4)
        tape.append((u.copy(), u2, u3, u4))
        u = u + dt / 6 * (a + 2 * b + 2 * c + d)
    slopes = 0.5 * (np.roll(u, 1) - np.roll(u, -1))
    index = int(slopes.argmax())
    gap = float(np.sort(slopes)[-1] - np.sort(slopes)[-2])
    v = slope_vector(n, index)
    for u1, u2, u3, u4 in reversed(tape):
        incoming = v.copy()
        v1, v2, v3, v4 = (dt / 6 * incoming, dt / 3 * incoming,
                           dt / 3 * incoming, dt / 6 * incoming)
        v = incoming.copy()
        stage = jt(u4, v4)
        v += stage
        v3 += dt * stage
        stage = jt(u3, v3)
        v += stage
        v2 += dt / 2 * stage
        stage = jt(u2, v2)
        v += stage
        v1 += dt / 2 * stage
        v += jt(u1, v1)
    return float(slopes[index]), v, index, gap


def row_space(matrix):
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    return vt[singular > 1e-10]


def outside(vectors, rows):
    return vectors - (vectors @ rows.T) @ rows


def sensitivity_correction(full, mean, basis, sensitivities, controls):
    reduced = _pod_reconstruct(full[None], mean, basis)[0]
    local_index = int(np.argmin(np.roll(reduced, -1) - np.roll(reduced, 1)))
    local = slope_vector(len(full), local_index)
    constraints = row_space(np.vstack((basis, np.ones(len(full)) / np.sqrt(len(full)), local)))
    projected = outside(sensitivities, constraints)
    _, singular, vt = np.linalg.svd(projected, full_matrices=False)
    if singular[0] < 1e-12:
        raise ValueError("No controlled-QoI sensitivity remains outside constraints")
    direction = vt[0]
    if direction[np.argmax(abs(direction))] < 0:
        direction = -direction
    coefficient = float((full - reduced) @ direction)
    delta = coefficient * direction
    # Controls have zero FIRST-ORDER effect on all action QoIs and the old slope.
    null_constraints = row_space(np.vstack((constraints, sensitivities)))
    control_directions = outside(controls, null_constraints)
    norms = np.linalg.norm(control_directions, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("Degenerate sensitivity-null control")
    control_directions /= norms[:, None]
    control_delta = abs(coefficient) * control_directions
    diagnostics = {
        "initial_gradient_index": local_index,
        "added_energy": float(delta @ delta),
        "coefficient": coefficient,
        "initial_selected_gradient_change": float(delta @ local),
        "pod_coordinate_change": float(np.linalg.norm(delta @ basis.T)),
        "mean_change": float(delta.mean()),
        "control_energy_error": float(np.max(abs(np.sum(control_delta**2, axis=1) - delta @ delta))),
        "control_linear_qoi_change": float(np.max(abs(control_delta @ sensitivities.T))),
        "control_initial_gradient_change": float(np.max(abs(control_delta @ local))),
        "control_pod_change": float(np.max(abs(control_delta @ basis.T))),
        "leading_sensitivity_fraction": float(singular[0]**2 / np.sum(singular**2)),
        "direction_effective_grid_points": float(1 / np.sum(direction**4)),
        "direction_energy_within_two_sites": float(np.sum(direction[(local_index + np.arange(-2, 3)) % len(full)]**2)),
    }
    return reduced, reduced + delta, reduced + control_delta, direction, diagnostics


def run_distributed(output: Path, draws=2000, controls=20):
    direction_rows = []

    def restore(full, mean, basis, raw_controls, *, parameter, config, case, parameter_index):
        reduced = _pod_reconstruct(full[None], mean, basis)[0]
        horizon = config.burgers_decision_time - config.burgers_observation_times[-1]
        evaluations = [adjoint_qoi(reduced, parameter, horizon, config.burgers_dt, gain)
                       for gain in config.burgers_control_gains]
        jacobian = np.asarray([item[1] for item in evaluations])
        r, restored, matched, direction, d = sensitivity_correction(full, mean, basis, jacobian, raw_controls)
        _, local_restored, _, local_d = restore_local_gradient(full, mean, basis, raw_controls)
        local_delta = local_restored - r
        local_norm = np.linalg.norm(local_delta)
        if local_norm < 1e-14:
            raise ValueError("Cannot define energy-matched original local-gradient direction")
        local_energy_matched = r + np.linalg.norm(restored - r) * local_delta / local_norm
        extras = {"local_gradient_original": local_restored, "local_energy_matched": local_energy_matched}
        for ai, (value, gradient, index, gap) in enumerate(evaluations):
            field = _controlled_burgers(restored, parameter, horizon, config.burgers_dt, config.burgers_control_gains[ai])
            actual = float(qois(field))
            d.update({f"action{ai}_reduced_qoi": value,
                      f"action{ai}_linear_qoi_change": float(gradient @ (restored-r)),
                      f"action{ai}_actual_qoi_change": actual-value,
                      f"action{ai}_reduced_active_site": index,
                      f"action{ai}_restored_active_site": int(np.argmax(0.5*(np.roll(field,1)-np.roll(field,-1)))),
                      f"action{ai}_active_site_gap": gap})
        d["local_original_added_energy"] = local_d["added_energy"]
        d["local_matched_energy_error"] = float(abs(np.sum((local_energy_matched-r)**2)-d["added_energy"]))
        for j in range(len(full)):
            direction_rows.append({"case": case, "parameter_index": parameter_index, "grid_index": j,
                "direction": float(direction[j]), "correction": float(restored[j]-r[j]),
                **{f"action{ai}_sensitivity": float(jacobian[ai,j]) for ai in range(3)}})
        return r, restored, matched, d, extras

    result = run(output, draws, controls, restorer=restore, target_label="distributed_local_fixed",
        specification={
            "parent_experiment_commit": "917c8ce628daa08dabdf42e4e777f2f96db8334c",
            "candidate": "leading joint discrete-adjoint direction of three controlled QoIs outside POD, mean and initial local gradient",
            "primary_case": 5,
            "selection": "case5 retained as primary prior failure; all ten original cases retained",
            "controls": "one-direction equal-energy random corrections with zero first-order effect on all three QoIs; plus energy-matched original local direction",
            "outcome_status": "sequential exploratory test on previously examined cases and paired noise, not independent confirmation",
            "direction_weights": "equal unnormalised QoI sensitivities for all three gains; fixed before outcomes",
            "dimension_budget": "one adaptive direction plus one full-field scalar, not a fixed global rank17 model",
        })
    pd.DataFrame(direction_rows).to_csv(output / "sensitivity_directions.csv", index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/distributed_restoration"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--controls", type=int, default=20)
    args = parser.parse_args()
    run_distributed(args.output, args.draws, args.controls)
