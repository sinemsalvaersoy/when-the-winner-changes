import numpy as np
import pytest

pytest.importorskip("scipy")
from pinn_audit.selective_restoration import independent_response, restore_local_gradient, slope_vector


def test_restoration_preserves_coordinates_mean_and_matches_control_energy():
    rng = np.random.default_rng(7)
    basis = np.linalg.qr(rng.normal(size=(64, 16)))[0].T
    full, mean = rng.normal(size=(2, 64))
    reduced, restored, controls, d = restore_local_gradient(full, mean, basis, rng.normal(size=(20, 64)))
    g = slope_vector(64, d["gradient_index"])
    assert np.isclose(restored @ g, full @ g, atol=1e-12)
    assert np.allclose((restored - reduced) @ basis.T, 0, atol=1e-12)
    assert np.isclose(restored.mean(), reduced.mean(), atol=1e-12)
    assert np.allclose((controls - reduced) @ basis.T, 0, atol=1e-12)
    assert np.allclose((controls - reduced).mean(axis=1), 0, atol=1e-12)
    assert np.allclose(controls @ g, reduced @ g, atol=1e-12)
    assert np.allclose(np.linalg.norm(controls - reduced, axis=1), np.linalg.norm(restored - reduced), atol=1e-12)


def test_independent_solver_against_exact_cole_hopf_solution():
    # phi=1+a*exp(-nu*t)*cos(x), u=-2*nu*d_x(log(phi)).
    n, nu, a, t = 64, 0.04, 0.35, 0.12
    x = 2 * np.pi * np.arange(n) / n
    initial = 2 * nu * a * np.sin(x) / (1 + a * np.cos(x))
    decay = a * np.exp(-nu * t)
    exact = 2 * nu * decay * np.sin(x) / (1 + decay * np.cos(x))
    errors = []
    for points in (64, 128, 256):
        field = independent_response(initial[None, None], [nu], [0.0], t, points)[0, 0]
        errors.append(np.max(np.abs(field - exact)))
    assert errors[-1] < 1e-8
    assert errors[0] / errors[1] > 10
    assert errors[1] / errors[2] > 10


def test_independent_feedback_preserves_constant_state():
    initial = np.full((2, 3, 16), 0.42)
    result = independent_response(initial, [0.02, 0.05], [0, 2, 5], 0.12, 32)
    assert np.allclose(result, initial, atol=1e-12)
