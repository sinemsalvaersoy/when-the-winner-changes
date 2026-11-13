import numpy as np
import pytest

pytest.importorskip("scipy")
from pinn_audit.distributed_restoration import adjoint_qoi, sensitivity_correction
from pinn_audit.action_conditioned_decisions import _controlled_burgers
from pinn_audit.representation_boundaries import _burgers_initial, _burgers_qoi
from pinn_audit.selective_restoration import slope_vector


@pytest.mark.parametrize("gain", [0.0, 2.0, 5.0])
def test_discrete_adjoint_matches_two_sided_directional_differences(gain):
    rng = np.random.default_rng(2026)
    u = _burgers_initial(rng, 64)
    value, gradient, index, gap = adjoint_qoi(u, .03, .12, .002, gain)
    assert gap > 1e-5
    assert np.isclose(value, _burgers_qoi(_controlled_burgers(u, .03, .12, .002, gain)), atol=1e-14)
    for _ in range(3):
        d = rng.normal(size=64); d /= np.linalg.norm(d)
        eps = 1e-5
        fd = (_burgers_qoi(_controlled_burgers(u+eps*d, .03, .12, .002, gain))
              - _burgers_qoi(_controlled_burgers(u-eps*d, .03, .12, .002, gain))) / (2*eps)
        assert np.isclose(fd, gradient @ d, rtol=1e-6, atol=1e-9)


def test_distributed_direction_preserves_local_slope_and_matched_controls_are_null():
    rng = np.random.default_rng(44)
    full, mean = rng.normal(size=(2,64))
    basis = np.linalg.qr(rng.normal(size=(64,16)))[0].T
    jacobian = rng.normal(size=(3,64))
    r, restored, controls, direction, diag = sensitivity_correction(full, mean, basis, jacobian, rng.normal(size=(20,64)))
    delta = restored-r
    g = slope_vector(64, diag['initial_gradient_index'])
    assert np.allclose(delta @ basis.T, 0, atol=1e-12)
    assert np.isclose(delta.mean(), 0, atol=1e-12)
    assert np.isclose(delta @ g, 0, atol=1e-12)
    assert np.allclose((controls-r) @ jacobian.T, 0, atol=1e-12)
    assert np.allclose((controls-r) @ basis.T, 0, atol=1e-12)
    assert np.allclose((controls-r) @ g, 0, atol=1e-12)
    assert np.allclose(np.linalg.norm(controls-r,axis=1), np.linalg.norm(delta), atol=1e-12)
    assert np.isclose((full-restored) @ direction, 0, atol=1e-12)
