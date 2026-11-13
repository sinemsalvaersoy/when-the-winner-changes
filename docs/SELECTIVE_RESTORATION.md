# Selective restoration of a local gradient outside rank 16

This diagnostic tests whether restoring one missing local gradient in the
pre-control field is sufficient to recover intervention decisions. It does not
assume that a larger POD representation succeeds for this reason.

## Frozen design

The experiment starts from upstream commit
`cbe947d1564248ad44459912badb2b76699e890e` and uses the Burgers portion of
`ActionConfig`: ten evaluation cases, eight calibration cases, 64 grid points,
seven viscosity candidates, three control gains, and the original calibrated
hazard threshold and synthetic action costs. Both PDE systems are prepared so
the legacy noise stream is unchanged. The existing boundary-audit case 7 and
its 2/30 binary-label disagreements are a separate experiment.

Every arm receives the identical full-grid posterior. The only intervention is
on the field at the start of controlled propagation. No posterior is refitted,
and no cost, action, calibration threshold, or observation is changed.

For each parameter candidate, let `v` be the rank-16 reconstruction. Select the
site `j` of its steepest descending centered difference, before examining any
controlled outcome. Let `g_j` be that linear difference functional and `P_N`
the projection orthogonal to both the retained POD rows and the constant mode.
The unit direction is `d=P_N g_j/||P_N g_j||`. The correction is
`delta = ((g_j @ (u-v))/(g_j @ d))*d`. Thus the selected local gradient is
restored exactly, while POD coefficients and spatial mean remain unchanged.

Twenty fixed-seed random directions are projected into the same discarded
subspace and made orthogonal to `d`. Each receives exactly the same correction
norm as the target, separately for every case and parameter. Each arm adds one
direction. These are gradient-orthogonal controls, not known decision-irrelevant
controls: nonlinear evolution can make them affect the future maximum gradient.
The target is an adaptive direction plus its selected location, not a fixed
rank-17 POD code. Rank 32 is a larger-capacity comparator, not a matched control.

This is an oracle diagnostic: the missing scalar is read from the full field.
It tests the consequences of supplying that information, not whether a deployed
reduced model can infer it from the existing sensors.

## Reference and outcomes

The numerical reference is a separately implemented fourth-order periodic
finite-difference method with adaptive DOP853, compared at 128, 256 and 512
points. It starts from the same uncompressed observation-time fields. The final
field is sampled onto the original 64 points before applying the unchanged QoI.
The benchmark QoI is an unnormalised centered difference; adding a grid-spacing
factor only in the reference would silently change the decision rule.

The reference implementation is tested against an analytic unforced Cole-Hopf
solution and a constant solution under feedback. This is independent numerical
continuation, not experimental validation or an independent physical model.
It shares initial fields, viscosity hypotheses, control law and loss function.

Before outcomes are evaluated, `specification.json` records a relative QoI
convergence tolerance of 0.001, an absolute expected-loss difference tolerance
of 0.0001, and a reference action-margin criterion greater than twice the
maximum expected-loss change between the two finest resolutions. These are
empirical convergence screens, not rigorous error bounds. Failed screens remain
flagged in the paired data; no universal reference-validity claim is made.

The audit reproduces the original 30 paired noise draws and adds 2,000 fresh
paired draws per case. Noise uncertainty remains conditional on these ten
physical cases and the original case-generation seed. It records agreement with
the original full-grid action, agreement with the reference-optimal action,
reference posterior-expected regret, signed extra loss relative to the
full-grid choice, and true-parameter regret. Full-grid agreement alone is not
treated as physical correctness.

Paired bootstrap intervals use 2,000 resamples of noise draws within each case.
They do not describe uncertainty over new physical cases. The twenty controls
are reported as a distribution, without a post hoc significance claim.

## Reproduce

```bash
pip install -e '.[dev,restoration]'
python -m pinn_audit.selective_restoration
pytest -q
```

The command writes `results/selective_restoration/`: `specification.json`,
`paired.csv.gz`, `summary.csv`, `paired_comparison.csv`,
`restoration_checks.csv`, `reference_convergence.csv`, and `response_qois.csv`.
The large paired file is regenerated locally; compact results are versioned.
See `RESULTS.md` in that directory for the measured outcome and limits.
