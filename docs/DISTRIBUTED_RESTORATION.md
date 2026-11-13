# Distributed initial-state information for controlled gradient decisions

This follow-up retains case 5 as the primary failure of the local-gradient
restoration experiment. All ten original action-audit cases remain included.
The previous result is not overwritten. This is a sequential exploratory
experiment, not independent confirmation on unseen physical cases.

## Candidate fixed before outcome evaluation

At each viscosity candidate, start with its rank-16 state at control onset
(t=0.12). For each of the three fixed gains, differentiate the controlled
final maximum descending difference at t=0.24 with respect to that starting
field. The code implements the discrete adjoint of the existing RK4 map.
The active maximum is chosen from the reduced prediction; full-field future
responses and action labels do not enter direction selection. Finite-difference
tests at gains 0, 2 and 5 check the derivative implementation.

Stack the three derivative rows with equal, unnormalised weights. Project them
orthogonal to the rank-16 POD rows, the spatial mean, and the descending-gradient
functional at the original reduced state's steepest site. The leading right
singular vector defines one unit direction d. The correction is
`((u_full-u_rank16) @ d)*d`.

This restores one missing weighted field coefficient. It leaves the existing
POD coordinates, mean, and selected initial local gradient unchanged. It need
not leave the maximum over all initial sites unchanged; nor does it enforce
the nonlinear final QoI exactly. The active final site can change, and the
derivative is branch-specific at ties. The output records active-site gaps,
site changes, and linear predicted versus actual nonlinear QoI changes.

The direction is allowed to depend on the reduced field, viscosity, and all
three known controls, but not on the selected action, posterior sample, loss
gradient, full future response, or observed success. It is an oracle diagnostic:
the coefficient supplied along d is measured from the full initial field.
This is not a deployable 17-mode POD model or a training-free cost reduction.
It requires PDE derivatives and a missing scalar for every parameter candidate.

## Comparators

* Frozen rank 16, full grid, and rank 32.
* The original local-gradient restoration, without changing its result.
* That same local-gradient direction rescaled to the new correction's norm.
  This separates direction selection from merely adding more perturbation energy.
* Twenty random one-direction controls of the same norm. These preserve the
  selected initial gradient, POD coordinates and mean, and have zero first-order
  effect on all three final QoIs at the reduced state. They can still affect
  nonlinear outcomes. They are mechanistic null controls, not random samples
  from all possible useful representation changes.

Every comparison keeps the full-grid posterior, observations, parameter grid,
calibrated threshold, action costs, actions and decision rule fixed. The same
30 legacy and 2,000 fresh paired draws per case from the previous experiment
are reused deliberately. They must not be counted as new independent evidence.

## Numerical assessment and uncertainty

The prior 128/256/512 finite-difference/DOP853 reference procedure and convergence
screens remain unchanged. It shares the physical model and control-onset state;
independence is numerical, not experimental. Unresolved reference draws remain
flagged. Report agreement and expected regret separately, including cases where
the full-grid action is not the reference action.

Paired bootstrap intervals resample noise draws within each fixed case. The
control range is descriptive. No population claim over physical systems or
formal multiple-comparison significance claim follows from these intervals.
The distribution of sensitivity weights is reported; nonzero weights alone
do not prove every spatial point is necessary.

## Reproduce

```bash
pip install -e '.[dev,restoration]'
python -m pinn_audit.distributed_restoration
pytest -q
```

Outputs are written to `results/distributed_restoration`. The large
`paired.csv.gz` and `sensitivity_directions.csv` are regenerated locally.
Compact summaries, physical-response tables, convergence data, and restoration
checks are versioned. The specification is written before evaluation.
