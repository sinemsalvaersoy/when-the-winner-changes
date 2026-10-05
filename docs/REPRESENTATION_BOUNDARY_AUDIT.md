# Representation boundary audit

## Research question

What a representation must preserve depends on what the representation is for.
Accurate state reconstruction does not guarantee accurate evolution. Preserving
both does not guarantee parameter inference or the decision built from it.

For the same physical case, adequacy can depend on spatial resolution for the
state, time horizon for the dynamics, the parameter being inferred, and the
actions and loss encoded in the decision rule. The audit therefore asks:

> For which purpose is a representation sufficient, and where does it cease to
> be sufficient?

## Four boundaries

The experiment sweeps the same ordered POD ranks for viscous Burgers and a
fixed-end damped wave equation. Each case receives four distinct tests.

| Boundary | Quantity | Prespecified pass rule |
|:--|:--|:--|
| State | Relative L2 reconstruction error over observation fields | at most 0.05 |
| Dynamics | Relative L2 future-field error after evolving the compressed initial state | at most 0.15 |
| Inference | Mean total-variation distance from the full-grid parameter posterior | at most 0.10 |
| Decision | Disagreement with the full-grid action and false-safe rate relative to that action | both at most 0.05 |

The decision is to intervene when posterior probability of the physical hazard
exceeds 0.5. The physical hazard threshold is the median future quantity of
interest on separate calibration cases. Evaluation cases never set the hazard
threshold.

The preservation false-safe rate means that reduced representation says safe
when the paired full-grid posterior says intervene. False-safe and false-alarm
rates against the simulated physical label are also retained, but do not define
the representation boundary because they mix representation loss with the
conditional full-grid oracle's own decision error.

For Burgers, the inferred parameter is viscosity and the future quantity is the
magnitude of the steepest negative gradient. For the damped wave system, the
inferred parameter is damping and the future quantity is absolute displacement
near three quarters of the domain.

## Boundary definition

A first passing rank can be misleading when a criterion passes at one rank and
fails again at a larger rank. The primary result is therefore the stable
boundary: the smallest rank at which the criterion passes and continues to pass
at every larger tested rank.

The output separately records:

1. first passing rank
2. stable boundary rank
3. width between first and stable passage
4. order violations
5. unresolved boundaries

This prevents monotonicity from being silently assumed.

## Controls

All modal ranks share the same physical cases and full-grid Gaussian noise
draws. Posterior changes are therefore paired within case and noise draw.
Full-rank POD is an exact-subspace negative control for state reconstruction and
compressed-initial-condition dynamics. It should recover the full-grid result
to numerical precision.

The full-grid posterior is a conditional oracle, not ground truth about the
physical world. It isolates representation loss under this simulator, parameter
grid, prior, noise model, and decision rule.

## Results

All 80 case-criterion boundaries are resolved by rank 32. Full-rank POD at 64
modes recovers state and compressed-initial-condition dynamics to numerical
precision, passing the exact-subspace negative control.

| PDE family | State median | Dynamics median | Inference median | Decision median |
|:--|--:|--:|--:|--:|
| Burgers | 32 | 16 | 16 | 2 |
| Damped wave | 32 | 32 | 16 | 2 |

The median is not an ordering theorem. Burgers case 7 requires rank 32 for the
decision even though state and dynamics stabilize at rank 16. Conversely, most
cases preserve the binary action at much lower rank than the posterior or field.
The action can therefore discard information safely when the lost directions do
not cross its threshold, while a boundary case can make the same loss critical.

No pass-then-fail order violations occur in this versioned sweep. That is an
observed property of these cases and thresholds, not an assumed property of POD.

The design contains 20 evaluation cases, 30 paired noise draws per case, and six
modal ranks. Burgers evaluation contains eight hazardous and two safe cases;
damped wave contains four hazardous and six safe cases. The class counts are
reported because decision preservation can look artificially easy when the
tested cases do not approach the action boundary.

![Case-level representation boundaries](../results/representation_boundaries.png)

## Interpretation

The result separates four questions that retained variance or field error alone
cannot answer. It also shows why the boundaries need not become progressively
harder from state to decision. A decision rule is an information bottleneck of
its own: it may ignore posterior changes that leave the selected action fixed.
But near its threshold, a small representation loss can matter more than a much
larger field error elsewhere.

The claim is conditional on the two simulators, parameter grids, Gaussian noise,
uniform discrete priors, calibrated hazard thresholds, and binary loss. It is
not evidence that rank 2 is generally adequate for physical decisions.

## Run

```bash
pinn-audit --boundaries
```

The command writes case metrics, paired noise draws, extracted boundaries,
summary statistics, the complete threshold specification, and the principal
boundary figure to `results/`.
