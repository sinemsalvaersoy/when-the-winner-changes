# Distributed restoration: the previously failed case remains primary

13 November 2026. This sequential exploratory experiment retains case 5,
the original local-gradient failure, with the same 2,000 paired noise draws.
They are not newly independent draws relative to the preceding experiment.
All ten physical cases remain in the evaluation. The original local result
is preserved as a comparator, not replaced by the new result.

## Decision result

Disagreements with the original full-grid action, out of 2,000 per case:

| Representation | Case 4 | Primary case 5 |
| :--- | ---: | ---: |
| Rank 16 | 420 | 1328 |
| Original local gradient restored | 389 | 1328 |
| Original local direction at the new correction's energy | 133 | 1328 |
| Distributed sensitivity, original local gradient fixed | 14 | 340 |
| 20 equal-energy first-order-null controls | 420 each | 1328 each |
| Rank 32 | 6 | 35 |

Case 1 changes from one disagreement to zero. The remaining seven cases keep
zero disagreements for the target. These observations are conditional on the
fixed cases, finite action set and noise model.

The net improvement in case 5 is not a monotone rescue: all 1,328 previous
disagreements are corrected, but 340 new disagreements are introduced. The
reference recommends moderate control in 1,328 draws and strong control in
672. The new target chooses moderate control for all 1,328 of the first group,
but also for 340 of the second group. Thus the remaining failures are weaker
control than the reference recommends. The target's 17% disagreement still
fails the original 5% criterion. Rank 32 has 1.75% disagreement in this case.

Case 4 recovers 406 previous choices with no new disagreements. Its target
disagreement is 0.7%, with 14 full-grid disagreements and 13 reference
disagreements because one original grid decision differs from the reference.

## Consequences under the frozen synthetic loss

Mean posterior-expected regret against the finite-difference reference:

| Representation | Case 4 | Case 5 |
| :--- | ---: | ---: |
| Rank 16 | 0.001636361 | 0.010612829 |
| Original local gradient | 0.001389669 | 0.010612829 |
| Energy-matched local gradient | 0.000148008 | 0.010612829 |
| Distributed sensitivity | 0.000001874 | 0.001274203 |
| Rank 32 | 0.000000255 | 0.000012364 |

The primary case's mean regret reduction is 0.009338626, approximately 88%.
Its conditional paired-bootstrap 95% interval is approximately 0.008833 to
0.009864. Case 4's reduction is 0.001634487, with interval approximately
0.001470 to 0.001808. These intervals concern paired noise variation in fixed,
previously examined cases, not unseen systems. Losses are dimensionless and
synthetic; no real-world safety or monetary benefit is demonstrated.

## What information was supplied?

For each viscosity candidate, one weighted coefficient of the discarded
control-onset field is supplied. The weight vector is selected using the joint
discrete-adjoint sensitivity of the three controlled final gradient QoIs at the
rank-16 state. It is constrained to preserve the initial gradient functional
used by the first experiment, the POD coordinates, and the field mean.

In case 5, the initial selected site is 48; the reduced controlled final
maximum is at site 49 for all three gains. The sensitivity direction has an
effective support of about 4.6 to 8.0 grid points across viscosity candidates
(inverse participation ratio). About 49% to 71% of its squared weight lies
within two sites of the original steepest site. This is a spatially weighted
front-neighbourhood correction, not evidence that the entire domain is needed.
At lower viscosities some corrected maxima switch active site, so a fixed-site
linear derivative does not completely describe the nonlinear response.

For the true-parameter candidate in case 5 (viscosity 0.062), the moderate-control
QoI is 0.267275 at rank 16, 0.257504 on the full grid, 0.255971 after the
distributed correction, and 0.257294 at rank 32. The new correction improves
the prediction but overshoots. This is consistent with remaining decision
errors, not proof of a complete mechanism across the parameter posterior.

The leading sensitivity direction captures over 99.8% of the three-row
projected sensitivity energy in case 5. That is a statement about the local
derivative matrix, not about field variance or global decision sufficiency.

## Verification and limits

36 tests pass, including adjoint versus two-sided finite differences at all
three gains, preserved initial local slope/POD/mean, matched control energy,
analytic Cole-Hopf convergence of the independent numerical continuation,
and the existing boundary and action tests. CI installs the restoration extra
so the new tests are not silently skipped for a missing SciPy dependency.

For all ten cases and both noise batches, baseline full-grid/rank16/rank32
decisions and posterior hashes reproduce the preceding experiment. Original
local-restoration decisions also reproduce exactly. The largest selected
initial-gradient change is below 7.7e-17; POD change below 2.0e-16; mean change
below 2.7e-18. Control first-order QoI change is below 6.6e-17. Thus the measured
improvement does not come from refitting inference or restoring the selected
initial local gradient.

The numerical-reference screens remain unchanged: all 2,000 primary-case
draws pass; 1,994/2,000 case-4 draws pass. Other cases retain the reference
limitations documented in the previous experiment. The separate solver shares
the physical model and starting state and is not experimental validation.

The new result supports a contribution from a particular distributed initial
field coefficient. It does not show that rank 32 works through exactly this
coefficient, that it is necessary, or that the coefficient can be measured from
the current sensors. Each parameter candidate still uses its own oracle scalar.
Equal-energy null controls do not span all alternative physical explanations.

The next experiment should address the 340 introduced errors explicitly,
without tuning on these draws and presenting that as confirmation. A change
to multiple sensitivity directions, active-site treatment or a decision-loss
target would be a new representation and needs a new, frozen specification
and held-out physical cases.
