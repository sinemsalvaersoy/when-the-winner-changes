# Selective restoration result, 13 November 2026

Restoring one local pre-control gradient gives a partial, selective improvement
in one Burgers case. It does not explain the larger rank-32 recovery.

All ten original action-audit cases were evaluated. Each arm used the same
full-grid posterior, paired noise, control actions, hazard threshold and loss.
The original 30-draw action disagreement counts were reproduced for every case
at ranks 16 and 32. This is separate from the README's binary boundary case 7.

## Fresh noise results

Counts below are disagreements with the original full-grid action, out of 2,000
new paired noise draws per case. Case indices are the repository's zero-based
indices. Controls add one gradient-orthogonal direction with the same energy
as the targeted correction, for each parameter candidate.

| Case | Rank 16 | Local gradient restored | 20 matched controls, min to max | Rank 32 |
| :--- | ---: | ---: | ---: | ---: |
| 1 | 1 | 0 | 0 to 1 | 0 |
| 4 | 420 | 389 | 409 to 426 | 6 |
| 5 | 1328 | 1328 | 1328 to 1328 | 35 |

The other seven cases had no full-grid action disagreements in these arms.
This is not a universal decision-agreement guarantee. The original 30 draws
had zero rank-32 disagreement, but the new draws expose remaining rank-32
sensitivity. No physical case was selected or discarded after these results.

In case 4, the target recovers 31 full-grid choices without introducing new
disagreements. All 31 changes pass the numerical-reference stability screen.
The target has lower reference expected regret than all 20 matched controls
tested. This finite control ensemble does not establish a universal or
multiplicity-adjusted significance claim.

Mean posterior-expected regret against the independent numerical continuation:

| Case | Rank 16 | Gradient restored | Rank 32 |
| :--- | ---: | ---: | ---: |
| 4 | 0.001636361 | 0.001389669 | 0.0000002552 |
| 5 | 0.010612829 | 0.010612829 | 0.0000123636 |

For case 4 the paired mean regret reduction is approximately 0.000246692.
The conditional paired-bootstrap 95% interval is approximately 0.000167 to
0.000336 (2,000 resamples). These are small, dimensionless synthetic-loss
differences, not measured operational or monetary costs. Case 5 shows no
decision or regret improvement from this candidate information.

## Reference validity and controls

The fourth-order finite-difference/DOP853 continuation was compared on
128/256/512 grids and tested against an analytic unforced Cole-Hopf solution.
The largest 256/512 relative QoI difference across cases is about 0.000129.
The reference action/loss screen passes for 1,994/2,000 draws in case 4 and all
2,000 in case 5. Case 4 has one full-grid/reference choice difference, so
389 full-grid disagreements and 388 reference disagreements are distinct.

Cases 0, 6, 7 and 8 fail the stricter absolute expected-loss convergence screen
in all fresh draws despite passing the QoI screen. Their cost estimates remain
provisional. All raw rows retain the reference-stability flag; no blanket claim
of reference validity is made. The screen is an empirical refinement check,
not a certified bound, and the reference shares the initial state and physics.

The maximum restored-gradient mismatch is below 1.2e-16. POD-coordinate change
is below 2.1e-17; mean change below 6e-19. Matched-control energy discrepancies
are below 1e-17 and selected-gradient changes below 5.6e-17. Posterior hashes
are identical across all arms within each case and noise batch.

The full test suite passes: 32 tests. These include legacy binary-boundary
reproduction, exact-gradient restoration with conserved coordinates/mean,
matched control energy, an analytic Cole-Hopf reference-solver convergence
test, and constant-state invariance under feedback.

## Supported interpretation

The selected initial gradient contains some decision-relevant information in
case 4 under this intervention. Supplying it alone is insufficient, especially
in case 5. The result does not show that rank 32 succeeds specifically by
restoring this gradient, nor that any gradient statistic would fail.

The next distinct hypothesis is sensitivity of the controlled final QoI to
distributed pre-control field components. It requires a separately specified
experiment, such as an adjoint-weighted correction with equally budgeted
controls and held-out physical cases. The present target should not be tuned
on these outcomes and then reported as a prespecified success.

This remains an oracle diagnostic using an exact missing scalar, a single
case-generation seed, finite actions and a synthetic loss. Fresh noise draws
are not new independent physical systems.
