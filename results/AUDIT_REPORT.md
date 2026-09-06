# Audit report

## Executive finding

The reported HInv vPINN L2RE is 0.0119 in the main table and 0.456 in the detailed appendix, a 38.3× difference.
This discrepancy reverses the HInv winner for that table and changes which result supports the associated comparison. It is a reporting inconsistency, not evidence that either number is the correct experimental result.

## Audit counts

* 235 shared L2RE claims compared across main and appendix tables
* 2 numerical source conflicts above exact transcription tolerance
* 12 of 22 cases where the conditional log-normal estimate for the most likely fresh-run winner remains below the diagnostic 80% threshold
* 13 of 22 cases where the winning method changes across L2RE, L1RE, and maximum error
* 4 of 4 collocation ablations where the winning method changes with sampling budget

## Interpretation

A leaderboard winner is not automatically a stable property of a method. The audit separates source inconsistency, conditional rerun stability, and metric dependence. These diagnostics support different conclusions and should not be collapsed into one accuracy number.

The rerun analysis uses moment-matched distributions derived from the reported three-run mean and standard deviation. Its probabilities are conditional simulation estimates, not measured frequencies, posterior probabilities, or substitutes for raw seeds. Independence between methods is assumed because run-level pairing is unavailable.

Across log-normal, nonnegative-normal, and gamma models, the identity of the most likely winner is unchanged in 22 of 22 cases. The estimated winning probability can nevertheless move by as much as 21.0%, so probability magnitudes should not be read as distribution-free facts.

## Most fragile fresh-run leaders

| PDE case | Most probable method | Conditional log-normal estimate | Rank entropy |
| --- | --- | ---: | ---: |
| Heat 2d-LT | LAAF | 37.4% | 0.65 |
| Burgers 1d-C | LAAF | 39.2% | 0.58 |
| Wave 2d-CG | vPINN | 43.6% | 0.46 |
| Poisson 2d-MS | LAAF | 49.6% | 0.45 |
| Heat 2d-VC | LRA | 57.4% | 0.37 |
| NS 2d-LT | gPINN | 61.7% | 0.40 |
| Inverse HInv | LRA | 62.5% | 0.29 |
| NS 2d-C | LAAF | 63.0% | 0.29 |

## Distribution-model sensitivity

The full comparison is recorded in `distribution_sensitivity.csv`. These alternative models are robustness checks, not candidates for the true run distribution.
Maximum-error results are unavailable for some methods. Winner comparisons use the methods reported for each metric and should therefore be read together with coverage differences.

## Metric-sensitive cases

| PDE case | Number of winners | Winning methods |
| --- | ---: | --- |
| Heat 2d-MS | 3 | FBPINN, NTK, RAR |
| NS 2d-C | 3 | GAAF, LAAF, vPINN |
| Burgers 1d-C | 2 | LAAF, PINN |
| Burgers 2d-C | 2 | LRA, gPINN |
| Chaotic GS | 2 | FBPINN, MultiAdam |
| Chaotic KS | 2 | LRA, NTK |
| Heat 2d-CG | 2 | LAAF, LRA |
| Heat 2d-LT | 2 | GAAF, PINN |
| Heat 2d-VC | 2 | LRA, NTK |
| NS 2d-LT | 2 | PINN, gPINN |
| Poisson 2d-MS | 2 | LAAF, MultiAdam |
| Poisson 3d-CG | 2 | LRA, vPINN |
| Wave 2d-CG | 2 | FBPINN, GAAF |

## Collocation-budget sensitivity

| PDE case | Winner sequence from 512 to 32768 points |
| --- | --- |
| Burgers1d | PINN-LRA → PINN-LRA → PINN-LRA → PINN |
| GS | PINN-LRA → PINN → PINN-LRA → PINN |
| Heat2d-CG | PINN-LRA → PINN → PINN-LRA → PINN |
| Poisson2d-C | PINN-LRA → PINN-LRA → PINN → PINN-LRA |

## Provenance and limits

All numerical claims are extracted from the versioned arXiv source for PINNacle v2. The project does not claim misconduct or determine which conflicting entry is correct. Definitive resolution requires raw per-seed outputs or confirmation from the benchmark authors.

Source: [PINNacle, arXiv:2306.08827v2](https://arxiv.org/abs/2306.08827)
