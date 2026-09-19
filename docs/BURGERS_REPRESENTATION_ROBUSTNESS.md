# Burgers representation robustness experiment

## Question

Is the ranking reversal in the controlled Burgers pilot a reproducible
representation effect, or a consequence of one seed, modal budget, or spatial
resolution?

## Design

The experiment keeps the Burgers dynamics, data-generating family, train and
test sample counts, learners, and evaluation rules fixed. It varies three axes:

| Axis | Values |
|:--|:--|
| Paired data seed | 11, 17, 23, 29, 31, 37, 41, 43, 47, 53 |
| Spatial resolution | 64, 128, 256 points |
| Modal budget | 2, 4, 8, 16 modes |

Every block compares ridge regression and three nearest neighbours under the
full grid, ordinary L2 POD, and a gradient-weighted POD objective. The grid
prediction is intentionally repeated only to preserve the paired comparison at
each modal budget; these repetitions are not counted as independent evidence.

The two primary outcomes are mean relative L2 error and the circular error in
the location of the steepest negative gradient. Model winner probabilities are
computed across paired seeds. The CSV includes 95 percent percentile-bootstrap
intervals from 2,000 seed resamples and Wilson intervals for the finite number
of observed wins. The Wilson interval remains non-degenerate when all ten
observed seeds select the same winner.

## Result

The original four-mode reversal survives both seed and resolution variation.
For the local physical observable, KNN wins all ten paired seeds on the full
grid at every resolution. Ridge wins after four-mode POD in 10 of 10 seeds at
64 points, 9 of 10 at 128 points, and 10 of 10 at 256 points.

| Grid points | Ridge win probability on grid | Ridge win probability after four-mode POD |
|--:|--:|--:|
| 64 | 0.00 | 1.00 |
| 128 | 0.00 | 0.90 |
| 256 | 0.00 | 1.00 |

The effect is not universal across modal budgets. Ridge wins the local
observable after two-mode POD with probability 0.70, after four-mode POD with
mean probability 0.97 across resolutions, and after eight or sixteen modes
with probability 0.00. At those larger budgets, KNN again wins as it does on
the full grid.

This identifies a bounded mechanism:

\[
\text{ranking drift}
\;=\;
f(\text{representation},\,\text{modal budget},\,\text{observable}),
\]

not a general claim that POD reverses rankings.

Four L2 POD modes retain about 99.39 percent of output variance when averaged
over seeds and resolutions. Eight modes retain about 99.98 percent. The model
ranking changes between these two already high-retention regimes, confirming
that retained variance is not a sufficient diagnostic for preservation of a
local observable or its model ranking.

![Retained variance and observable error](../results/burgers_robustness_variance_observable.png)

![Winner probability](../results/burgers_robustness_winner_probability.png)

## Gradient-aware control

For this smooth, nearly translation-generated dataset, adding a periodic
gradient term to the POD inner product does not materially rotate the leading
subspace. Across all runs, its maximum change relative to L2 POD is below
0.0001 in relative L2 error and below 0.0025 radians in the location metric.

This negative result is informative. A generic gradient penalty is not enough
to preserve an argmin-based local observable when the covariance and derivative
operators select almost the same smooth modes. A more discriminating follow-up
should build the basis against observable sensitivity itself, or introduce
initial conditions with sharper and non-stationary local structure.

## Interpretation boundary

The sweep closes the single-seed and single-resolution objection. It does not
establish a universal law across viscosities, initial-condition families,
learners, or physical observables. The confidence intervals quantify seed
variation conditional on this simulator and experimental design. They are not
uncertainty intervals over all possible Burgers problems.

The full evidence trail is generated with:

```bash
pinn-audit --robustness
```

Raw scores, aggregate errors, winner probabilities, and uncertainty intervals are
stored in `results/burgers_robustness_runs.csv`,
`results/burgers_robustness_summary.csv`, and
`results/burgers_robustness_winner_probabilities.csv`.
