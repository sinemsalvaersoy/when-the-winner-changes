# Methodology

## Audit object

The object of this audit is not the PINN architecture itself. It is the chain that turns repeated numerical experiments into a benchmark claim:

\[
\text{runs} \rightarrow \text{summary statistic} \rightarrow \text{metric} \rightarrow \text{rank} \rightarrow \text{scientific claim}
\]

The project tests whether information survives each crossing.

## Layer 1: source consistency

Values describing the same method, PDE case, and metric are joined across the main result table and detailed appendix. Any nonzero numerical difference is surfaced. Differences are reported, not silently reconciled.

## Layer 2: rerun stability

PINNacle reports means and standard deviations over three runs. Because errors are positive, the primary audit constructs a moment-matched log-normal distribution for each method and simulates the probability that it would rank first on a fresh run. It assumes independence between methods because paired run-level results are unavailable.

This is a conditional predictive sensitivity analysis. It is not a measured frequency, a Bayesian posterior, or a confidence interval, and it cannot recover the raw seed distribution from three summary values. To expose model dependence, the audit repeats the calculation with moment-matched gamma and nonnegative-normal alternatives. Agreement in winner identity is evidence of robustness to these three choices only. The probability range records sensitivity of the magnitude.

## Layer 3: metric dependence

The reported winner is computed independently under L2 relative error, L1 relative error, and maximum pointwise error. A metric-sensitive case is one in which these definitions select different methods. This is not classified as an error: the metrics encode different scientific priorities, and maximum-error coverage is not identical for every method.

## Layer 4: collocation-budget dependence

The published ablation comparing PINN and PINN-LRA is re-ranked at 512, 2048, 8192, and 32768 collocation points. This tests whether a method advantage survives a change in how densely the physical domain is constrained. A switch is evidence of budget dependence, not evidence that either method fails.

## Decision rule

A benchmark claim is flagged when at least one of the following holds:

1. The same reported quantity conflicts across paper sections.
2. The most probable fresh-run winner has conditional log-normal probability below the diagnostic threshold of 0.80.
3. The winning method changes under another reported error definition.
4. The winning method changes with collocation budget.

The thresholds are diagnostic, not universal laws. They make review decisions explicit and auditable.

## Limits

This audit uses published summary statistics and versioned paper source. It cannot identify the correct value behind a reporting conflict, establish causality, estimate correlations between methods, or replace reproduction from raw predictions and per-seed outputs. Its 0.80 threshold is a transparent review convention rather than a universal scientific standard.
