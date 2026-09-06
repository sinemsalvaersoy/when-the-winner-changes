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

PINNacle reports means and standard deviations over three runs. Because errors are positive, the audit constructs a moment-matched log-normal distribution for each method and estimates the probability that it would rank first on a fresh run.

This quantity is a predictive sensitivity analysis. It is not a Bayesian posterior, and it cannot recover the raw seed distribution. The approximation is deliberately isolated so it can be replaced when per-seed outputs become available.

## Layer 3: metric dependence

The reported winner is computed independently under L2 relative error, L1 relative error, and maximum pointwise error. A metric-sensitive case is one in which these definitions select different methods.

## Layer 4: collocation-budget dependence

The published ablation comparing PINN and PINN-LRA is re-ranked at 512, 2048, 8192, and 32768 collocation points. This tests whether a method advantage survives a change in how densely the physical domain is constrained.

## Decision rule

A benchmark claim is flagged when at least one of the following holds:

1. The same reported quantity conflicts across paper sections.
2. The most probable fresh-run winner has probability below 0.80.
3. The winning method changes under another reported error definition.
4. The winning method changes with collocation budget.

The thresholds are diagnostic, not universal laws. They make review decisions explicit and auditable.

## Limits

This audit uses published summary statistics and versioned paper source. It cannot identify the correct value behind a reporting conflict, establish causality, or replace reproduction from raw predictions and per-seed outputs.
