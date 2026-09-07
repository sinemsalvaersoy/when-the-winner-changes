# Paired rerun pilot

## Aim

The pilot tests candidate explanations for winner changes rather than inferring them from aggregate tables. It begins with Burgers1d and Gray Scott because their published collocation ablations show contrasting response shapes while both change winner across budgets.

## Upstream environment

* Codebase: `i207M/PINNacle`
* Pinned commit: `6f8d69c9c84c41d19644e4ceebdc64c4a11d3adf`
* Upstream environment: Python 3.9 with the repository requirements
* Compared methods: vanilla Adam PINN and PINN LRA

The official benchmark exposes seed, method, iteration, and device arguments. The pilot adapter must restrict execution to the selected PDE class without altering the PDE definition, model architecture, optimizer schedule, or evaluation callback.

## Paired design

Every design block is identified by PDE case, initialization seed, collocation draw, and collocation point count. Both methods must be present in every block. A failed run invalidates the paired block until it is rerun successfully.

The smoke stage uses two seeds at 512 points to verify the complete capture path. The evidential stage expands to at least 10 seeds, multiple independent collocation draws, and the four published budgets. Thirty seeds remain the preferred target for stable variance estimates.

## Required outputs

Every run records training residual, residual on an independently sampled holdout grid, L2 relative error on a common evaluation grid, runtime, completion state, device, and exact upstream commit. The holdout grid must never be used for optimization, checkpoint selection, or early stopping.

## Interpretation

Winner switching across paired seeds supports initialization sensitivity. Switching across collocation draws at fixed seed and budget supports sampling sensitivity. A positive residual generalization gap supports collocation overfitting only when it is systematic and associated with the observed rank reversals.

No result from this pilot alone establishes failure to learn invariant physics. That claim requires a preregistered PDE specific invariant diagnostic.
