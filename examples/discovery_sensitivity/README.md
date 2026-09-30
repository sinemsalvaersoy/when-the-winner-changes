# Better classifier, stronger discovery?

Dr. Sinem Salva Ersoy | Second Mind Method

An executable **synthetic resonance search**, examining the relationship between ROC AUC and expected local discovery sensitivity. It is a mechanism pilot, not a collider result or a claim of a new statistical method.

## Physics question

Does the classifier preferred by global ROC AUC also produce the stronger selected mass spectrum under a specified profile likelihood? How does that conclusion depend on continuum normalization and shape constraints?

## Fixed protocol

- Illustrative invariant mass range: 80 to 170 GeV; 15 fixed bins.
- Gaussian signal centered at 125 GeV with 3 GeV width; smooth exponential continuum.
- Before classifier selection: expected signal yield 180, background yield 12,000.
- Four synthetic reconstructed features. The mass is excluded from classifier inputs. One feature is correlated with continuum mass, making selection effects on the mass distribution observable. These features are phenomenological, not detector calibrated.
- Logistic regression and histogram gradient boosting see the same four features, training events and balanced training labels. Hyperparameters are fixed. Their computational budgets are not equal.
- Independent train, validation and test simulations for every seed, with 16,000 generated events per class per split. Equal generation counts do not define the physical signal/background prior.
- Each model chooses its cut from the same 17 signal quantile menu using nominal validation sensitivity only. Require at least ten expected background events in every validation mass bin. Freeze the cut for test evaluation and all nuisance scenarios.
- AUC is evaluated over the full test sample; physical sensitivity uses the selected binned mass spectrum. They intentionally answer different questions.

## Likelihood and nuisance assumptions

For nominal Asimov counts n_i = s_i + b_i, evaluate the background-only profile likelihood against the known global optimum at mu=1 and theta=0. The statistic is

q0 = min_theta { 2 sum_i [b_i(theta) - n_i + n_i log(n_i/b_i(theta))] + theta_norm^2 + theta_shape^2 }.

Expected local asymptotic Z = sqrt(q0). This is not observed significance and does not include a mass scan or a look-elsewhere correction.

Normalization: multiply the background by exp(theta_norm log(1+delta)). Shape: apply exp(theta_shape tau (m_i-125)/45) and renormalize to preserve selected total yield. Both nuisance parameters have unit Gaussian constraints and are shared across the mass bins. Set a strength to zero to remove its parameter.

The same nuisance definition and strengths are used for both classifiers. These are **post-selection phenomenological template deformations**, not detector variations propagated through feature reconstruction or classifier migration. That propagation is a necessary later extension. The present study evaluates median sensitivity assuming the nuisance model is correct; it does not establish robustness to misspecification.

Normalization strengths: 0, 0.05, 0.15, 0.30. Shape tilt strengths: 0, 0.15, 0.35, 0.60. No scan setting or seed is selected after looking at a reversal.

## Run

Python 3.11 or newer; Python 3.12 recommended.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python pilot.py
```

Outputs: individual runs, selected signal/background templates, scenario summaries, figure and machine readable report in `results/`. The checked-in outputs were generated with seeds 11, 22, 33, 44 and 55.

## Interpretation limits

Five seeds assess simulation and training variability; their standard deviation is not an experimental uncertainty or a bootstrap confidence interval. Templates currently have no finite Monte Carlo nuisance parameters. Signal shape and acceptance uncertainties are absent. Nuisance shapes are deliberately narrow and prescribed. AUC/sensitivity disagreement is not, by itself, evidence of an invalid benchmark. Alternative score-bin likelihoods, independent threshold protocols, nuisance misspecification studies, larger simulation samples and toy calibration of asymptotic Z are useful extensions. No ranking reversal is required for success.

This pilot is distinct from the Burgers representation pilot: it examines a classifier selection followed by a physical likelihood, rather than representation-dependent prediction errors.

## Statistical reference

Cowan, Cranmer, Gross and Vitells, *Asymptotic formulae for likelihood-based tests of new physics*, https://arxiv.org/abs/1007.1727. The likelihood here is intentionally simple and is implemented directly with SciPy; it is not a pyhf result.

## Event-level detector extension

See `DETECTOR_RESPONSE.md` for the paired response experiment, mechanism decomposition, numerical findings and explicit profile-range limitations. Run `python detector_pilot.py` to reproduce `results/detector/`. This is a separate nominal baseline with detector smearing and larger simulation samples.
