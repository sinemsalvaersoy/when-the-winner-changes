# Detector response before classification

## Status

Executable synthetic mechanism study, with five prespecified seeds (11, 22, 33, 44, 55) and 32,000 events per class per train/validation/test split. The new generator reinterprets the original generated features and mass as latent phenomenological quantities, then applies an additional response. Results are a new baseline and should not be numerically compared to the first pilot as if only one setting changed.

## Response and paired events

Nominal reconstructed mass is m_latent + 1.5 GeV * z_mass. Varied mass is (1+scale)*m_latent + 1.5 GeV*resolution*z_mass. The proxy features receive independent Gaussian noise of widths [0.10, 0.15, 0.08, 0.08], multiplied by the same resolution factor. Feature x2 additionally shifts by scale*m_latent/30, modeling a shared scale response of a mass-correlated proxy. This is a stipulated toy response, not a detector calibration.

The identical response equations apply to signal and background; response code never uses the class label. Keep the same random draws for each nominal/varied event. This isolates parameter changes from independent resampling fluctuations.

Prespecified variations: scale +/-1%; resolution factor 0.9/1.1. Each variation acts separately. Train both models on nominal reconstructed features only. Choose nominal validation cuts with the same protocol as the first pilot, then freeze model and cut. Never retrain, retune or choose a test threshold on varied samples.

## What is traced

`migrations.csv`: per class and variation, count events entering/exiting score selection; entering/exiting final acceptance including the fixed mass window; and changing mass bin among events retained in both selections. Check selected_varied - selected_nominal = entered - exited.

`templates.json`: record full varied spectrum and two decomposition controls: varied scores with nominal mass (score-only), nominal scores with varied mass (mass-only). They are diagnostics, not additive components: interactions between score and mass movement exist.

`events_*.npz`: paired per-event nominal and varied reconstructed masses, classifier scores, labels and frozen thresholds. Array row number is the event identity within its seed/model. Both models reuse the same test latent events and response draws.

`runs.csv`: varied AUC, signal/background yields and conditional sensitivity assuming that varied background is known exactly. This conditional sensitivity is not systematic uncertainty profiling.

## Detector-derived likelihood nuisance

On the nominal s+b Asimov spectrum, build positive log-linear interpolation of each selected background template between nominal and its supplied detector endpoints. Combine scale and resolution factors multiplicatively, and add a shared 15% normalization nuisance. All parameters have unit Gaussian penalties.

Response parameters are constrained to [-1,1], the tested endpoint range; normalization is bounded at +/-8 standard deviations. There is no response extrapolation. The range restriction is material: these results are not an unrestricted Gaussian response profile. Scale/resolution interactions are not validated by joint response templates. Signal variations are saved but do not enter the background-only numerator of q0; the nominal unrestricted Asimov optimum remains mu=1, theta=0. This is a nominal expected discovery test, not coverage or false-signal validation under shifted truth.

Profiled response may change both shape and acceptance: there is no forced total-yield renormalization. Thus this extends the first pilot's post-selection shape tilt into response-derived selection templates.

## Results

Mean gross background score migration as a percentage of all generated background events:

| Variation | Logistic | Boosted |
| --- | ---: | ---: |
| Scale -1% | 0.543% | 0.262% |
| Scale +1% | 0.569% | 0.269% |
| Resolution -10% | 0.172% | 0.209% |
| Resolution +10% | 0.187% | 0.229% |

Gross migration counts entrants plus exits; it is not the net efficiency change and is not normalized to selected background.

| Nominal expected local Z, averaged over seeds | Logistic | Boosted |
| --- | ---: | ---: |
| Known background | 6.707 | 7.878 |
| 15% normalization profile | 6.182 | 7.075 |
| Normalization and detector response profile | 6.064 | 7.058 |

Boosted retains higher expected sensitivity in this study. Scale-induced score migration is lower for boosted, but resolution-induced migration is higher. There is no universal model robustness ordering from these two perturbations.

One of ten response profile fits reaches a response range boundary; inspect `profiled.csv`. Quoted averages therefore include a boundary-limited result. They must not be presented as calibrated detector performance or completed asymptotic validation.

## Checks and next step

Eleven local tests pass, including identity response, label-independent response, shared-draw scale closure, migration conservation, template endpoint closure, independent scalar likelihood check and nested sensitivity bounds. CI configuration is provided; remote CI has not been run.

Finite template simulation uncertainties are not fitted. Latent feature laws are synthetic; mass smearing changes edge acceptance. The present Gaussian response model omits non-Gaussian tails and correlated reconstruction failures. Further physical development should specify a real detector observable and calibrated response variation, validate combined nuisance templates, and test background-only false-signal rates with pseudoexperiments. These are open questions, not completed claims.

## Run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python detector_pilot.py
python -m unittest discover -s tests -v
```
