# First run: a negative result for ranking reversal

Five prespecified seeds and 16 prespecified nuisance scenarios produced **no AUC versus expected sensitivity ranking reversal**. Boosted AUC averaged 0.90557; logistic AUC averaged 0.84644. Boosting also retained higher expected local sensitivity in every seed/scenario.

This is evidence only within this synthetic generator, chosen models, fixed cuts and nuisance family. It does not show that AUC generally predicts physics sensitivity. No model, seed or nuisance grid was retuned to produce a reversal.

Next scientific extension: generate physically motivated nuisance variations before classification, propagate selection migrations, and test profile likelihood coverage or false-positive calibration under deliberate misspecification. Do not merely increase arbitrary nuisance strengths to chase a winner change.
