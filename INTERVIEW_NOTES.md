# Interview notes: uplift-lab

Every number here comes from `reports/*.json` (file and key in brackets).

## Design choices and why

**Sufficient statistics everywhere.** Every estimator in `abkit` (Welch, Lin, CUPAC, AIPW) reduces to
per-arm counts, means and covariances, and `MomentAccumulator` builds those from streamed batches.
The 13.98M-row readout is therefore one pass over Parquet at ~2 GB peak RAM
[`readout.json: peak_ram_gb`], and the same code is unit-tested on 3,000-row arrays against
statsmodels. A separate "big data" code path would have needed its own validation.

**Exact bootstrap from counts.** Resampling n Bernoulli rows with replacement gives a
Binomial(n, p̂) success count, so the full-population bootstrap (10,000 reps) is two binomial draws
per replicate rather than 14M-row resamples. It is identical in distribution to the row bootstrap,
and a test checks it against the row-resampling path.

**Lin (2013) interacted adjustment, not plain ANCOVA.** The fully interacted version is never less
efficient than the difference in means, even when slopes differ by arm. The variance includes the
super-population term `(b1-b0)' S_x (b1-b0) / N`, so the adjusted CI is directly comparable to Welch,
and a 300-experiment coverage test checks both at ~95%.

**CUPAC with cross-fitting.** No pre-period metric exists, so the covariate is an ML prediction of
the outcome from the 12 pre-treatment features. Each row is scored by a model trained only on the
other fold (bucket parity). A unit test perturbs one unit's outcome and asserts that unit's
prediction does not change. Treatment is never a model input, and `check_pretreatment` rejects
`exposure`/`visit`/`conversion`/`treatment` as covariates.

**AIPW as the estimate of record.** I planned CUPAC as the headline. The balance check showed
assignment is only *conditionally* random (pooled tests), and in that setting regression
adjustment is consistent only if the outcome model is right. AIPW is consistent if either the
outcome models or the propensity model is right, so it is the estimate of record. CUPAC+Lin remains
the headline for variance reduction.

**mSPRT for continuous monitoring.** A closed-form normal-mixture likelihood ratio gives
always-valid p-values and confidence sequences with one formula. Group-sequential alpha spending
would need pre-planned look times, which a dashboard user won't respect. The mixing variance τ² is
set to the square of the fixed-horizon MDE.

**Transformed outcome instead of the 50/50 class-transformation trick.** Jaskowski-Jaroszewicz
class transformation assumes P(T=1) = 0.5. Criteo is 85/15, so I used the Athey-Imbens
transformed outcome `z = y (t-e) / (e(1-e))`, which is valid for any known propensity.

**Response-model baseline for uplift.** "Target the users most likely to visit" is what a team
without uplift modelling would do, so it is a more honest baseline than random. It is a strong one
here, because visit uplift scales with baseline visit propensity.

## What didn't work / surprises

* **My first readout was wrong in an instructive way.** Adjusted ITTs came out 25-36% below the raw
  ITT, which is impossible under clean randomization (Hausman z ≈ 40)
  [`readout.json: adjusted.visit.*.hausman_vs_raw.z`]. I first suspected a bug, but statsmodels
  agreement ruled it out. The cause: the treatment-predictability AUC is 0.511 with a CI excluding
  0.5 [`balance.treatment_predictability_auc`], and the CSV is stored as 13 single-arm blocks with
  visit rates of 0.0%-8.6% [`file_blocks`]. The file pools several incrementality tests with
  different treatment ratios.
* **A decile table almost fooled me.** `ntile(f3)` showed deciles that were 100% treated. f3 has
  one value covering 11.4M rows, and ntile broke the ties by file order, which is sorted by block.
  Lesson: never bin a heavily tied feature with ntile, and check distinct-value counts first.
* **The first AIPW unit test failed for a real reason.** With a noisy proxy of the test id, AIPW
  removes only part of the bias. I kept that as a separate test (`test_noisy_proxy_leaves_residual_confounding`)
  because it is exactly the residual-confounding caveat on Criteo.
* **The bootstrap is slightly liberal on rare conversions.** A/A FPR is 5.52% (CI 5.09-5.98%) for
  the percentile bootstrap vs 5.37% for Welch [`simulations.json: aa.false_positive_rate`].
  Percentile intervals undercover with ~30 control conversions per sim.
* **mSPRT is conservative and costs power.** A/A FPR is 2.34% (target ≤5%). With a true lift equal
  to the fixed-horizon MDE, mSPRT power by the planned horizon is 55.9%, vs 79.8% for the
  fixed-horizon test [`peeking.ab_power_*`]. That is the price of being allowed to peek.

## Known weaknesses

* **Residual confounding.** Test ids are not released. Adjusted visit estimates span 0.67-0.77 pp,
  wider than their own CIs, so the size of the raw overstatement is model dependent. Its direction
  is not.
* **CACE uses an approximation.** The adjusted CACE omits the ITT/exposure covariance term, and the
  exclusion restriction (assignment affects outcomes only through exposure) is assumed.
* **Uplift evaluation sits on a weakly confounded test set.** I report IPW-weighted metrics
  alongside the plain ones. The learners themselves use a constant propensity.
* **Small hyperparameter search.** Two LightGBM configs per learner, chosen on validation Qini, and
  the train split is a 50% sample (RAM and time on a shared laptop).
* **Variance formula.** Lin/CUPAC SEs use the asymptotic (Neyman / influence-function) formula, not
  HC2. Tests show they agree within 2% at n = 3,000.
* **CI is not yet run.** `.github/workflows/ci.yml` has never run on GitHub (no remote exists).
  Locally, ruff and pytest pass.

## 10 likely questions

1. **The SRM passed. Why didn't you trust the randomization?** SRM only tests arm *counts*.
   Balance tests whether assignment depends on covariates. Here every |SMD| is < 0.1, yet at n = 14M
   the largest is 65 SEs from zero, a classifier predicts treatment at AUC 0.511, and adjustment
   moves the estimate by ~40 SEs. The share being 0.8500001 also suggests the ratio was fixed when
   the file was built.
2. **Isn't it cheating to "pick" the adjusted estimate because it's different?** It would be if I
   had searched over specifications for a preferred answer. I instead used diagnostics that exist
   for this (Hausman test, propensity AUC, post-stratification) and a doubly-robust estimator chosen
   before looking at its value. All adjusted estimators move the same way, and the A/A sims show
   the Hausman check rejects ~5% under clean randomization.
3. **How does CUPAC avoid leakage when there is no pre-period?** Through cross-fitting. Row i's
   covariate comes from a model that never saw row i, and the model uses only pre-treatment
   features, so with respect to unit i it is a fixed function of X, which is independent of T. A/A
   FPR with CUPAC is 4.91% over 10,000 sims.
4. **CI width fell 17% but you say 31% of the sample was saved. Which is it?** Both. Width scales
   with SE, and n scales with SE². (1 − 0.170)² ≈ 0.69, so equal precision needs ~31% fewer users.
5. **Why is the conversion gain so much smaller?** R² is lower (0.13 vs 0.33 out of fold). Rare
   outcomes are mostly noise at the individual level, and variance reduction is ≈ R².
6. **Explain why peeking inflates the false-positive rate.** Each look is another chance for a
   random walk to cross ±1.96σ. With 1,000 looks the A/A FPR reached 65.6%. The mSPRT bounds the
   probability that a test martingale *ever* exceeds 1/α (Ville's inequality), so stopping at any
   time is valid.
7. **When would you not use mSPRT?** When you will read the result once at a pre-registered
   horizon. Fixed-horizon has more power (79.8% vs 55.9% at the same n here). Group-sequential
   designs are a middle ground for a few planned looks.
8. **How did you validate the power calculator?** I injected a +10% relative lift into resampled
   real control users and ran 2,000 sims at each of 4 sample sizes. Analytic and simulated power
   differ by at most 1.1 points for Welch and 0.5 points for CUPAC (using the readout R² as the
   input).
9. **Qini AUC vs uplift@k: which would you show a marketer?** Captured share of incremental
   outcomes at the budget they actually have (e.g. top 30%). Qini AUC is for model selection.
10. **What would you do with the test id?** Stratify exactly (post-stratification or test fixed
    effects), report per-test effects and their heterogeneity, and pool with precision weights.
    That would remove the residual-confounding caveat.
