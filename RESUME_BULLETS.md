A/B Testing & Uplift Modeling | Python · DuckDB · SciPy · statsmodels · LightGBM · Streamlit

- Diagnosed pooled-test confounding in a 13.98M-user ad trial that a passing SRM check missed (treatment AUC 0.511, Hausman z = 40); doubly robust AIPW cut reported visit lift from 1.03 to 0.74 pp [1]
- Shrank visit-lift confidence intervals 17% (31% fewer users for equal precision) with cross-fitted LightGBM CUPAC, holding the A/A false-positive rate at 4.9% across 10,000 simulations [2]
- Showed peeking at 1,000 interim looks inflates A/A false positives to 65.6% and built an mSPRT sequential test that held them at 2.3% over 5,000 simulations; power formula matched simulation within 1.1 pts [3]
- Benchmarked S/T/X/transformed-outcome LightGBM uplift learners on 5.6M held-out users: top-30% targeting captured 86% of incremental visits vs 30% at random, on par with a response-model baseline (85%) [5]

**Alternates**

- Built abkit, a tested Python experimentation library (SRM, Welch/bootstrap, power/MDE, CUPED/CUPAC, AIPW, mSPRT) validated against SciPy/statsmodels, streaming 13.98M rows at 2.0 GB peak RAM [4]
- Built a DuckDB/Parquet pipeline computing exact binomial bootstraps and covariate-adjusted ITTs for 13.98M users in one streamed pass, with 59 pytest checks vs SciPy/statsmodels/scikit-uplift [4]
- Ranked users by LightGBM uplift scores chosen on validation, raising the top-10% share of incremental visits captured from 50% (response model) to 60% on a 5.6M-user test split, with Poisson-bootstrap CIs [5]

**Footnotes** (all files under `reports/`)

1. `readout.json`: `balance.treatment_predictability_auc.auc` = 0.5106 (CI 0.5090-0.5122); `adjusted.visit.aipw.hausman_vs_raw.z` = 39.9 (35.2-45.3 across adjusted estimators); `adjusted.visit.raw.abs_effect` = 1.034 pp vs `adjusted.visit.aipw.abs_effect` = 0.744 pp; `srm.p_value` = 0.999; `data_summary.json: n_rows` = 13,979,592.
2. `readout.json`: `adjusted.visit.cupac_plus_lin.ci_width_reduction` = 0.170, `.sample_size_saving` = 0.311; `simulations.json`: `aa.false_positive_rate.visit_cupac.rate` = 0.0491, `aa.n_sims` = 10,000 (100,000 users per sim, resampled real control users).
3. `simulations.json`: `peeking.aa_false_positive_rate_naive_by_looks.1000.rate` = 0.656, `peeking.aa_false_positive_rate_msprt.rate` = 0.0234, `peeking.n_sims` = 5,000 (200,000 users each); `power.max_abs_gap_raw` = 0.0106.
4. `readout.json: peak_ram_gb` = 2.04 (2.09 on the first run), `scope`; `uv run pytest`: 59 passed in 46.8 s (local, CPU).
5. `uplift.json`: `visit.chosen_model` = transformed outcome; `visit.test_bootstrap.best_capture_at_30` = 0.863 (0.842-0.878) vs `response_capture_at_30` = 0.852 (0.836-0.866); `visit.test.random.capture_at_30` = 0.302; `visit.test.transformed_outcome.capture_at_10` = 0.595 vs `visit.test.response_model.capture_at_10` = 0.497 (point estimates, no bootstrap at 10%); test n = 5,591,832. For conversion the response model is ahead (0.921 vs 0.905 at 30%), so no conversion claim is made.
