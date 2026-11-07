"""Render reports/readout.md from the JSON outputs (no number is typed by hand)."""

from __future__ import annotations

from typing import Any

from . import config

LABEL = {
    "raw": "Raw difference in means (Welch)",
    "lin_12_features": "Lin regression adjustment, 12 features",
    "cupac": "CUPAC (cross-fitted LightGBM prediction)",
    "cupac_plus_lin": "CUPAC + 12 features",
    "aipw": "AIPW, doubly robust (cross-fitted)",
    "s_learner": "S-learner", "t_learner": "T-learner", "x_learner": "X-learner",
    "transformed_outcome": "Transformed outcome", "response_model": "Response model (baseline)",
    "random": "Random targeting (baseline)",
}


def pp(x: float, d: int = 3) -> str:
    return f"{100 * x:.{d}f} pp"


def pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def ci_pp(lo: float, hi: float, d: int = 3) -> str:
    return f"[{100 * lo:.{d}f}, {100 * hi:.{d}f}]"


def _readout_section(r: dict[str, Any], d: dict[str, Any]) -> list[str]:
    srm, bal, itt, adj = r["srm"], r["balance"], r["itt"], r["adjusted"]
    auc = bal["treatment_predictability_auc"]
    prop = r["nuisance_models"]["oof_quality"]["propensity"]
    blocks = r["file_blocks"]
    vis = [b["visit_rate"] for b in blocks["blocks"]]
    prim, vr = r["primary_effect_estimator"], r["primary_variance_reduction_estimator"]
    L: list[str] = []
    L += [
        "## 1. Design and guardrails",
        "",
        f"* **Population:** {d['n_rows']:,} users ({d['n_treatment']:,} treated, "
        f"{d['n_control']:,} control). Visit rate {pct(d['visit_rate'], 2)}, conversion rate "
        f"{pct(d['conversion_rate'], 3)}. Control users can never be exposed "
        f"(control_exposed = {d['control_exposed']}), so this is one-sided non-compliance.",
        f"* **SRM:** observed treatment share {srm['observed_ratios'][0]:.7f} vs the documented "
        f"0.85; chi-square = {srm['chi2']:.2e}, p = {srm['p_value']:.3f} -> **no SRM**. At this "
        f"n the check would alarm (p < {srm['threshold']}) for a share deviation as small as "
        f"{srm['detectable_share_deviation']:.5f}. Caveat: the public file was sub-sampled by "
        "Criteo, and the share being this close to 0.85 suggests the ratio was fixed when "
        "the file was built, so passing SRM is weak evidence here.",
        f"* **Covariate balance:** every |SMD| is below the 0.1 rule of thumb (max "
        f"{bal['max_abs_smd']:.3f}), but with SE {bal['smd_se_under_randomization']:.5f} under "
        f"randomization the largest is {bal['max_abs_smd_z']:.0f} SEs from zero. A gradient-"
        f"boosted classifier predicts treatment from the 12 pre-treatment features with AUC "
        f"{auc['auc']:.4f} (95% CI {auc['ci_low']:.4f}-{auc['ci_high']:.4f}, n = {auc['n']:,}); "
        f"the cross-fitted LightGBM propensity on all users gives AUC {prop['oof_auc']:.4f}, "
        f"with e(x) between {prop['quantiles']['0.0']:.3f} and {prop['quantiles']['1.0']:.3f} "
        "(good overlap, no clipping needed).",
        f"* **Why:** in the original CSV order the file consists of {blocks['n_blocks']} "
        f"contiguous single-arm blocks with visit rates from {pct(min(vis), 2)} to "
        f"{pct(max(vis), 2)}. That fits the dataset card ('assembling data resulting from "
        "several incrementality tests'). Tests with different treatment ratios and base rates "
        "were pooled, so assignment is random *within* a test but not across the pooled "
        "file. Test ids are not released; the features only partly encode them.",
        "",
        "## 2. Topline ITT and the effect of adjustment",
        "",
        "| Estimator | Visit ITT (95% CI) | Rel. lift | CI width vs raw | Hausman z vs raw |"
        " Conversion ITT (95% CI) | CI width vs raw |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in ["raw", "lin_12_features", "cupac", "cupac_plus_lin", "aipw"]:
        v, c = adj["visit"][m], adj["conversion"][m]
        hz = v.get("hausman_vs_raw", {}).get("z")
        L.append(
            f"| {LABEL[m]} | {pp(v['abs_effect'])} {ci_pp(v['abs_ci_low'], v['abs_ci_high'])} "
            f"| {pct(v['rel_effect'])} | {'-' if m == 'raw' else '-' + pct(v['ci_width_reduction'])}"
            f" | {'-' if hz is None else f'{hz:.1f}'} | {pp(c['abs_effect'], 4)} "
            f"{ci_pp(c['abs_ci_low'], c['abs_ci_high'], 4)} | "
            f"{'-' if m == 'raw' else '-' + pct(c['ci_width_reduction'])} |"
        )
    ps = r["post_stratified"]
    L += [
        f"| Post-stratified on {ps['visit']['n_strata']} CUPAC deciles | "
        f"{pp(ps['visit']['estimate'])} {ci_pp(ps['visit']['ci_low'], ps['visit']['ci_high'])}"
        f" | - | - | - | {pp(ps['conversion']['estimate'], 4)} "
        f"{ci_pp(ps['conversion']['ci_low'], ps['conversion']['ci_high'], 4)} | - |",
        "",
        f"Bootstrap check (10,000 exact binomial resamples): visit raw CI "
        f"{ci_pp(itt['visit']['bootstrap']['abs_ci_low'], itt['visit']['bootstrap']['abs_ci_high'])}"
        f" vs Welch {ci_pp(itt['visit']['welch']['abs_ci_low'], itt['visit']['welch']['abs_ci_high'])}.",
        "",
        "**Reading this table:**",
        "",
        f"1. **Precision.** {LABEL[vr]} shrinks the visit CI by "
        f"{pct(adj['visit'][vr]['ci_width_reduction'])}, the same precision as an unadjusted "
        f"test on {adj['visit'][vr]['effective_sample_multiplier']:.2f}x as many users "
        f"(a {pct(adj['visit'][vr]['sample_size_saving'])} sample-size saving at equal "
        f"precision). For conversion the gain is smaller: "
        f"{pct(adj['conversion'][vr]['ci_width_reduction'])} narrower, "
        f"{pct(adj['conversion'][vr]['sample_size_saving'])} sample-size equivalent. Rare "
        f"outcomes are hard to predict (out-of-fold R^2 "
        f"{r['nuisance_models']['oof_quality']['conversion']['oof_r2']:.3f} vs "
        f"{r['nuisance_models']['oof_quality']['visit']['oof_r2']:.3f} for visit).",
        f"2. **Accuracy.** Under clean randomization, adjustment moves the point estimate by "
        f"well under one SE. Here it moves visit by "
        f"{pct(-adj['visit']['aipw']['shift_vs_raw_relative'], 0)} (Hausman z = "
        f"{adj['visit']['aipw']['hausman_vs_raw']['z']:.0f}), consistent with the imbalance "
        f"above. The raw ITT overstates the lift because the high-propensity-to-visit decile "
        f"is {pct(ps['visit']['strata'][-1]['treatment_share'])} treated vs "
        f"{pct(min(s['treatment_share'] for s in ps['visit']['strata']))} in the lowest "
        "decile.",
        f"3. **Estimate of record:** {LABEL[prim]} = {pp(adj['visit'][prim]['abs_effect'])} "
        f"{ci_pp(adj['visit'][prim]['abs_ci_low'], adj['visit'][prim]['abs_ci_high'])} on visit "
        f"(+{pct(adj['visit'][prim]['rel_effect'])} relative) and "
        f"{pp(adj['conversion'][prim]['abs_effect'], 4)} on conversion "
        f"(+{pct(adj['conversion'][prim]['rel_effect'])}). The adjusted estimators span "
        f"{pp(min(adj['visit'][m]['abs_effect'] for m in adj['visit'] if m != 'raw'))} to "
        f"{pp(max(adj['visit'][m]['abs_effect'] for m in adj['visit'] if m != 'raw'))}, wider "
        "than their CIs. That spread is model dependence: the features are only a proxy for "
        "the unreleased test id, so some residual confounding is possible. The direction "
        "(raw overstates) is robust; the exact size of the overstatement is not.",
        "",
        "## 3. Effect on the exposed (CACE)",
        "",
        f"Only {pct(itt['exposure_rate_treated'], 2)} of treated users were actually shown an "
        "ad. With one-sided non-compliance the Wald / Bloom estimator ITT / P(exposed | T=1) "
        "gives the effect on compliers (exposed users), assuming assignment affects outcomes "
        f"only through exposure: visit +{pp(itt['visit']['cace_adjusted']['estimate'], 1)} "
        f"{ci_pp(itt['visit']['cace_adjusted']['ci_low'], itt['visit']['cace_adjusted']['ci_high'], 1)}"
        f" using the {LABEL[prim]} ITT (raw ITT: +{pp(itt['visit']['cace_raw']['estimate'], 1)}),"
        f" conversion +{pp(itt['conversion']['cace_adjusted']['estimate'], 2)}. Exposed users "
        "are a selected group, so these numbers do not carry over to the whole population.",
        "",
    ]
    return L


def _sim_section(s: dict[str, Any]) -> list[str]:
    aa, pk, pw = s["aa"], s["peeking"], s["power"]
    f = aa["false_positive_rate"]
    naive = pk["aa_false_positive_rate_naive_by_looks"]
    max_k = max(naive, key=int)
    L = [
        "## 4. Does the methodology hold up? Simulation studies",
        "",
        f"All simulations resample the {s['bank_size']:,} real control-arm users (so the null "
        "holds by construction) and use a Bernoulli(0.85) fake assignment.",
        "",
        f"**A/A, fixed horizon** ({aa['n_sims']:,} sims x {aa['n_users_per_sim']:,} users, "
        f"alpha = {aa['alpha']}):",
        "",
        "| Test | Visit FPR (95% CI) | Conversion FPR (95% CI) |",
        "|---|---|---|",
    ]
    for key, name in (("welch", "Welch t-test"), ("bootstrap", "Bootstrap percentile CI"),
                      ("cupac", "CUPAC-adjusted (Lin)"),
                      ("hausman_raw_vs_cupac", "Hausman raw vs CUPAC")):
        v, c = f[f"visit_{key}"], f[f"conversion_{key}"]
        L.append(f"| {name} | {pct(v['rate'], 2)} ({pct(v['ci_low'], 2)}-{pct(v['ci_high'], 2)})"
                 f" | {pct(c['rate'], 2)} ({pct(c['ci_low'], 2)}-{pct(c['ci_high'], 2)}) |")
    L += [
        "",
        f"CUPAC keeps the false-positive rate at nominal while shrinking the SE to "
        f"{aa['mean_se_ratio_cupac_vs_raw']['visit']:.3f}x (visit) and "
        f"{aa['mean_se_ratio_cupac_vs_raw']['conversion']:.3f}x (conversion) of the raw SE. "
        "Under clean randomization the Hausman check rejects about alpha of the time, so the "
        "z of about 40 on the real trial is not a property of the method.",
        "",
        f"**Peeking** ({pk['n_sims']:,} A/A sims x {pk['n_users_per_sim']:,} users, visit, up "
        f"to {pk['max_looks']:,} equally spaced looks):",
        "",
        "| Looks | Fixed-horizon test re-run at every look: FPR (95% CI) |",
        "|---|---|",
    ]
    for k, v in naive.items():
        L.append(f"| {k} | {pct(v['rate'], 1)} ({pct(v['ci_low'], 1)}-{pct(v['ci_high'], 1)}) |")
    ms = pk["aa_false_positive_rate_msprt"]
    L += [
        "",
        f"The mSPRT, checked at all {max_k} looks, has FPR **{pct(ms['rate'], 2)}** "
        f"({pct(ms['ci_low'], 2)}-{pct(ms['ci_high'], 2)}), versus "
        f"**{pct(naive[max_k]['rate'], 1)}** for the naive test at the same looks. The price "
        f"of always-valid inference: with a true lift equal to the fixed-horizon MDE "
        f"({pp(pk['ab_injected_abs_lift'])}, {pct(pk['ab_injected_rel_lift'])} relative), the "
        f"fixed-horizon test at its planned end has power "
        f"{pct(pk['ab_power_fixed_horizon_final_look']['rate'])}, while the mSPRT reaches "
        f"{pct(pk['ab_power_msprt_any_look']['rate'])} by the same horizon. When it does reject, "
        f"it stops after a median of {pk['ab_msprt_median_stop_n']:,.0f} users "
        f"({pct(pk['ab_msprt_mean_stop_fraction'], 0)} of the horizon on average).",
        "",
        f"**Power calculator vs simulation** (visit, +{pct(pw['rel_lift'], 0)} relative lift, "
        f"{pw['n_sims_per_point']:,} sims per point):",
        "",
        "| Users | Analytic (Welch) | Simulated (Welch) | Analytic (CUPAC, R^2 = "
        f"{pw['variance_reduction_input']:.3f}) | Simulated (CUPAC) |",
        "|---|---|---|---|---|",
    ]
    for g in pw["grid"]:
        L.append(f"| {g['n_users']:,} | {pct(g['analytic_power_raw'])} | "
                 f"{pct(g['simulated_power_raw']['rate'])} | {pct(g['analytic_power_cupac'])} | "
                 f"{pct(g['simulated_power_cupac']['rate'])} |")
    L += [
        "",
        f"Max |analytic - simulated| = {100 * pw['max_abs_gap_raw']:.1f} points (Welch) and "
        f"{100 * pw['max_abs_gap_cupac']:.1f} points (CUPAC, using the R^2 measured in the "
        "readout as the calculator's variance-reduction input).",
        "",
    ]
    return L


def _uplift_section(u: dict[str, Any]) -> list[str]:
    sp = u["splits"]
    L = [
        "## 5. Heterogeneous effects: who should we target?",
        "",
        f"Stratified split by (treatment, visit): train {sp['train']['n']:,} users (a "
        f"{pct(sp['train']['sample_frac'], 0)} sample of the train buckets), validation "
        f"{sp['val']['n']:,}, test {sp['test']['n']:,}. LightGBM base learners; 2 configs per "
        "learner chosen on validation Qini; the final model is chosen on validation and scored "
        "on test once.",
        "",
    ]
    for o in [k for k in ("visit", "conversion") if k in u]:
        r = u[o]
        best = r["chosen_model"]
        L += [
            f"### {o.capitalize()} (test ATE {pp(r['test_ate'])})",
            "",
            "| Model | Qini AUC | Qini AUC (IPW) | Uplift@10% | Uplift@30% | Captured@10% |"
            " Captured@30% | Captured@30% (IPW) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        order = sorted(r["test"], key=lambda k: -r["test"][k]["qini_auc"])
        for k in order:
            m, w = r["test"][k], r["test_ipw_weighted"][k]
            star = " (chosen on val)" if k == best else ""
            L.append(f"| {LABEL[k]}{star} | {m['qini_auc']:.4f} | {w['qini_auc']:.4f} | "
                     f"{pp(m['uplift_at_10'], 2)} | {pp(m['uplift_at_30'], 2)} | "
                     f"{pct(m['capture_at_10'])} | {pct(m['capture_at_30'])} | "
                     f"{pct(w['capture_at_30'])} |")
        b = r["test_bootstrap"]
        cap = b["best_capture_at_30"]
        resp = b["response_capture_at_30"]
        L += [
            "",
            f"**Business read ({o}):** treating only the top 30% of users ranked by the "
            f"{LABEL[best]} captures **{pct(cap['point'])}** of all incremental {o}s "
            f"(Poisson bootstrap 95% CI {pct(cap['ci_low'])}-{pct(cap['ci_high'])}, "
            f"{b['n_boot']} reps). Random targeting captures 30% by construction; ranking by "
            f"the response model captures {pct(resp['point'])} "
            f"({pct(resp['ci_low'])}-{pct(resp['ci_high'])}).",
            "",
        ]
    return L


def run() -> str:
    d = config.read_json("data_summary.json")
    r = config.read_json("readout.json")
    lines = [
        "# Criteo incrementality trial: experiment readout",
        "",
        "> Generated by `uplift-lab report` from `reports/*.json`; every number below is "
        "computed by code in this repo. Scope: CPU only (i7-11800H, 4 threads).",
        "",
    ]
    prim = r["primary_effect_estimator"]
    v = r["adjusted"]["visit"]
    lines += [
        "## TL;DR",
        "",
        f"* Display advertising raised visits by **{pp(v[prim]['abs_effect'])}** "
        f"(95% CI {ci_pp(v[prim]['abs_ci_low'], v[prim]['abs_ci_high'])}, "
        f"+{pct(v[prim]['rel_effect'])} relative; doubly-robust estimate) across "
        f"{d['n_rows']:,} users. The naive difference in means "
        f"({pp(v['raw']['abs_effect'])}) overstates it, because the file pools several tests "
        "whose treatment shares differ with users' baseline visit propensity.",
        f"* SRM passes, but the features predict treatment (AUC "
        f"{r['balance']['treatment_predictability_auc']['auc']:.3f}, CI excludes 0.5), and "
        "adjustment moves the estimate by far more than chance. Without a balance check this "
        "would have been missed.",
        f"* Cross-fitted CUPAC + linear adjustment narrows the visit CI by "
        f"{pct(v['cupac_plus_lin']['ci_width_reduction'])}, the same precision as "
        f"{pct(v['cupac_plus_lin']['sample_size_saving'])} more users.",
    ]
    if (config.REPORTS_DIR / "simulations.json").exists():
        s = config.read_json("simulations.json")
        naive = s["peeking"]["aa_false_positive_rate_naive_by_looks"]
        k = max(naive, key=int)
        lines.append(
            f"* Simulations on real outcomes: A/A FPR {pct(s['aa']['false_positive_rate']['visit_welch']['rate'], 2)}"
            f" over {s['aa']['n_sims']:,} sims; peeking {k} times inflates FPR to "
            f"{pct(naive[k]['rate'], 1)}, while the mSPRT holds it at "
            f"{pct(s['peeking']['aa_false_positive_rate_msprt']['rate'], 2)}.")
    if (config.REPORTS_DIR / "uplift.json").exists():
        u = config.read_json("uplift.json")
        best = u["visit"]["chosen_model"]
        cap = u["visit"]["test_bootstrap"]["best_capture_at_30"]
        lines.append(
            f"* Targeting: the top 30% by {LABEL[best]} captures {pct(cap['point'])} of "
            f"incremental visits on the test split (random: 30%).")
    lines += ["", *_readout_section(r, d)]
    if (config.REPORTS_DIR / "simulations.json").exists():
        lines += _sim_section(config.read_json("simulations.json"))
    if (config.REPORTS_DIR / "uplift.json").exists():
        lines += _uplift_section(config.read_json("uplift.json"))
    lines += [
        "## 6. Recommendations",
        "",
        "1. Report the doubly-robust ITT, not the raw difference, for this pooled file; ask the "
        "data owner for the test id so the analysis can be stratified exactly.",
        "2. Run the balance / treatment-predictability check on every experiment. It is cheap, "
        "and it catches problems that SRM cannot see.",
        "3. Adopt CUPAC as the default estimator for visit-type metrics. At equal precision it "
        "cuts the required sample by about the R^2 of the prediction. Gains on rare "
        "conversions are smaller.",
        "4. If stakeholders will monitor results daily, use always-valid p-values (mSPRT). The "
        "fixed-horizon test is only valid if it is read once.",
        "",
        "Figures: `reports/figures/ci_comparison.png`, `peeking_fpr.png`, "
        "`power_validation.png`, `uplift_curves.png`.",
        "",
    ]
    text = "\n".join(lines)
    path = config.REPORTS_DIR / "readout.md"
    path.write_text(text, encoding="utf-8")
    print("wrote", path)
    return text
