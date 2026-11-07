"""uplift-lab Streamlit app: experiment design calculator + upload-a-CSV readout (abkit).

Run:  uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from abkit.analysis import analyze_experiment
from abkit.power import minimum_detectable_effect, required_sample_size

st.set_page_config(page_title="uplift-lab", layout="wide")
st.title("uplift-lab: experiment design & readout")
st.caption("Built on the abkit library (tested against scipy / statsmodels reference values).")

design_tab, readout_tab = st.tabs(["Sample size & MDE", "Experiment readout (CSV)"])

with design_tab:
    c1, c2, c3 = st.columns(3)
    metric = c1.radio("Metric type", ["Binary (rate)", "Continuous (mean)"], key="metric")
    baseline = c1.number_input("Baseline (control rate or mean)", value=0.038, format="%.5f",
                               key="baseline")
    sd = None
    if metric.startswith("Continuous"):
        sd = c1.number_input("Standard deviation", value=1.0, min_value=1e-9, key="sd")
    alpha = c2.number_input("Alpha (two-sided)", value=0.05, min_value=0.001, max_value=0.5,
                            key="alpha")
    power = c2.number_input("Power", value=0.80, min_value=0.5, max_value=0.99, key="power")
    share = c2.slider("Share of users in treatment", 0.05, 0.95, 0.5, 0.05, key="share")
    r2 = c3.slider("Variance reduction from CUPED/CUPAC (R², %)", 0, 90, 0, key="r2",
                   help="1 - (adjusted SE / raw SE)^2 from a past experiment on this metric.")
    mode = c3.radio("Solve for", ["Sample size", "MDE"], key="mode")
    try:
        if mode == "Sample size":
            rel = c3.number_input("Relative MDE (%)", value=5.0, min_value=0.01, key="rel_mde")
            ss = required_sample_size(baseline, rel / 100, relative=True, sd=sd, alpha=alpha,
                                      power=power, treat_fraction=share,
                                      variance_reduction=r2 / 100)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total users", f"{ss.n_total:,}")
            m2.metric("Treatment", f"{ss.n_treatment:,}")
            m3.metric("Control", f"{ss.n_control:,}")
            if r2:
                raw = required_sample_size(baseline, rel / 100, relative=True, sd=sd,
                                           alpha=alpha, power=power, treat_fraction=share)
                st.info(f"Without variance reduction you would need {raw.n_total:,} users "
                        f"({1 - ss.n_total / raw.n_total:.1%} more saved by adjustment).")
        else:
            n = c3.number_input("Total users available", value=200_000, min_value=100,
                                step=1_000, key="n_users")
            mde = minimum_detectable_effect(n, baseline, sd=sd, alpha=alpha, power=power,
                                            treat_fraction=share, variance_reduction=r2 / 100)
            m1, m2 = st.columns(2)
            m1.metric("Absolute MDE", f"{mde:.5g}")
            m2.metric("Relative MDE", f"{mde / baseline:.2%}")
    except ValueError as err:
        st.error(str(err))


def _demo_frame(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pre = rng.gamma(2.0, 1.0, n)
    variant = np.where(rng.random(n) < 0.5, "treatment", "control")
    p = np.clip(0.03 + 0.01 * pre + 0.004 * (variant == "treatment"), 0, 1)
    return pd.DataFrame({
        "user_id": np.arange(n),
        "assigned_at": pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.random(n)) * 14,
                                                                      unit="D"),
        "variant": variant,
        "pre_period_sessions": pre,
        "converted": rng.binomial(1, p),
        "revenue": rng.binomial(1, p) * rng.lognormal(3, 1, n),
    })


with readout_tab:
    st.write("Upload one row per randomized user: an assignment column, outcome column(s), "
             "optional **pre-treatment** covariates and an optional time column for the "
             "always-valid sequential test.")
    upload = st.file_uploader("CSV file", type="csv", key="csv")
    use_demo = st.checkbox("Use a synthetic demo dataset instead", value=upload is None,
                           key="demo")
    df = None
    if upload is not None and not use_demo:
        df = pd.read_csv(io.BytesIO(upload.getvalue()))
    elif use_demo:
        df = _demo_frame()
    if df is not None:
        st.dataframe(df.head(), use_container_width=True)
        cols = list(df.columns)
        a1, a2, a3 = st.columns(3)
        tcol = a1.selectbox("Assignment column", cols,
                            index=cols.index("variant") if "variant" in cols else 0, key="tcol")
        labels = list(pd.unique(df[tcol].dropna()))
        tval = a1.selectbox("Treatment label", labels,
                            index=labels.index("treatment") if "treatment" in labels else 0,
                            key="tval")
        expected = a1.number_input("Intended treatment share", value=0.5, min_value=0.01,
                                   max_value=0.99, key="expected")
        default_y = [c for c in ("converted", "revenue") if c in cols]
        ycols = a2.multiselect("Outcome column(s)", [c for c in cols if c != tcol],
                               default=default_y, key="ycols")
        default_x = [c for c in ("pre_period_sessions",) if c in cols]
        xcols = a2.multiselect("Pre-treatment covariates (CUPED / regression adjustment)",
                               [c for c in cols if c not in (tcol, *ycols)], default=default_x,
                               key="xcols")
        ocol = a3.selectbox("Time / order column (optional, for mSPRT)",
                            ["(none)", *cols],
                            index=(cols.index("assigned_at") + 1) if "assigned_at" in cols
                            else 0, key="ocol")
        alpha_r = a3.number_input("Alpha", value=0.05, min_value=0.001, max_value=0.5,
                                  key="alpha_r")
        if ycols:
            try:
                res = analyze_experiment(
                    df, tcol, ycols, xcols, treatment_value=tval,
                    expected_treatment_share=expected, alpha=alpha_r,
                    order_col=None if ocol == "(none)" else ocol, n_boot=1_000,
                )
            except ValueError as err:
                st.error(str(err))
            else:
                srm = res["srm"]
                (st.error if srm["mismatch"] else st.success)(
                    f"SRM check: observed treatment share {srm['observed_ratios'][0]:.4f} vs "
                    f"intended {srm['expected_ratios'][0]:.4f}, chi² p = {srm['p_value']:.3g}"
                    + (" - investigate before trusting results." if srm["mismatch"] else ".")
                )
                if "balance" in res:
                    st.caption("Covariate balance (|SMD| < 0.1 is conventionally balanced)")
                    st.dataframe(pd.DataFrame(res["balance"]), use_container_width=True)
                rows = []
                for y, r in res["outcomes"].items():
                    w, b = r["welch"], r["bootstrap"]
                    rows.append({"outcome": y, "method": "Welch", "control": w["control_mean"],
                                 "treatment": w["treatment_mean"], "lift": w["abs_effect"],
                                 "ci_low": w["abs_ci_low"], "ci_high": w["abs_ci_high"],
                                 "rel_lift": w["rel_effect"], "p_value": w["p_value"]})
                    rows.append({"outcome": y, "method": "Bootstrap (percentile)",
                                 "lift": b["abs_effect"], "ci_low": b["abs_ci_low"],
                                 "ci_high": b["abs_ci_high"], "rel_lift": b["rel_effect"]})
                    if "adjusted" in r:
                        a = r["adjusted"]
                        rows.append({"outcome": y, "method": "Regression-adjusted (Lin)",
                                     "lift": a["abs_effect"], "ci_low": a["abs_ci_low"],
                                     "ci_high": a["abs_ci_high"], "rel_lift": a["rel_effect"],
                                     "p_value": a["p_value"],
                                     "ci_width_reduction": a["ci_width_reduction"]})
                    if "sequential" in r:
                        s = r["sequential"]
                        rows.append({"outcome": y, "method": "mSPRT (always valid)",
                                     "lift": s["estimate"][-1], "ci_low": s["ci_low"][-1],
                                     "ci_high": s["ci_high"][-1],
                                     "p_value": s["always_valid_p"][-1]})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                for y, r in res["outcomes"].items():
                    if "sequential" in r:
                        s = r["sequential"]
                        st.caption(f"{y}: always-valid p-value by sample size")
                        st.line_chart(pd.DataFrame({"users": s["looks_n"],
                                                    "always_valid_p": s["always_valid_p"]})
                                      .set_index("users"))
