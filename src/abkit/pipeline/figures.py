"""Static PNG figures for the README and readout (rendered from reports/*.json only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MUTED = "#9a9993"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})

LABELS = {
    "raw": "Raw difference in means",
    "lin_12_features": "Lin adjustment, 12 features",
    "cupac": "CUPAC (cross-fitted LightGBM)",
    "cupac_plus_lin": "CUPAC + 12 features",
    "aipw": "AIPW (doubly robust)",
    "post_stratified": "Post-stratified on CUPAC deciles",
    "s_learner": "S-learner", "t_learner": "T-learner", "x_learner": "X-learner",
    "transformed_outcome": "Transformed outcome", "response_model": "Response model (baseline)",
    "random": "Random targeting",
}


def _save(fig: plt.Figure, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def ci_comparison(readout: dict) -> Path:
    adj, ps = readout["adjusted"], readout["post_stratified"]
    methods = ["raw", "lin_12_features", "cupac", "cupac_plus_lin", "aipw"]
    colors = [MUTED, *SERIES[:4], SERIES[5]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, outcome in zip(axes, config.OUTCOMES, strict=True):
        rows = [(LABELS[m], adj[outcome][m]["abs_ci_low"], adj[outcome][m]["abs_ci_high"],
                 adj[outcome][m]["abs_effect"],
                 "baseline" if m == "raw"
                 else f"-{100 * adj[outcome][m]['ci_width_reduction']:.1f}% width")
                for m in methods]
        p = ps[outcome]
        rows.append((LABELS["post_stratified"], p["ci_low"], p["ci_high"], p["estimate"],
                     f"{p['n_strata']} strata"))
        ys = np.arange(len(rows))[::-1]
        for y, (_, lo, hi, pt, note), color in zip(ys, rows, colors, strict=True):
            lo, hi, pt = 100 * lo, 100 * hi, 100 * pt
            ax.plot([lo, hi], [y, y], color=color, lw=6, solid_capstyle="round", alpha=0.9)
            ax.plot(pt, y, "o", ms=6, color=SURFACE, mec=INK, mew=1.2)
            ax.annotate(f"{pt:.3f} ± {(hi - lo) / 2:.3f} pp  ({note})", (hi, y), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(ys, [r[0] for r in rows] if outcome == "visit" else [])
        ax.set_title(f"{outcome.capitalize()}: ITT with 95% CI", loc="left")
        ax.set_xlabel("Absolute lift, percentage points")
        ax.grid(axis="y", visible=False)
        lo_all = min(100 * r[1] for r in rows)
        hi_all = max(100 * r[2] for r in rows)
        span = hi_all - lo_all
        ax.set_xlim(lo_all - 0.1 * span, hi_all + 1.0 * span)
    fig.suptitle("13.98M users: adjustment narrows the CI and moves it - the raw ITT is "
                 "confounded by pooling tests", x=0.01, ha="left", fontsize=12,
                 fontweight="semibold")
    fig.tight_layout()
    return _save(fig, "ci_comparison.png")


def peeking(sims: dict) -> Path:
    p = sims["peeking"]
    naive = p["aa_false_positive_rate_naive_by_looks"]
    looks = np.array([int(k) for k in naive])
    rate = np.array([100 * naive[k]["rate"] for k in naive])
    lo = np.array([100 * naive[k]["ci_low"] for k in naive])
    hi = np.array([100 * naive[k]["ci_high"] for k in naive])
    ms = p["aa_false_positive_rate_msprt"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.fill_between(looks, lo, hi, color=SERIES[1], alpha=0.12, lw=0)
    ax.plot(looks, rate, "-o", color=SERIES[1], ms=5, mec=SURFACE, mew=1.5,
            label="Fixed-horizon test, re-run at every look")
    ax.errorbar([looks[-1]], [100 * ms["rate"]],
                yerr=[[100 * (ms["rate"] - ms["ci_low"])], [100 * (ms["ci_high"] - ms["rate"])]],
                fmt="o", color=SERIES[0], ms=7, mec=SURFACE, mew=1.5, capsize=0, lw=2,
                label=f"mSPRT always-valid p-value, {looks[-1]} looks")
    ax.axhline(100 * 0.05, color=INK2, lw=1)
    ax.annotate("alpha = 5%", (looks[0], 5), xytext=(2, 4), textcoords="offset points",
                fontsize=8.5, color=INK2)
    ax.annotate(f"{rate[-1]:.1f}%", (looks[-1], rate[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9)
    ax.annotate(f"{100 * ms['rate']:.1f}%", (looks[-1], 100 * ms["rate"]), xytext=(6, -2),
                textcoords="offset points", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(looks, [str(k) for k in looks])
    ax.minorticks_off()
    ax.set_xlabel("Number of interim looks (equally spaced)")
    ax.set_ylabel("A/A false-positive rate (%)")
    ax.set_ylim(0, max(rate.max(), 10) * 1.12)
    ax.set_title(f"Peeking inflates false positives; mSPRT holds alpha "
                 f"({p['n_sims']:,} A/A sims, {p['n_users_per_sim']:,} users each)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    return _save(fig, "peeking_fpr.png")


def uplift_curves(up: dict) -> Path:
    outcomes = [o for o in config.OUTCOMES if o in up]
    fig, axes = plt.subplots(1, len(outcomes), figsize=(5.2 * len(outcomes), 4), squeeze=False)
    order = ["x_learner", "t_learner", "s_learner", "transformed_outcome", "response_model"]
    for ax, outcome in zip(axes[0], outcomes, strict=True):
        curves = up[outcome]["test_curves"]
        frac = 100 * np.array(curves["random"]["fraction"])
        ax.plot(frac, frac, color=MUTED, lw=1.2, label=LABELS["random"])
        for kind, color in zip(order, SERIES, strict=False):
            u = np.array(curves[kind]["uplift"])
            share = 100 * u / u[-1]
            best = kind == up[outcome]["chosen_model"]
            ax.plot(frac, share, color=color, lw=2.4 if best else 1.6,
                    label=LABELS[kind] + (" (chosen on val)" if best else ""))
        best = up[outcome]["chosen_model"]
        cap = 100 * up[outcome]["test"][best]["capture_at_30"]
        ax.plot([30], [cap], "o", color=SERIES[order.index(best)], ms=7, mec=SURFACE, mew=2)
        ax.annotate(f"top 30% -> {cap:.0f}% of incremental {outcome}s", (30, cap),
                    xytext=(10, -14), textcoords="offset points", fontsize=8.5)
        ax.set_xlabel("Users targeted, ranked by score (%)")
        ax.set_ylabel(f"Share of incremental {outcome}s captured (%)")
        ax.set_title(f"{outcome.capitalize()} uplift curve, test n={up['splits']['test']['n']:,}",
                     loc="left")
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save(fig, "uplift_curves.png")


def power_validation(sims: dict) -> Path:
    pw = sims["power"]
    n = np.array([r["n_users"] for r in pw["grid"]]) / 1_000
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, color, label in (("raw", SERIES[0], "Welch"), ("cupac", SERIES[2], "CUPAC")):
        an = [100 * r[f"analytic_power_{key}"] for r in pw["grid"]]
        sim = [100 * r[f"simulated_power_{key}"]["rate"] for r in pw["grid"]]
        lo = [100 * r[f"simulated_power_{key}"]["ci_low"] for r in pw["grid"]]
        hi = [100 * r[f"simulated_power_{key}"]["ci_high"] for r in pw["grid"]]
        ax.plot(n, an, color=color, lw=1.6, label=f"{label}: analytic calculator")
        ax.errorbar(n, sim, yerr=[np.subtract(sim, lo), np.subtract(hi, sim)], fmt="o",
                    color=color, ms=6, mec=SURFACE, mew=1.5, lw=1.5, capsize=0,
                    label=f"{label}: simulated ({pw['n_sims_per_point']:,} sims/point)")
    ax.axhline(80, color=GRID, lw=1)
    ax.set_xscale("log")
    ax.set_xticks(n, [f"{int(v)}K" for v in n])
    ax.minorticks_off()
    ax.set_ylim(0, 105)
    ax.set_xlabel("Users per experiment (85/15 split)")
    ax.set_ylabel("Power (%)")
    ax.set_title(f"Power for a {100 * pw['rel_lift']:.0f}% relative lift in visit rate",
                 loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save(fig, "power_validation.png")


def run() -> list[Path]:
    out = []
    if (config.REPORTS_DIR / "readout.json").exists():
        out.append(ci_comparison(config.read_json("readout.json")))
    if (config.REPORTS_DIR / "simulations.json").exists():
        sims = config.read_json("simulations.json")
        out += [peeking(sims), power_validation(sims)]
    if (config.REPORTS_DIR / "uplift.json").exists():
        out.append(uplift_curves(config.read_json("uplift.json")))
    for p in out:
        print("wrote", p)
    return out
