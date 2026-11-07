from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abkit.analysis import analyze_experiment
from abkit.inference import diff_in_means

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 4_000
    pre = rng.normal(size=n)
    arm = np.where(rng.random(n) < 0.5, "B", "A")
    y = rng.binomial(1, np.clip(0.2 + 0.1 * pre + 0.03 * (arm == "B"), 0, 1))
    return pd.DataFrame({"arm": arm, "y": y, "pre": pre, "ts": np.arange(n)})


def test_analyze_experiment_contract(frame):
    res = analyze_experiment(frame, "arm", ["y"], ["pre"], treatment_value="B", order_col="ts",
                             n_boot=200)
    assert {"srm", "balance", "outcomes", "n_analyzed"} <= res.keys()
    out = res["outcomes"]["y"]
    t = (frame["arm"] == "B").astype(int)
    ref = diff_in_means(frame["y"], t)
    assert out["welch"]["abs_effect"] == pytest.approx(ref.abs_effect)
    assert out["adjusted"]["abs_se"] < out["welch"]["abs_se"]
    seq = out["sequential"]
    assert len(seq["always_valid_p"]) == len(seq["looks_n"])
    assert seq["looks_n"][-1] == len(frame)


def test_analyze_experiment_rejects_leaky_covariates(frame):
    with pytest.raises(ValueError):
        analyze_experiment(frame, "arm", ["y"], ["y"], treatment_value="B")
    with pytest.raises(ValueError):
        analyze_experiment(frame.assign(exposure=1), "arm", ["y"], ["exposure"],
                           treatment_value="B")
    with pytest.raises(ValueError):
        analyze_experiment(frame, "arm", ["y"])  # string labels need treatment_value


def test_streamlit_app_runs_headless():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Total users" in labels
    at.radio(key="mode").set_value("MDE").run()
    assert not at.exception
    assert "Relative MDE" in [m.label for m in at.metric]
