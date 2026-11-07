"""Paths and resource limits for the Criteo pipeline (override with environment variables)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.environ.get("ML_DATA_ROOT", "D:/ml-portfolio/data")) / "uplift"
MODELS_DIR = Path(os.environ.get("ML_MODELS_ROOT", "D:/ml-portfolio/models")) / "uplift"
REPORTS_DIR = Path(os.environ.get("UPLIFT_REPORTS_DIR", REPO_ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
RAW_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
RAW_CSV = DATA_DIR / "raw" / "criteo-research-uplift-v2.1.csv.gz"
PARQUET = DATA_DIR / "criteo_uplift_v21.parquet"
CUPAC_PARQUET = DATA_DIR / "cupac_predictions.parquet"

FEATURES = [f"f{i}" for i in range(12)]
OUTCOMES = ["visit", "conversion"]
DESIGN_TREATMENT_SHARE = 0.85  # documented treatment ratio of the Criteo v2.1 release
SEED = 20261107

# Shared-machine etiquette: 4 threads, DuckDB capped at 3 GB.
THREADS = int(os.environ.get("UPLIFT_THREADS", "4"))
DUCKDB_MEMORY = os.environ.get("UPLIFT_DUCKDB_MEMORY", "3GB")

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, str(THREADS))


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_json(name: str, payload: dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    return path


def read_json(name: str) -> dict[str, Any]:
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))
