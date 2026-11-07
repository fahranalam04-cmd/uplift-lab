"""Download / convert the Criteo Uplift v2.1 CSV to Parquet and stream it back out.

The Parquet file adds two columns:

* ``row_id``: 0-based position in the original CSV.
* ``bucket``: 0..9, assigned by ranking rows within each (treatment, visit) stratum by a
  seeded hash, so every bucket is an exactly stratified 10% slice. Buckets 0-3 are the uplift
  train split, 4-5 validation and 6-9 test; ``bucket % 2`` is the CUPAC cross-fitting fold.
"""

from __future__ import annotations

import hashlib
import time
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from . import config

SPLITS = {"train": (0, 1, 2, 3), "val": (4, 5), "test": (6, 7, 8, 9)}


def bucket_expression(seed: int = config.SEED, n_buckets: int = 10) -> str:
    """SQL for an exactly stratified bucket: rank rows within each (treatment, visit) stratum
    by a seeded hash of row_id, then take the rank modulo ``n_buckets``."""
    return (
        f"CAST((row_number() OVER (PARTITION BY treatment, visit "
        f"ORDER BY hash(row_id + {int(seed)}), row_id) - 1) % {int(n_buckets)} AS TINYINT)"
    )


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    tmp = config.DATA_DIR / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads={config.THREADS}")
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY}'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    return con


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> Path:
    """Fetch the official Criteo release from the Hugging Face hub (criteo/criteo-uplift)."""
    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if force or not config.RAW_CSV.exists():
        print(f"downloading {config.RAW_URL}")
        urllib.request.urlretrieve(config.RAW_URL, config.RAW_CSV)  # noqa: S310 - fixed https URL
    digest = sha256(config.RAW_CSV)
    if digest != config.RAW_SHA256:
        raise RuntimeError(f"checksum mismatch: {digest}")
    return config.RAW_CSV


def prepare(force: bool = False) -> dict[str, object]:
    """CSV.gz -> Parquet (zstd, 1M-row groups) with row_id and stratified bucket columns."""
    if config.PARQUET.exists() and not force:
        out = summary()
        prev = config.REPORTS_DIR / "data_summary.json"
        if prev.exists():  # keep the timing of the original conversion run
            out["conversion_seconds"] = config.read_json("data_summary.json").get(
                "conversion_seconds")
        config.write_json("data_summary.json", out)
        return out
    src = download()
    t0 = time.time()
    staging = config.DATA_DIR / "staging_raw.parquet"
    feats = ", ".join(f"'{f}': 'DOUBLE'" for f in config.FEATURES)
    cols = f"{{{feats}, 'treatment': 'TINYINT', 'conversion': 'TINYINT', 'visit': 'TINYINT', " \
           "'exposure': 'TINYINT'}"
    con = connect()
    con.execute("SET preserve_insertion_order=true")
    con.execute(
        f"COPY (SELECT * FROM read_csv('{src.as_posix()}', header=true, columns={cols})) "
        f"TO '{staging.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)"
    )
    feat_list = ", ".join(config.FEATURES)
    con.execute(
        f"""
        COPY (
          WITH src AS (
            SELECT file_row_number AS row_id, *
            FROM read_parquet('{staging.as_posix()}', file_row_number=true)
          )
          SELECT row_id, {feat_list}, treatment, conversion, visit, exposure,
                 {bucket_expression()} AS bucket
          FROM src
          ORDER BY row_id
        ) TO '{config.PARQUET.as_posix()}'
          (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)
        """
    )
    con.close()
    staging.unlink()
    out = summary()
    out["conversion_seconds"] = round(time.time() - t0, 1)
    config.write_json("data_summary.json", out)
    return out


def summary() -> dict[str, object]:
    con = connect()
    p = config.PARQUET.as_posix()
    row = con.execute(
        f"""SELECT count(*), sum(treatment), avg(treatment), avg(visit), avg(conversion),
                   avg(exposure), sum(CASE WHEN treatment = 0 THEN exposure ELSE 0 END)
            FROM read_parquet('{p}')"""
    ).fetchone()
    buckets = con.execute(
        f"""SELECT bucket, count(*), avg(treatment), avg(visit)
            FROM read_parquet('{p}') GROUP BY bucket ORDER BY bucket"""
    ).fetchall()
    con.close()
    assert row is not None
    return {
        "source_url": config.RAW_URL,
        "source_sha256": config.RAW_SHA256,
        "license": "CC BY-NC-SA 4.0 (non-commercial research use)",
        "parquet_path": str(config.PARQUET),
        "parquet_bytes": config.PARQUET.stat().st_size,
        "n_rows": int(row[0]),
        "n_treatment": int(row[1]),
        "n_control": int(row[0] - row[1]),
        "treatment_share": float(row[2]),
        "visit_rate": float(row[3]),
        "conversion_rate": float(row[4]),
        "exposure_rate": float(row[5]),
        "control_exposed": int(row[6]),
        "buckets": [
            {"bucket": int(b), "n": int(n), "treatment_share": float(ts), "visit_rate": float(v)}
            for b, n, ts, v in buckets
        ],
    }


def iter_batches(
    columns: Sequence[str], batch_size: int = 1_000_000, path: Path | None = None
) -> Iterator[dict[str, np.ndarray]]:
    """Stream columns of the Parquet file as numpy arrays, one row group at a time."""
    pf = pq.ParquetFile(path or config.PARQUET)
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns)):
        yield {c: batch.column(c).to_numpy(zero_copy_only=False) for c in columns}


def load_buckets(
    buckets: Sequence[int],
    columns: Sequence[str],
    sample_frac: float = 1.0,
    seed: int = config.SEED,
    path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load selected buckets (optionally a seeded Bernoulli subsample) into numpy via DuckDB.

    Features come back as float32 to halve memory; labels as int8. ``path`` can point at the
    cross-fitted prediction file, which shares ``row_id`` / ``bucket`` with the main file, so
    the same arguments return row-aligned arrays.
    """
    con = connect()
    cols = ", ".join(columns)
    blist = ", ".join(str(int(b)) for b in buckets)
    where = f"bucket IN ({blist})"
    if sample_frac < 1.0:
        thresh = int(sample_frac * 1_000_000)
        where += f" AND hash(row_id + {seed + 1}) % 1000000 < {thresh}"
    tbl = con.sql(
        f"SELECT {cols} FROM read_parquet('{(path or config.PARQUET).as_posix()}') "
        f"WHERE {where} "
        "ORDER BY row_id"
    ).to_arrow_table()
    con.close()
    out: dict[str, np.ndarray] = {}
    for c in columns:
        arr = tbl.column(c).to_numpy()
        out[c] = arr.astype(np.float32) if c in config.FEATURES else arr
    return out
