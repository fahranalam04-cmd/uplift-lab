import duckdb
import numpy as np
import pytest

from abkit.pipeline.data import SPLITS, bucket_expression


@pytest.fixture
def con():
    rng = np.random.default_rng(0)
    n = 20_003
    c = duckdb.connect()
    c.execute("CREATE TABLE d (row_id BIGINT, treatment TINYINT, visit TINYINT)")
    c.executemany("INSERT INTO d VALUES (?, ?, ?)",
                  list(zip(range(n), rng.binomial(1, 0.85, n).tolist(),
                           rng.binomial(1, 0.05, n).tolist(), strict=True)))
    yield c
    c.close()


def _buckets(con, seed):
    return dict(con.execute(f"SELECT row_id, {bucket_expression(seed)} FROM d").fetchall())


def test_buckets_are_exactly_stratified(con):
    rows = con.execute(
        f"""SELECT treatment, visit, b, count(*) FROM
            (SELECT treatment, visit, {bucket_expression()} AS b FROM d)
            GROUP BY ALL"""
    ).fetchall()
    per_stratum: dict[tuple[int, int], list[int]] = {}
    for t, v, _, n in rows:
        per_stratum.setdefault((t, v), []).append(n)
    for counts in per_stratum.values():
        assert len(counts) == 10
        assert max(counts) - min(counts) <= 1  # every bucket is a stratified 10% slice


def test_buckets_are_deterministic_and_seeded(con):
    a, b = _buckets(con, 1), _buckets(con, 1)
    assert a == b
    c = _buckets(con, 2)
    assert np.mean([a[k] != c[k] for k in a]) > 0.8


def test_splits_are_disjoint_and_cover_all_buckets():
    sets = [set(v) for v in SPLITS.values()]
    assert set().union(*sets) == set(range(10))
    assert sum(len(s) for s in sets) == 10
