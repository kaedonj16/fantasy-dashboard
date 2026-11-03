"""Cleanup Batch 2a: parquet IO resolves to pyarrow with no engine= arg.

Every read_parquet / to_parquet call site in the repo passes no engine,
so pandas silently picked fastparquet while it was installed. fastparquet
(and its cramjam dependency) are removed from requirements and pyarrow is
pinned instead; this proves the no-engine path now lands on pyarrow and
round-trips.
"""
import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def test_parquet_engine_is_pyarrow():
    from pandas.io.parquet import get_engine

    engine = get_engine("auto")
    assert "arrow" in type(engine).__name__.lower()


def test_parquet_roundtrip_without_engine_arg(tmp_path):
    import pandas as pd

    df = pd.DataFrame(
        {
            "player": ["alpha", "beta", "gamma"],
            "pts": [10.5, 0.0, 23.25],
            "week": [1, 1, 2],
        }
    )
    path = tmp_path / "fixture.parquet"
    df.to_parquet(path)  # no engine=, same shape as the app call sites
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(back, df)
