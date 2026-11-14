import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def make_ohlcv(n=1500, seed=0, drift=0.0004, vol=0.015, start="2019-01-01", price=1000.0):
    """Synthetic daily OHLCV (geometric random walk with regime-ish drift swings)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    swing = 0.0008 * np.sin(np.arange(n) / 120.0)
    rets = rng.normal(drift + swing, vol, n)
    close = price * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
    df.index.name = "Date"
    return df


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run inside an isolated directory so state/ and logs/ are not touched."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "state").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path
