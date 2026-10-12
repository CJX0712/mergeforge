"""core 基础设施：线程钉死（确定性前提）+ 种子入口。"""

from __future__ import annotations

import os

import numpy as np
import pytest

from core import thread_pin
from core.seed import current_seed, make_rng, set_all


def test_pin_threads_sets_all_blas_vars(monkeypatch):
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.delenv(var, raising=False)
    thread_pin.pin_threads()
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        assert os.environ[var] == "1"


def test_pin_threads_does_not_override_existing(monkeypatch):
    """已显式设置的不覆盖（setdefault 语义）。"""
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    thread_pin.pin_threads()
    assert os.environ["OMP_NUM_THREADS"] == "4"


def test_make_rng_deterministic():
    a = make_rng(42)
    b = make_rng(42)
    assert np.array_equal(a.normal(size=10), b.normal(size=10))


def test_make_rng_rejects_non_int():
    with pytest.raises(TypeError):
        make_rng(1.5)  # type: ignore[arg-type]


def test_set_all_sets_global_stream():
    py_rng, np_rng = set_all(9)
    assert current_seed() == 9
    assert np.array_equal(np_rng.normal(size=5), np.random.RandomState(9).normal(size=5))
    assert py_rng.random() == __import__("random").Random(9).random()
