"""BLAS/OpenMP 线程数钉死（确定性前提）。

多线程 BLAS 的矩阵乘求和顺序随调度变化，同 seed 两次运行会产生 ~1e-14 级
差异（MatrixForge 实测 max|Δ|≈1.5e-14）。必须在 numpy 导入前设置环境变量。
"""

from __future__ import annotations

import os

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

_PINNED = False


def pin_threads() -> None:
    """把线程数环境变量钉为 1（仅当调用者未显式设置时）。"""
    global _PINNED  # noqa: PLW0603 （进程级一次性状态标记）
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")
    _PINNED = True


def threads_pinned() -> bool:
    return _PINNED
