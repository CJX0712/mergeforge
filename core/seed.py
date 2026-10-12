"""全局随机种子入口（唯一）。

使用 numpy legacy ``RandomState`` 而非 ``Generator``：NEP 19 只保证
``RandomState`` 的流跨 numpy 版本稳定，``Generator`` 不保证（混沌级联会
放大微小差异，导致 CI 门禁翻转）。一切随机性（数据、初始化、shuffle、
合并掩码）必须经由本模块或由 ``set_all`` 返回的 ``RandomState`` 派生。
"""

from __future__ import annotations

import random

import numpy as np

_GLOBAL_SEED: int | None = None


def make_rng(seed: int) -> np.random.RandomState:
    """创建确定性的 legacy RandomState。"""
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed 必须为整数，收到 {type(seed).__name__}")
    return np.random.RandomState(int(seed))


def set_all(seed: int) -> tuple[random.Random, np.random.RandomState]:
    """一次设齐全局 random 与 numpy 流，返回 (py_rng, np_rng)。"""
    global _GLOBAL_SEED  # noqa: PLW0603 （模块级种子登记，进程内单值）
    _GLOBAL_SEED = int(seed)
    random.seed(int(seed))
    np.random.seed(int(seed))  # legacy global stream
    return random.Random(int(seed)), np.random.RandomState(int(seed))


def current_seed() -> int | None:
    return _GLOBAL_SEED
