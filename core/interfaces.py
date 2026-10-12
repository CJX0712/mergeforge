"""MergeForge 接口契约（Protocol）。

合并方法只依赖「扁平参数 + 形状清单」表示，不依赖具体模型类；
任何满足 :class:`TrainableModel` 协议的同构模型集合都可被合并。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TrainableModel(Protocol):
    """可训练模型协议（MLP 满足；未来可换其他同构架构）。"""

    def forward(self, X: np.ndarray) -> np.ndarray: ...

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]: ...

    def get_flat(self) -> np.ndarray: ...

    def set_flat(self, flat: np.ndarray) -> None: ...

    @property
    def param_spec(self): ...


class MergeMethod(Protocol):
    """合并方法协议：输入 base 扁平参数 + K 个 specialist 扁平参数。"""

    name: str

    def merge(self, base_flat: np.ndarray, model_flats: list[np.ndarray], spec) -> np.ndarray: ...
