"""Adam 优化器（对扁平参数）。"""

from __future__ import annotations

import numpy as np


class Adam:
    """标准 Adam（Kingma & Ba 2015）。状态随参数扁平向量。"""

    def __init__(
        self,
        n_params: int,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr 必须 > 0")
        self.lr = float(lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.eps = float(eps)
        self.m = np.zeros(n_params, dtype=np.float64)
        self.v = np.zeros(n_params, dtype=np.float64)
        self.t = 0

    def step(self, flat: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * (grad * grad)
        m_hat = self.m / (1.0 - self.b1**self.t)
        v_hat = self.v / (1.0 - self.b2**self.t)
        return flat - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> dict:
        return {"m": self.m.copy(), "v": self.v.copy(), "t": self.t}
