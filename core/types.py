"""MergeForge 核心数据类型。

约定：
- 一切浮点为 float64（合并是逐元素运算，精度即正确性）。
- 模型参数以「扁平向量 + 形状清单」表示，合并方法只依赖该表示，
  与具体网络结构解耦（任何同构模型集合都可合并）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TaskData:
    """单个任务的 (train/val/test) 数据。标签为 0..C-1 整数。"""

    task_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __post_init__(self) -> None:
        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} 必须为 np.ndarray")
        for name in ("X_train", "X_val", "X_test"):
            if getattr(self, name).shape[1] != self.X_train.shape[1]:
                raise ValueError(f"{name} 特征维度与 train 不一致")
        for name in ("y_train", "y_val", "y_test"):
            y = getattr(self, name)
            if y.ndim != 1:
                raise ValueError(f"{name} 必须为一维标签向量")
            if y.size and int(y.min()) < 0:
                raise ValueError(f"{name} 含负标签")
        # 注意：不校验「val/test 出现的类别必须见于 train」——小样本下缺类合法，
        # 且该约束属于 DGP 层（n_classes 已知）而非数据容器层。
        # 标签上界由 MLP.loss_and_grad 对 output_dim 的校验兜底。


@dataclass
class ParamSpec:
    """参数张量清单：扁平向量 <-> 命名张量。"""

    names: list[str]
    shapes: list[tuple[int, ...]]

    @property
    def n_params(self) -> int:
        return int(sum(int(np.prod(s)) for s in self.shapes))

    def split(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        if flat.shape != (self.n_params,):
            raise ValueError(f"扁平参数长度 {flat.shape} != ({self.n_params},)")
        out: dict[str, np.ndarray] = {}
        i = 0
        for name, shape in zip(self.names, self.shapes, strict=True):
            n = int(np.prod(shape))
            out[name] = flat[i : i + n].reshape(shape)
            i += n
        return out

    def flatten(self, tensors: dict[str, np.ndarray]) -> np.ndarray:
        parts = [np.asarray(tensors[name], dtype=np.float64).ravel() for name in self.names]
        return np.concatenate(parts)


@dataclass
class MethodResult:
    """单个合并方法在单个 seed 下的评测结果。"""

    method: str
    per_task_acc: dict[int, float]  # task_id -> test accuracy
    val_acc: float  # 校准/选择所用的验证集准确率（如有）
    n_models_used: int  # 1 = 单模型；K = 需要全部模型的集成参照
    extra: dict = field(default_factory=dict)

    @property
    def mean_acc(self) -> float:
        return float(np.mean(list(self.per_task_acc.values())))


@dataclass
class SeedResult:
    """单个 seed 的完整实验结果。"""

    seed: int
    base: MethodResult
    joint: MethodResult | None
    specialists: list[MethodResult]
    merged: dict[str, MethodResult]
    ensemble: MethodResult
    # 诊断：base -> specialist 与 specialist -> specialist 的线性插值障碍
    barrier_base_to_specialist: dict[int, float] = field(default_factory=dict)
    barrier_specialist_pair: float = 0.0
    # 任务身份路由的集成上界（K 模型 + 任务 ID；恢复率分母）
    oracle_ensemble: MethodResult | None = None
