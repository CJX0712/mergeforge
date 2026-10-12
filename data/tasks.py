"""多任务合成数据生成（固定 seed，可复现）。

统一 DGP（regime A3：私有标签子空间 + 任务从输入推断；regime B：任务 ID 入输入）：
- 共享潜因子 M ∈ R^{d×r}（跨任务相同）→ 任务间共享 trunk 结构；
- 任务私有 readout V_k ∈ R^{r×C} 且**类别子空间不相交**：全局标签空间为
  K×C，任务 k 独占类别块 [kC, (k+1)C)（多任务微调合并的经典设定——
  各专家的头部知识位于不相交的参数子空间，参数平均即「路由组合」）；
- 任务均值平移 μ_k（‖μ_k‖ = mean_shift_rho）→ 任务可从输入 x 识别；
- 可选每任务正交旋转 R_k（rotate_inputs）；
- 潜变量 z ~ N(0, I_r)，观测 x = μ_k + (z @ Mᵀ) @ R_kᵀ，
  标签 y = k·C + argmax(z @ V_k) + 标签噪声（翻盘保持在任务自己的类别块内）。

可合并性不变量（regime A3）：专家 k 的头部行在块 k 上强、其他块被 softmax
压制 ≈ 0 → 均匀平均后每行仍只携带对应专家的信号（稀释 1/K 但 argmax 不变），
合并单模型无需任务身份即可路由；而朴素概率集成在私有标签设定下失效
（各专家把概率质量倒进自己的块）——合并相对集成的结构性优势。

train/val/test 互不相交；val 仅供合并方法校准，test 只用于最终评测
（无数据泄漏：合并方法从未接触 test）。
"""

from __future__ import annotations

import numpy as np

from core.seed import make_rng
from core.thread_pin import pin_threads

pin_threads()

from core.types import TaskData  # noqa: E402


def _orthogonal(rng: np.random.RandomState, d: int) -> np.ndarray:
    """QR 分解生成正交矩阵（Haar 分布近似）。"""
    a = rng.normal(0.0, 1.0, size=(d, d))
    q, _ = np.linalg.qr(a)
    return q


def _sample_split(
    rng: np.random.RandomState,
    n: int,
    M: np.ndarray,
    V: np.ndarray,
    mu: np.ndarray,
    R: np.ndarray,
    label_noise: float,
    n_classes: int,
    class_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """class_offset = k·C：任务 k 的类别块起点（全局标签空间 K×C）。"""
    z = rng.normal(0.0, 1.0, size=(n, M.shape[1]))
    o = z @ M.T
    if R is not None:
        o = o @ R.T
    X = o + mu
    y = (z @ V).argmax(axis=1)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        n_flip = int(flip.sum())
        if n_flip:
            # 翻转为「同任务类别块内」的另一个随机类（非原类）
            new_y = rng.randint(0, n_classes - 1, size=n_flip)
            new_y = np.where(new_y >= y[flip], new_y + 1, new_y)
            y[flip] = new_y
    return X, (y + class_offset).astype(np.int64)


def make_multitask_dataset(
    d: int,
    n_tasks: int,
    n_classes: int,
    n_train: int,
    n_val: int,
    n_test: int,
    label_noise: float,
    latent_rank: int,
    prototype_scale: float,
    mean_shift_rho: float,
    rotate_inputs: bool,
    input_task_id: bool,
    seed: int,
) -> dict[int, TaskData]:
    """生成 K 个任务的 (train/val/test)。同 seed 两次调用逐位一致。

    全局标签空间为 n_tasks × n_classes；任务 k 的标签落在
    [k·n_classes, (k+1)·n_classes)（私有类别子空间，见模块 docstring）。
    """
    rng = make_rng(seed)
    M = rng.normal(0.0, 1.0, size=(d, latent_rank)) / np.sqrt(latent_rank)
    tasks: dict[int, TaskData] = {}
    for k in range(n_tasks):
        rng_k = make_rng(seed * 100_003 + k)
        V = rng_k.normal(0.0, 1.0, size=(latent_rank, n_classes)) * prototype_scale
        R = _orthogonal(rng_k, d) if rotate_inputs else None
        if mean_shift_rho > 0:
            mu = rng_k.normal(0.0, 1.0, size=d)
            mu = mu / np.linalg.norm(mu) * mean_shift_rho
        else:
            mu = np.zeros(d)
        splits = {}
        for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
            X, y = _sample_split(rng_k, n, M, V, mu, R, label_noise, n_classes, k * n_classes)
            splits[split] = (X, y)
        tasks[k] = TaskData(
            task_id=k,
            X_train=splits["train"][0],
            y_train=splits["train"][1],
            X_val=splits["val"][0],
            y_val=splits["val"][1],
            X_test=splits["test"][0],
            y_test=splits["test"][1],
        )
    return tasks


def add_task_id(X: np.ndarray, task_id: int, n_tasks: int) -> np.ndarray:
    """拼接 task one-hot（列序 = task_id）。"""
    onehot = np.zeros((X.shape[0], n_tasks), dtype=np.float64)
    onehot[:, task_id] = 1.0
    return np.hstack([X, onehot])
