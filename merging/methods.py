"""合并方法库（Model Soups / Task Arithmetic / TIES / DARE / Greedy Soup）。

全部方法只操作「扁平参数 + 形状清单」，与模型结构解耦：
任何同构模型集合（MLP、LoRA 增量、Transformer 权重）都可复用。

参考文献：
- Wortsman et al. 2022, Model Soups（uniform / greedy）
- Ilharco et al. 2023, Task Arithmetic（base + λΣΔ）
- Yadav et al. 2023, TIES-Merging（trim → elect sign → disjoint mean）
- Yu et al. 2024, DARE（drop & rescale 任务向量去噪）
"""

from __future__ import annotations

import numpy as np

from core.errors import MergeError
from core.thread_pin import pin_threads

pin_threads()


def _check_consistent(base_flat: np.ndarray, model_flats: list[np.ndarray], spec) -> None:
    if not model_flats:
        raise MergeError("至少需要 1 个 specialist 模型")
    for i, m in enumerate(model_flats):
        if m.shape != base_flat.shape:
            raise MergeError(f"model[{i}] 长度 {m.shape} != base {base_flat.shape}")


def task_vectors(base_flat: np.ndarray, model_flats: list[np.ndarray]) -> np.ndarray:
    """Δ_k = M_k - base，堆成 (K, P)。"""
    return np.stack([m - base_flat for m in model_flats], axis=0)


def tensor_slices(spec) -> dict[str, slice]:
    """扁平向量中各参数张量的下标区间。"""
    slices: dict[str, slice] = {}
    i = 0
    for name, shape in zip(spec.names, spec.shapes, strict=True):
        n = int(np.prod(shape))
        slices[name] = slice(i, i + n)
        i += n
    return slices


def uniform_soup(base_flat: np.ndarray, model_flats: list[np.ndarray], spec=None) -> np.ndarray:
    """均匀参数平均（Wortsman 2022）。对输入顺序置换不变。"""
    _check_consistent(base_flat, model_flats, spec)
    return np.mean(np.stack(model_flats, axis=0), axis=0)


def task_arithmetic(
    base_flat: np.ndarray,
    deltas: np.ndarray,
    lam: float,
) -> np.ndarray:
    """base + λ·Σ_k Δ_k（Ilharco 2023）。"""
    if deltas.ndim != 2:
        raise MergeError(f"deltas 必须为 (K, P)，收到 {deltas.shape}")
    return base_flat + float(lam) * deltas.sum(axis=0)


def ties_merge(
    base_flat: np.ndarray, deltas: np.ndarray, rho: float, lam: float = 1.0
) -> np.ndarray:
    """TIES-Merging（Yadav 2023）：

    1. trim：每个任务向量只保留 |Δ| 最大的 ρ 比例元素；
    2. elect sign：逐元素取跨模型修剪后值的和符号（多数决）；
    3. disjoint mean：只平均与当选符号一致的值（其余记 0）。
    """
    if deltas.ndim != 2:
        raise MergeError(f"deltas 必须为 (K, P)，收到 {deltas.shape}")
    if not (0.0 < rho <= 1.0):
        raise MergeError(f"rho 必须在 (0, 1]，收到 {rho}")
    K = deltas.shape[0]
    trimmed = np.zeros_like(deltas)
    for k in range(K):
        mags = np.abs(deltas[k])
        if rho >= 1.0:
            trimmed[k] = deltas[k]
            continue
        thr = np.quantile(mags, 1.0 - rho)
        trimmed[k] = np.where(mags >= thr, deltas[k], 0.0)
    elected = np.sign(trimmed.sum(axis=0))  # (P,) 多数符号；0 = 无共识
    agree = (np.sign(trimmed) == elected[None, :]) & (elected[None, :] != 0)
    counts = agree.sum(axis=0)
    fused = np.where(agree, trimmed, 0.0).sum(axis=0) / np.maximum(counts, 1)
    return base_flat + float(lam) * fused


def dare_merge(
    base_flat: np.ndarray,
    deltas: np.ndarray,
    delta: float,
    lam: float = 1.0,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """DARE（Yu 2024）：以概率 δ 逐元素丢弃任务向量，幸存元素放大 1/(1-δ)。"""
    if not (0.0 <= delta < 1.0):
        raise MergeError(f"delta 必须在 [0, 1)，收到 {delta}")
    if delta == 0.0:
        return base_flat + float(lam) * deltas.sum(axis=0)
    rng = rng or np.random.RandomState(0)
    K, P = deltas.shape
    keep_scale = np.empty_like(deltas)
    for k in range(K):
        mask = (rng.random(P) >= delta).astype(np.float64)
        keep_scale[k] = mask / (1.0 - delta)
    return base_flat + float(lam) * (deltas * keep_scale).sum(axis=0)


def greedy_soup(
    base_flat: np.ndarray,
    model_flats: list[np.ndarray],
    spec,
    eval_fn,
) -> tuple[np.ndarray, list[int]]:
    """Greedy Soup（Wortsman 2022）：在验证集上贪心向前选择成员。

    eval_fn(flat) -> 验证集分数（越大越好）。返回 (合并参数, 选中下标)。
    """
    _check_consistent(base_flat, model_flats, spec)
    K = len(model_flats)
    singles = [eval_fn(m) for m in model_flats]
    order = sorted(range(K), key=lambda i: (-singles[i], i))
    selected = [order[0]]
    best_acc = eval_fn(np.mean(np.stack([model_flats[i] for i in selected]), axis=0))
    improved = True
    while improved:
        improved = False
        for j in range(K):
            if j in selected:
                continue
            cand = [*selected, j]
            acc = eval_fn(np.mean(np.stack([model_flats[i] for i in cand]), axis=0))
            if acc > best_acc + 1e-12:
                best_acc, selected, improved = acc, cand, True
    flat = np.mean(np.stack([model_flats[i] for i in selected], axis=0), axis=0)
    return flat, selected
