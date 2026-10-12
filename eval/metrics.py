"""评测指标：逐任务准确率 / 遗忘 / 集成参照 / 线性插值障碍 / 恢复率。

所有指标口径统一：test 折只用于最终评测；val 折只用于合并方法校准。
"""

from __future__ import annotations

import numpy as np

from core.thread_pin import pin_threads

pin_threads()

from data.tasks import add_task_id  # noqa: E402
from models.mlp import MLP  # noqa: E402


def flat_to_model(flat: np.ndarray, spec, template: MLP) -> MLP:
    """用 template 的结构承载扁平参数。"""
    m = template.copy()
    m.set_flat(flat)
    return m


def _task_X(task, k: int, cfg, split: str) -> tuple[np.ndarray, np.ndarray]:
    X = getattr(task, f"X_{split}")
    y = getattr(task, f"y_{split}")
    if cfg.input_task_id:
        X = add_task_id(X, k, cfg.n_tasks)
    return X, y


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    return float((model.predict(X) == y).mean())


def per_task_accuracies(model: MLP, tasks: dict, cfg, split: str = "test") -> dict[int, float]:
    return {k: accuracy(model, *_task_X(tasks[k], k, cfg, split)) for k in sorted(tasks)}


def mean_task_accuracy(model: MLP, tasks: dict, cfg, split: str = "test") -> float:
    accs = per_task_accuracies(model, tasks, cfg, split)
    return float(np.mean(list(accs.values())))


def ensemble_per_task_accuracies(
    models: list[MLP], tasks: dict, cfg, split: str = "test"
) -> dict[int, float]:
    """朴素集成参照（softmax 概率平均；推理需要全部 K 个模型）。

    注意：私有类别子空间下该集成会失效——每个专家把概率质量倒进自己的
    类别块，平均后真实类与随机块类并列（实测近似随机）。它衡量「无任务
    身份时的无监督组合」上界，故 G6 恢复率改用 oracle 路由集成。
    """
    out: dict[int, float] = {}
    for k in sorted(tasks):
        X, y = _task_X(tasks[k], k, cfg, split)
        probs = np.mean([m.predict_proba(X) for m in models], axis=0)
        out[k] = float((probs.argmax(axis=1) == y).mean())
    return out


def oracle_ensemble_per_task_accuracies(
    models: list[MLP], tasks: dict, cfg, split: str = "test"
) -> dict[int, float]:
    """任务身份路由的集成上界：任务 k 的数据交给 specialist k（K 模型 + 任务 ID）。

    任务可从输入 x 识别（mean_shift_rho>0）时该路由合法，是「K 个专家 +
    任务身份」能达到的最强组合，作为合并恢复率的参照上界。
    """
    out: dict[int, float] = {}
    for k in sorted(tasks):
        X, y = _task_X(tasks[k], k, cfg, split)
        out[k] = float((models[k].predict(X) == y).mean())
    return out


def linear_barrier(
    flat_a: np.ndarray,
    flat_b: np.ndarray,
    tasks: dict,
    cfg,
    spec,
    template: MLP,
    n_points: int = 11,
    split: str = "test",
) -> dict:
    """线性插值障碍（Frankle & Carbin 2020 的 connectivity 诊断）。

    返回路径上各 λ 的 mean task accuracy、端点准确率与障碍
    barrier = min(端点) - min(路径)。障碍 ≈ 0 表示两模型同盆（可线性连接）。
    """
    accs: list[float] = []
    for lam in np.linspace(0.0, 1.0, n_points):
        flat = (1.0 - lam) * flat_a + lam * flat_b
        model = flat_to_model(flat, spec, template)
        accs.append(mean_task_accuracy(model, tasks, cfg, split))
    accs_arr = np.asarray(accs)
    return {
        "lambdas": np.linspace(0.0, 1.0, n_points).tolist(),
        "accs": accs,
        "end_a": accs[0],
        "end_b": accs[-1],
        "min_path": float(accs_arr.min()),
        "barrier": float(min(accs[0], accs[-1]) - accs_arr.min()),
    }


def recovery_ratio(base_mean: float, merged_mean: float, ensemble_mean: float) -> float:
    """恢复率 = (merged - base) / (ensemble - base)。

    >1 表示合并单模型超过集成参照（不应发生）；≈1 表示完全恢复；
    <1 表示只恢复了一部分（集成差距的 1-r 部分丢失）。
    """
    denom = ensemble_mean - base_mean
    if abs(denom) < 1e-12:
        return float("nan")
    return float((merged_mean - base_mean) / denom)
