"""旗舰合并方法 MergeFuse = TIES 符号选举 + 逐张量 λ 校准 + 非劣守护。

设计来源（集大成，零自研 SOTA）：
- TIES-Merging（Yadav et al. 2023）：trim → elect sign → disjoint mean；
- AdaMerging（ICML 2024）的逐层/逐张量系数校准思想（lite 版：坐标下降）；
- Model Soups（Wortsman 2022）的非劣守护：若校准后验证分低于 uniform soup，
  回退 soup（防止验证集噪声把旗舰带到沟里）。

与 DARE 的关系：DARE 作为独立基线与消融项保留（drop & rescale 是另一族
去噪机制，与 TIES 的 trim/sign 机制正交但实证上不叠加，见 model_card）。
"""

from __future__ import annotations

import numpy as np

from core.thread_pin import pin_threads

pin_threads()

from merging.methods import (  # noqa: E402
    task_vectors,
    tensor_slices,
    ties_merge,
    uniform_soup,
)


def merge_fuse(base_flat, model_flats, spec, eval_fn, cfg):
    """返回 (合并参数, 诊断信息)。eval_fn(flat) -> 验证集分数（越大越好）。"""
    deltas = task_vectors(base_flat, model_flats)
    slices = tensor_slices(spec)

    # ---- 0. uniform soup 参照（非劣守护的锚）----
    soup_flat = uniform_soup(base_flat, model_flats, spec)
    soup_acc = eval_fn(soup_flat)

    # ---- 1. TIES：ρ 在验证网格上选优 ----
    best_rho, best_rho_acc, ties_flat = None, -np.inf, None
    for rho in cfg.ties_rho_grid:
        flat = ties_merge(base_flat, deltas, rho, lam=1.0)
        acc = eval_fn(flat)
        if acc > best_rho_acc + 1e-15:
            best_rho, best_rho_acc, ties_flat = float(rho), float(acc), flat

    # ---- 2. 逐张量 λ 坐标下降（2 pass，从 TIES 解出发）----
    fused_delta = ties_flat - base_flat
    lambdas = dict.fromkeys(slices, 1.0)

    def build(lam_map: dict[str, float]) -> np.ndarray:
        out = base_flat.copy()
        for name, sl in slices.items():
            out[sl] = base_flat[sl] + lam_map[name] * fused_delta[sl]
        return out

    cur_flat, cur_acc = ties_flat, best_rho_acc
    n_evals = len(cfg.ties_rho_grid)
    for _pass in range(2):
        for name in slices:
            for lam in cfg.layer_lambda_grid:
                if lam == lambdas[name]:
                    continue
                trial = dict(lambdas)
                trial[name] = float(lam)
                acc = eval_fn(build(trial))
                n_evals += 1
                if acc > cur_acc + 1e-12:
                    lambdas, cur_acc, cur_flat = trial, float(acc), build(trial)

    # ---- 3. 非劣守护：不得低于 uniform soup ----
    guard_triggered = cur_acc < soup_acc - 1e-12
    if guard_triggered:
        return soup_flat, {
            "guard_triggered": True,
            "rho": best_rho,
            "lambdas": dict(lambdas),
            "val_acc": float(soup_acc),
            "ties_val_acc": float(best_rho_acc),
            "soup_val_acc": float(soup_acc),
            "n_evals": n_evals,
        }
    return cur_flat, {
        "guard_triggered": False,
        "rho": best_rho,
        "lambdas": dict(lambdas),
        "val_acc": float(cur_acc),
        "ties_val_acc": float(best_rho_acc),
        "soup_val_acc": float(soup_acc),
        "n_evals": n_evals,
    }
