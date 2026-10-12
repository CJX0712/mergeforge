"""MergeForge 主流水线：单 seed 实验 + 多 seed 基准 + 门禁评测。

调用方向（单向无环）：cli → pipeline → {data, finetune, merging, eval} → core。
"""

from __future__ import annotations

import time

import numpy as np

from core.config import MergeConfig
from core.seed import make_rng
from core.thread_pin import pin_threads

pin_threads()

from core.types import MethodResult, SeedResult  # noqa: E402
from data.tasks import make_multitask_dataset  # noqa: E402
from eval.metrics import (  # noqa: E402
    ensemble_per_task_accuracies,
    flat_to_model,
    linear_barrier,
    mean_task_accuracy,
    oracle_ensemble_per_task_accuracies,
    per_task_accuracies,
)
from finetune.trainer import make_init_model, train_base, train_specialist  # noqa: E402
from merging.fuse import merge_fuse  # noqa: E402
from merging.methods import (  # noqa: E402
    dare_merge,
    greedy_soup,
    task_arithmetic,
    task_vectors,
    ties_merge,
    uniform_soup,
)

# 预注册门禁（方案阶段定死，不因结果迁就）
GATE_FUSE_VS_SOUP = 0.01  # 旗舰必须胜 uniform soup 的幅度（配对判据）
GATE_FUSE_VS_BASE = 0.02  # 旗舰必须胜 base（不合并）的幅度
GATE_FUSE_VS_SPECIALIST = 0.0  # 旗舰不得低于单 specialist 的跨任务均值
GATE_FUSE_VS_GREEDY = 0.0  # 旗舰不得低于 val 贪心选择（单模型选择基线）

CORE_METRIC_METHODS = (
    "base",
    "joint",
    "uniform_soup",
    "greedy_soup",
    "task_arithmetic",
    "ties",
    "dare",
    "mergefuse",
    "ensemble",
    "oracle_ensemble",
    "specialist_mean",
)


class MergePipeline:
    """一个配置 = 一套完整实验。"""

    def __init__(self, cfg: MergeConfig) -> None:
        cfg.validate()
        self.cfg = cfg

    # ---- 单 seed ----
    def run_seed(self, seed: int) -> SeedResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        tasks = make_multitask_dataset(
            d=cfg.d,
            n_tasks=cfg.n_tasks,
            n_classes=cfg.n_classes,
            n_train=cfg.n_train,
            n_val=cfg.n_val,
            n_test=cfg.n_test,
            label_noise=cfg.label_noise,
            latent_rank=cfg.latent_rank,
            prototype_scale=cfg.prototype_scale,
            mean_shift_rho=cfg.mean_shift_rho,
            rotate_inputs=cfg.rotate_inputs,
            input_task_id=cfg.input_task_id,
            seed=seed,
        )

        # specialist 锚点：base（从联合训练模型微调，特征基共享）或
        # init（从共享初始化训练）。freeze_trunk 时只训输出层（LoRA 式）。
        init = make_init_model(cfg, seed)
        base = train_base(tasks, cfg, seed)
        # joint 参照恒用全局标签（pretext 只作用于 base 本身）
        joint = train_base(tasks, cfg, seed, epochs=cfg.joint_epochs, pretext=False)
        anchor_model = init if cfg.specialist_from == "init" else base
        specialists = [
            train_specialist(
                anchor_model,
                tasks[k],
                k,
                cfg,
                seed,
                freeze_trunk=cfg.freeze_trunk,
                block_loss=cfg.specialist_block_loss,
            )
            for k in range(cfg.n_tasks)
        ]
        template = anchor_model.copy()
        spec = anchor_model.param_spec

        def val_acc_of(flat: np.ndarray) -> float:
            model = flat_to_model(flat, spec, template)
            return mean_task_accuracy(model, tasks, cfg, "val")

        anchor_flat = anchor_model.get_flat()
        model_flats = [m.get_flat() for m in specialists]
        deltas = task_vectors(anchor_flat, model_flats)

        # ---- 参照 ----
        base_res = MethodResult(
            method="base",
            per_task_acc=per_task_accuracies(base, tasks, cfg, "test"),
            val_acc=val_acc_of(anchor_flat),
            n_models_used=1,
        )
        joint_res = MethodResult(
            method="joint",
            per_task_acc=per_task_accuracies(joint, tasks, cfg, "test"),
            val_acc=mean_task_accuracy(joint, tasks, cfg, "val"),
            n_models_used=1,
        )
        spec_res = [
            MethodResult(
                method=f"specialist_{k}",
                per_task_acc=per_task_accuracies(m, tasks, cfg, "test"),
                val_acc=mean_task_accuracy(m, tasks, cfg, "val"),
                n_models_used=1,
            )
            for k, m in enumerate(specialists)
        ]

        # ---- 合并方法 ----
        merged: dict[str, MethodResult] = {}

        soup_flat = uniform_soup(anchor_flat, model_flats, spec)
        merged["uniform_soup"] = MethodResult(
            method="uniform_soup",
            per_task_acc=per_task_accuracies(
                flat_to_model(soup_flat, spec, template), tasks, cfg, "test"
            ),
            val_acc=val_acc_of(soup_flat),
            n_models_used=1,
        )

        greedy_flat, greedy_idx = greedy_soup(anchor_flat, model_flats, spec, val_acc_of)
        merged["greedy_soup"] = MethodResult(
            method="greedy_soup",
            per_task_acc=per_task_accuracies(
                flat_to_model(greedy_flat, spec, template), tasks, cfg, "test"
            ),
            val_acc=val_acc_of(greedy_flat),
            n_models_used=1,
            extra={"selected": list(greedy_idx)},
        )

        # Task Arithmetic：λ 网格（val 选优）
        best_lam, best_lam_acc, ta_flat = None, -np.inf, None
        for lam in cfg.ta_lambda_grid:
            flat = task_arithmetic(anchor_flat, deltas, lam)
            acc = val_acc_of(flat)
            if acc > best_lam_acc + 1e-15:
                best_lam, best_lam_acc, ta_flat = float(lam), float(acc), flat
        merged["task_arithmetic"] = MethodResult(
            method="task_arithmetic",
            per_task_acc=per_task_accuracies(
                flat_to_model(ta_flat, spec, template), tasks, cfg, "test"
            ),
            val_acc=best_lam_acc,
            n_models_used=1,
            extra={"lambda": best_lam},
        )

        # TIES：(ρ, λ) 联合网格
        best_key, best_ties_acc, ties_flat = None, -np.inf, None
        for rho in cfg.ties_rho_grid:
            for lam in cfg.ta_lambda_grid:
                flat = ties_merge(anchor_flat, deltas, rho, lam)
                acc = val_acc_of(flat)
                if acc > best_ties_acc + 1e-15:
                    best_key, best_ties_acc, ties_flat = (float(rho), float(lam)), float(acc), flat
        merged["ties"] = MethodResult(
            method="ties",
            per_task_acc=per_task_accuracies(
                flat_to_model(ties_flat, spec, template), tasks, cfg, "test"
            ),
            val_acc=best_ties_acc,
            n_models_used=1,
            extra={"rho": best_key[0], "lambda": best_key[1]},
        )

        # DARE：(δ, λ) 联合网格（掩码 rng 按 seed 确定性派生）
        dare_rng = make_rng(seed * 100_007 + 4242)
        best_dkey, best_dare_acc, dare_flat = None, -np.inf, None
        for delta in cfg.dare_delta_grid:
            for lam in cfg.ta_lambda_grid:
                flat = dare_merge(anchor_flat, deltas, delta, lam, rng=dare_rng)
                acc = val_acc_of(flat)
                if acc > best_dare_acc + 1e-15:
                    best_dkey, best_dare_acc, dare_flat = (
                        (float(delta), float(lam)),
                        float(acc),
                        flat,
                    )
        merged["dare"] = MethodResult(
            method="dare",
            per_task_acc=per_task_accuracies(
                flat_to_model(dare_flat, spec, template), tasks, cfg, "test"
            ),
            val_acc=best_dare_acc,
            n_models_used=1,
            extra={"delta": best_dkey[0], "lambda": best_dkey[1]},
        )

        # 旗舰 MergeFuse
        fuse_flat, fuse_diag = merge_fuse(anchor_flat, model_flats, spec, val_acc_of, cfg)
        merged["mergefuse"] = MethodResult(
            method="mergefuse",
            per_task_acc=per_task_accuracies(
                flat_to_model(fuse_flat, spec, template), tasks, cfg, "test"
            ),
            val_acc=fuse_diag["val_acc"],
            n_models_used=1,
            extra=fuse_diag,
        )

        # ---- 集成参照（两个口径）----
        # ensemble：朴素概率平均（无任务身份；私有标签子空间下失效，作为对照）
        ens_test = ensemble_per_task_accuracies(specialists, tasks, cfg, "test")
        ens_val = ensemble_per_task_accuracies(specialists, tasks, cfg, "val")
        ensemble_res = MethodResult(
            method="ensemble",
            per_task_acc=ens_test,
            val_acc=float(np.mean(list(ens_val.values()))),
            n_models_used=cfg.n_tasks,
        )
        # oracle_ensemble：任务身份路由（K 模型 + 任务 ID）——恢复率的合法上界
        orc_test = oracle_ensemble_per_task_accuracies(specialists, tasks, cfg, "test")
        orc_val = oracle_ensemble_per_task_accuracies(specialists, tasks, cfg, "val")
        oracle_res = MethodResult(
            method="oracle_ensemble",
            per_task_acc=orc_test,
            val_acc=float(np.mean(list(orc_val.values()))),
            n_models_used=cfg.n_tasks,
        )

        # ---- 诊断：线性插值障碍 ----
        barriers = {
            k: linear_barrier(anchor_flat, model_flats[k], tasks, cfg, spec, template, n_points=9)[
                "barrier"
            ]
            for k in range(cfg.n_tasks)
        }
        pair_barrier = linear_barrier(
            model_flats[0], model_flats[1], tasks, cfg, spec, template, n_points=9
        )["barrier"]

        elapsed = time.perf_counter() - t0
        result = SeedResult(
            seed=seed,
            base=base_res,
            joint=joint_res,
            specialists=spec_res,
            merged=merged,
            ensemble=ensemble_res,
            barrier_base_to_specialist=barriers,
            barrier_specialist_pair=float(pair_barrier),
            oracle_ensemble=oracle_res,
        )
        result.elapsed_sec = elapsed  # type: ignore[attr-defined]
        return result

    # ---- 多 seed 基准 + 门禁 ----
    def benchmark(self, seeds: list[int] | None = None) -> dict:
        seeds = list(seeds or self.cfg.seeds)
        t0 = time.perf_counter()
        results = [self.run_seed(s) for s in seeds]

        def collect(name: str) -> list[float]:
            out = []
            for r in results:
                if name == "base":
                    out.append(r.base.mean_acc)
                elif name == "joint":
                    out.append(r.joint.mean_acc if r.joint is not None else float("nan"))
                elif name == "ensemble":
                    out.append(r.ensemble.mean_acc)
                elif name == "oracle_ensemble":
                    out.append(
                        r.oracle_ensemble.mean_acc
                        if r.oracle_ensemble is not None
                        else float("nan")
                    )
                elif name == "specialist_mean":
                    out.append(float(np.mean([m.mean_acc for m in r.specialists])))
                else:
                    out.append(r.merged[name].mean_acc)
            return out

        methods: dict[str, dict] = {}
        for name in CORE_METRIC_METHODS:
            per_seed = collect(name)
            methods[name] = {
                "mean": float(np.mean(per_seed)),
                "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                "per_seed": [float(x) for x in per_seed],
            }

        fuse = methods["mergefuse"]
        soup = methods["uniform_soup"]
        base = methods["base"]
        joint = methods["joint"]
        spec_mean = methods["specialist_mean"]
        greedy = methods["greedy_soup"]
        ens = methods["ensemble"]
        oracle = methods["oracle_ensemble"]

        # 配对差（同 seed 配对消除 DGP 级方差；跨 seed 非配对 std 含 DGP 方差，
        # 会淹没方法差异——FlowForge 踩坑库 §N：跨数据集聚合不可做显著性判据）
        fuse_diffs = [
            r.merged["mergefuse"].mean_acc - r.merged["uniform_soup"].mean_acc for r in results
        ]
        n_seeds = len(results)
        diff_mean = float(np.mean(fuse_diffs))
        diff_std = float(np.std(fuse_diffs, ddof=1)) if n_seeds > 1 else 0.0
        paired_sig = diff_mean > 0.5 * diff_std  # SOP ½σ 规则作用于配对差
        wins = int(sum(1 for d in fuse_diffs if d > 0))

        gates = {
            "G1_fuse_vs_soup": {
                "delta": float(fuse["mean"] - soup["mean"]),
                "paired_delta_mean": diff_mean,
                "paired_delta_std": diff_std,
                "threshold": GATE_FUSE_VS_SOUP,
                "paired_significant": bool(paired_sig),
                "wins": wins,
                "n_seeds": n_seeds,
                "pass": bool(
                    (fuse["mean"] - soup["mean"]) >= GATE_FUSE_VS_SOUP
                    and paired_sig
                    and wins >= max(2, (n_seeds + 1) // 2)
                ),
            },
            "G2_fuse_vs_base": {
                "delta": float(fuse["mean"] - base["mean"]),
                "threshold": GATE_FUSE_VS_BASE,
                "pass": bool((fuse["mean"] - base["mean"]) >= GATE_FUSE_VS_BASE),
            },
            "G3_fuse_vs_specialist": {
                "delta": float(fuse["mean"] - spec_mean["mean"]),
                "threshold": GATE_FUSE_VS_SPECIALIST,
                "pass": bool((fuse["mean"] - spec_mean["mean"]) >= GATE_FUSE_VS_SPECIALIST),
            },
            "G4_fuse_vs_greedy": {
                "delta": float(fuse["mean"] - greedy["mean"]),
                "threshold": GATE_FUSE_VS_GREEDY,
                "pass": bool((fuse["mean"] - greedy["mean"]) >= GATE_FUSE_VS_GREEDY),
            },
            "G5_joint_reference": {
                "joint_mean": joint["mean"],
                "base_mean": base["mean"],
                "joint_minus_base": float(joint["mean"] - base["mean"]),
                "fuse_minus_joint": float(fuse["mean"] - joint["mean"]),
                # joint<=base 时（多任务干扰/过拟合导致联合训练无增益）recovery 无定义，
                # 置 None 而非输出巨大的伪比值；报告叙述据此切换解释路径。
                "recovery_vs_joint": (
                    float((fuse["mean"] - base["mean"]) / max(joint["mean"] - base["mean"], 1e-12))
                    if joint["mean"] > base["mean"]
                    else None
                ),
                "pass": True,  # 参照项，非门禁
            },
            "G6_ensemble_reference": {
                "ensemble_mean": ens["mean"],
                "oracle_ensemble_mean": oracle["mean"],
                # 恢复率分母用 oracle 路由集成（任务身份下的最强 K 模型组合）；
                # 朴素概率集成在私有标签子空间下失效，仅作对照列示。
                "recovery": float(
                    (fuse["mean"] - base["mean"]) / max(oracle["mean"] - base["mean"], 1e-12)
                ),
                "pass": True,  # 参照项，非门禁
            },
        }

        # 消融：TIES-only（旗舰去掉逐张量 λ）与守护统计
        guard_flags = [
            bool(r.merged["mergefuse"].extra.get("guard_triggered", False)) for r in results
        ]
        ablations = {
            "uniform_soup": soup["mean"],
            "ties_only": methods["ties"]["mean"],
            "mergefuse": fuse["mean"],
            "guard_triggered_runs": int(sum(guard_flags)),
            "fuse_minus_ties": float(fuse["mean"] - methods["ties"]["mean"]),
            "fuse_minus_soup": float(fuse["mean"] - soup["mean"]),
        }

        # 失败案例（从最后一个 seed 的 results 派生，禁止预设结论）
        # 归因分支顺序：先判「旗舰 vs soup」（合并机制信号），再判「集成可恢复性」
        # （任务难度信号）——避免任务难度掩盖旗舰未超 soup 的事实。
        last = results[-1]
        failure_cases = []
        for k in range(self.cfg.n_tasks):
            sp_own = last.specialists[k].per_task_acc[k]
            fu = last.merged["mergefuse"].per_task_acc[k]
            so = last.merged["uniform_soup"].per_task_acc[k]
            en = last.ensemble.per_task_acc[k]
            ba = last.base.per_task_acc[k]
            failure_cases.append(
                {
                    "task": k,
                    "specialist_own": sp_own,
                    "mergefuse": fu,
                    "uniform_soup": so,
                    "ensemble": en,
                    "base": ba,
                    "forgetting": sp_own - fu,
                    "fuse_minus_soup": fu - so,
                }
            )
        failure_cases.sort(key=lambda c: -c["forgetting"])
        for c in failure_cases[:3]:
            if c["forgetting"] <= 0:
                c["attribution"] = "旗舰无遗忘（相对 specialist 本任务）"
            elif c["uniform_soup"] < c["mergefuse"] - 1e-9:
                c["attribution"] = "合并机制共性遗忘（旗舰优于 soup，仍不及 specialist）"
            elif c["ensemble"] >= c["specialist_own"] - 0.02:
                c["attribution"] = (
                    f"任务可恢复（集成参照 {c['ensemble']:.3f} 接近 specialist "
                    f"{c['specialist_own']:.3f}），但旗舰未超 soup"
                    f"（fuse-soup={c['fuse_minus_soup']:+.4f}）：逐张量 λ 校准未帮到该任务"
                )
            else:
                c["attribution"] = (
                    f"任务本身难：集成参照 {c['ensemble']:.3f} 亦未恢复 specialist "
                    f"{c['specialist_own']:.3f}；旗舰 vs soup={c['fuse_minus_soup']:+.4f}"
                )

        report = {
            "system": "mergeforge",
            "version": "0.1.0",
            "author": "晨星",
            "config": {
                "d": self.cfg.d,
                "n_tasks": self.cfg.n_tasks,
                "n_classes": self.cfg.n_classes,
                "n_train": self.cfg.n_train,
                "n_val": self.cfg.n_val,
                "n_test": self.cfg.n_test,
                "label_noise": self.cfg.label_noise,
                "latent_rank": self.cfg.latent_rank,
                "prototype_scale": self.cfg.prototype_scale,
                "mean_shift_rho": self.cfg.mean_shift_rho,
                "rotate_inputs": self.cfg.rotate_inputs,
                "input_task_id": self.cfg.input_task_id,
                "global_classes": self.cfg.n_tasks * self.cfg.n_classes,
                "specialist_from": self.cfg.specialist_from,
                "freeze_trunk": self.cfg.freeze_trunk,
                "specialist_block_loss": self.cfg.specialist_block_loss,
                "base_pretext": self.cfg.base_pretext,
                "hidden_dims": list(self.cfg.hidden_dims),
                "base_epochs": self.cfg.base_epochs,
                "ft_epochs": self.cfg.ft_epochs,
                "joint_epochs": self.cfg.joint_epochs,
                "lr": self.cfg.lr,
                "lr_schedule": self.cfg.lr_schedule,
                "batch_size": self.cfg.batch_size,
            },
            "seeds": seeds,
            "methods": methods,
            "gates": gates,
            "gates_pass": bool(all(g["pass"] for g in gates.values())),
            "ablations": ablations,
            "failure_cases": failure_cases[:3],
            "barriers": {
                "base_to_specialist_mean": float(
                    np.mean([b for r in results for b in r.barrier_base_to_specialist.values()])
                ),
                "specialist_pair_mean": float(
                    np.mean([r.barrier_specialist_pair for r in results])
                ),
            },
            "elapsed_sec": round(time.perf_counter() - t0, 3),
        }
        return report

    # ---- 确定性校验 ----
    def determinism_check(self, seed: int) -> dict:
        """同 seed 两次运行，比较核心指标（逐位一致才算确定性）。"""
        r1 = self.run_seed(seed)
        r2 = self.run_seed(seed)
        diffs: dict[str, float] = {}
        for name in ("uniform_soup", "mergefuse", "ties", "task_arithmetic", "dare", "greedy_soup"):
            for k, acc in r1.merged[name].per_task_acc.items():
                diffs[f"{name}_task{k}"] = abs(acc - r2.merged[name].per_task_acc[k])
        diffs["base_mean"] = abs(r1.base.mean_acc - r2.base.mean_acc)
        diffs["ensemble_mean"] = abs(r1.ensemble.mean_acc - r2.ensemble.mean_acc)
        max_diff = max(diffs.values())
        return {
            "seed": seed,
            "max_abs_diff": float(max_diff),
            "bit_identical": bool(max_diff == 0.0),
            "n_compared": len(diffs),
        }
