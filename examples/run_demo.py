#!/usr/bin/env python3
"""MergeForge 端到端演示：合成多任务数据 → base 预训练 → K 个 specialist
微调 → 7 种合并方法基准（3 seeds，mean±std）→ 门禁评测 → 落盘 benchmark.json。

用法：
    python examples/run_demo.py                # 规范基准（默认 3 seeds）
    python examples/run_demo.py --determinism  # 附带同 seed 二次运行逐位校验
    MERGEFORGE_N_TASKS=3 MERGEFORGE_BASE_EPOCHS=5 python examples/run_demo.py   # 冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.thread_pin

core.thread_pin.pin_threads()

from core.config import MergeConfig  # noqa: E402
from pipeline.merge_pipeline import MergePipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MergeForge demo")
    parser.add_argument("--out", default="benchmark.json")
    parser.add_argument("--determinism", action="store_true")
    args = parser.parse_args()

    cfg = MergeConfig.from_env()
    pipe = MergePipeline(cfg)
    report = pipe.benchmark(list(cfg.seeds))

    print("=" * 72)
    print(" MergeForge · 世界顶级模型合并系统（作者：晨星）")
    print(f" 任务数 K={cfg.n_tasks} · 维度 d={cfg.d} · 每任务 train={cfg.n_train}")
    print(
        f" base: {cfg.base_epochs} epochs · specialist: {cfg.ft_epochs} epochs"
        f" · seeds={list(cfg.seeds)}"
    )
    print("=" * 72)
    header = f"{'method':<20}{'mean±std':<22}{'per-seed'}"
    print(header)
    print("-" * 72)
    for name in (
        "base",
        "joint",
        "specialist_mean",
        "uniform_soup",
        "greedy_soup",
        "task_arithmetic",
        "ties",
        "dare",
        "mergefuse",
        "ensemble",
        "oracle_ensemble",
    ):
        m = report["methods"][name]
        per_seed = "/".join(f"{x:.4f}" for x in m["per_seed"])
        print(f"{name:<20}{m['mean']:.4f} ± {m['std']:.4f}   {per_seed}")
    print("-" * 72)
    print("门禁（预注册，不因结果迁就）:")
    for gname, g in report["gates"].items():
        icon = "✅" if g["pass"] else "❌"
        detail = {k: v for k, v in g.items() if k != "pass"}
        print(f"  {icon} {gname}: {detail}")
    print(f"\nablations: {report['ablations']}")
    print(f"barriers: {report['barriers']}")
    print("失败案例（从 results 派生）:")
    for c in report["failure_cases"]:
        print(
            f"  task {c['task']}: specialist_own={c['specialist_own']:.4f} "
            f"mergefuse={c['mergefuse']:.4f} soup={c['uniform_soup']:.4f} "
            f"ensemble={c['ensemble']:.4f} → {c['attribution']}"
        )
    if args.determinism:
        det = pipe.determinism_check(cfg.seeds[0])
        report["determinism"] = det
        print(f"\ndeterminism check: {det}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nbenchmark.json 已写入 {args.out}（elapsed={report['elapsed_sec']:.1f}s）")
    print(f"gates_pass={report['gates_pass']}")
    return 0 if report["gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
