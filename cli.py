#!/usr/bin/env python3
"""MergeForge CLI。

用法：
    python cli.py run --seed 7                 # 单 seed 实验，打印逐任务表
    python cli.py benchmark                    # 多 seed 基准，写 benchmark.json
    python cli.py benchmark --determinism      # 附带同 seed 二次运行确定性校验
    python cli.py version

配置可用 ENV_MERGEFORGE_* 覆盖（如 MERGEFORGE_N_TASKS=3 MERGEFORGE_BASE_EPOCHS=5）。
"""

from __future__ import annotations

import argparse
import json
import sys

import core.thread_pin

core.thread_pin.pin_threads()

from core.config import MergeConfig  # noqa: E402
from pipeline.merge_pipeline import MergePipeline  # noqa: E402

_ICON = {"pass": "✅", "fail": "❌"}


def _fmt_row(name: str, m: dict) -> str:
    per_seed = ", ".join(f"{x:.4f}" for x in m["per_seed"])
    return f"{name:<18} {m['mean']:.4f} ± {m['std']:.4f}   per-seed: [{per_seed}]"


def cmd_run(args: argparse.Namespace) -> int:
    cfg = MergeConfig.from_env()
    pipe = MergePipeline(cfg)
    result = pipe.run_seed(args.seed)
    print(
        f"\n=== MergeForge 单 seed 运行 (seed={args.seed}, elapsed={result.elapsed_sec:.1f}s) ==="
    )
    print(f"base (不合并)            mean test acc = {result.base.mean_acc:.4f}")
    if result.joint is not None:
        print(f"joint 多任务参照         mean test acc = {result.joint.mean_acc:.4f}")
    oracle_val = result.oracle_ensemble.mean_acc if result.oracle_ensemble else float("nan")
    print(f"oracle 集成 (K模型+任务ID) mean test acc = {oracle_val:.4f}")
    print(f"朴素集成 (概率平均)       mean test acc = {result.ensemble.mean_acc:.4f}")
    spec_mean = float(sum(m.mean_acc for m in result.specialists) / len(result.specialists))
    print(f"单 specialist 跨任务均值 mean test acc = {spec_mean:.4f}")
    print("\n合并方法（单模型）:")
    for name in ("uniform_soup", "greedy_soup", "task_arithmetic", "ties", "dare", "mergefuse"):
        r = result.merged[name]
        print(f"  {name:<18} mean={r.mean_acc:.4f}  val={r.val_acc:.4f}  extra={r.extra}")
    print("\n逐任务 test acc (mergefuse):")
    for k, acc in sorted(result.merged["mergefuse"].per_task_acc.items()):
        sp = result.specialists[k].per_task_acc[k]
        print(
            f"  task {k}: mergefuse={acc:.4f}  specialist_own={sp:.4f}  forgetting={sp - acc:+.4f}"
        )
    barriers = {k: round(v, 4) for k, v in result.barrier_base_to_specialist.items()}
    print(f"\n线性插值障碍 base→specialist: {barriers}")
    print(f"线性插值障碍 specialist0→specialist1: {result.barrier_specialist_pair:.4f}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = MergeConfig.from_env()
    pipe = MergePipeline(cfg)
    seeds = args.seeds or list(cfg.seeds)
    report = pipe.benchmark(seeds)
    print(f"\n=== MergeForge 基准 (seeds={seeds}, elapsed={report['elapsed_sec']:.1f}s) ===")
    print(f"{'method':<18} {'mean±std':<24} per-seed")
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
        print(_fmt_row(name, report["methods"][name]))
    print("\n门禁（预注册）:")
    for gname, g in report["gates"].items():
        icon = _ICON["pass"] if g["pass"] else _ICON["fail"]
        detail = {k: v for k, v in g.items() if k not in ("pass",)}
        print(f"  {icon} {gname}: {detail}")
    print(f"\nablations: {report['ablations']}")
    print(f"barriers: {report['barriers']}")
    print("failure cases:")
    for c in report["failure_cases"]:
        print(f"  task {c['task']}: {c['attribution']} ({c})")
    if args.determinism:
        det = pipe.determinism_check(seeds[0])
        print(f"\ndeterminism: {det}")
        report["determinism"] = det
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入 {args.out}；gates_pass={report['gates_pass']}")
    return 0 if report["gates_pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mergeforge", description="世界顶级模型合并系统（作者：晨星）"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="单 seed 实验")
    p_run.add_argument("--seed", type=int, default=7)
    p_bench = sub.add_parser("benchmark", help="多 seed 基准")
    p_bench.add_argument("--seeds", type=int, nargs="*", default=None)
    p_bench.add_argument("--out", default="benchmark.json")
    p_bench.add_argument("--determinism", action="store_true")
    sub.add_parser("version", help="版本信息")
    args = parser.parse_args(argv)
    if args.command == "version":
        print("mergeforge 0.1.0 (author: 晨星)")
        return 0
    try:
        if args.command == "run":
            return cmd_run(args)
        return cmd_benchmark(args)
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
