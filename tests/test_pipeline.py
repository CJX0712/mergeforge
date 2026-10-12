"""流水线：小配置端到端 + 确定性逐位一致（同 seed 两次运行）。"""

from __future__ import annotations

import pytest

from pipeline.merge_pipeline import CORE_METRIC_METHODS, MergePipeline


def test_run_seed_end_to_end(tiny_cfg):
    pipe = MergePipeline(tiny_cfg)
    result = pipe.run_seed(7)
    assert result.seed == 7
    assert result.joint is not None
    assert len(result.specialists) == tiny_cfg.n_tasks
    for name in ("uniform_soup", "greedy_soup", "task_arithmetic", "ties", "dare", "mergefuse"):
        assert name in result.merged
        assert 0.0 <= result.merged[name].mean_acc <= 1.0
    assert result.ensemble.n_models_used == tiny_cfg.n_tasks
    assert len(result.barrier_base_to_specialist) == tiny_cfg.n_tasks
    assert result.elapsed_sec > 0


def test_run_seed_bitwise_deterministic(tiny_cfg):
    """同 seed 两次运行：全部合并方法逐任务准确率逐位一致。"""
    pipe = MergePipeline(tiny_cfg)
    r1 = pipe.run_seed(7)
    r2 = pipe.run_seed(7)
    for name in r1.merged:
        assert r1.merged[name].per_task_acc == r2.merged[name].per_task_acc, name
    assert r1.base.per_task_acc == r2.base.per_task_acc
    assert r1.ensemble.per_task_acc == r2.ensemble.per_task_acc
    assert r1.barrier_specialist_pair == r2.barrier_specialist_pair


def test_determinism_check_reports_bit_identical(tiny_cfg):
    pipe = MergePipeline(tiny_cfg)
    det = pipe.determinism_check(7)
    assert det["bit_identical"] is True
    assert det["max_abs_diff"] == 0.0
    assert det["n_compared"] > 0


def test_benchmark_report_structure(tiny_cfg):
    tiny_cfg.seeds = (7, 11)
    pipe = MergePipeline(tiny_cfg)
    report = pipe.benchmark([7, 11])
    assert set(report["methods"]) == set(CORE_METRIC_METHODS)
    for m in report["methods"].values():
        assert len(m["per_seed"]) == 2
        assert m["std"] >= 0.0
    for gate in (
        "G1_fuse_vs_soup",
        "G2_fuse_vs_base",
        "G3_fuse_vs_specialist",
        "G4_fuse_vs_greedy",
        "G5_joint_reference",
        "G6_ensemble_reference",
    ):
        assert gate in report["gates"]
    assert isinstance(report["gates_pass"], bool)
    assert "guard_triggered_runs" in report["ablations"]
    assert 0 < len(report["failure_cases"]) <= 3
    for case in report["failure_cases"]:
        assert case["attribution"]
    assert report["barriers"]["base_to_specialist_mean"] >= 0.0
    assert report["author"] == "晨星"


def test_g1_gate_logic(tiny_cfg):
    """门禁判据直接单测：配对差判据 + 多数胜出。"""
    pipe = MergePipeline(tiny_cfg)
    tiny_cfg.seeds = (7, 11, 23)
    report = pipe.benchmark([7, 11, 23])
    g1 = report["gates"]["G1_fuse_vs_soup"]
    assert g1["n_seeds"] == 3
    assert g1["paired_delta_mean"] == pytest.approx(g1["delta"], abs=1e-12)
    assert g1["paired_significant"] == (g1["paired_delta_mean"] > 0.5 * g1["paired_delta_std"])
    assert g1["wins"] <= 3
