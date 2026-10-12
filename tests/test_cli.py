"""CLI 冒烟：version / run / benchmark（含 ENV 覆盖小配置）。"""

from __future__ import annotations

import json

import pytest

from cli import main


def test_version(capsys):
    assert main(["version"]) == 0
    assert "mergeforge" in capsys.readouterr().out


def test_run_single_seed(monkeypatch, capsys):
    _set_tiny_env(monkeypatch)
    assert main(["run", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert "mergefuse" in out
    assert "线性插值障碍" in out


def test_benchmark_writes_report(monkeypatch, tmp_path, capsys):
    _set_tiny_env(monkeypatch)
    out_file = tmp_path / "bench.json"
    rc = main(["benchmark", "--seeds", "7", "--out", str(out_file)])
    # 小配置下门禁可能不过（规模效应），但报告必须完整落盘
    assert rc in (0, 1)
    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["system"] == "mergeforge"
    assert report["gates_pass"] in (True, False)
    assert "elapsed_sec" in report
    capsys.readouterr()


def test_benchmark_with_determinism(monkeypatch, tmp_path, capsys):
    _set_tiny_env(monkeypatch)
    out_file = tmp_path / "bench_det.json"
    assert main(["benchmark", "--seeds", "7", "--out", str(out_file), "--determinism"]) in (0, 1)
    report = json.loads(out_file.read_text(encoding="utf-8"))
    assert report["determinism"]["bit_identical"] is True


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main(["bogus-command"])


def _set_tiny_env(monkeypatch):
    """把规范配置压成小配置（CI/本地冒烟用，不动默认值）。"""
    env = {
        "MERGEFORGE_D": "8",
        "MERGEFORGE_N_TASKS": "3",
        "MERGEFORGE_N_TRAIN": "64",
        "MERGEFORGE_N_VAL": "32",
        "MERGEFORGE_N_TEST": "48",
        "MERGEFORGE_BASE_EPOCHS": "3",
        "MERGEFORGE_FT_EPOCHS": "3",
        "MERGEFORGE_JOINT_EPOCHS": "3",
        "MERGEFORGE_HIDDEN_DIMS": "(8, 8)",
        "MERGEFORGE_SEEDS": "(7,)",
        "MERGEFORGE_TIES_RHO_GRID": "(0.5,)",
        "MERGEFORGE_LAYER_LAMBDA_GRID": "(0.5, 1.0)",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
