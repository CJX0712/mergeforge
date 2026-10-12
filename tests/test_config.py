"""配置：ENV 覆盖类型转换（PEP 563 回归）+ schema 校验。"""

from __future__ import annotations

import pytest

from core.config import MergeConfig
from core.errors import ConfigError


def test_defaults_validate():
    cfg = MergeConfig()
    cfg.validate()  # 不抛即过


@pytest.mark.parametrize(
    "env_key,raw,attr,expected",
    [
        ("MERGEFORGE_N_TASKS", "3", "n_tasks", 3),
        ("MERGEFORGE_BASE_EPOCHS", "7", "base_epochs", 7),
        ("MERGEFORGE_LABEL_NOISE", "0.1", "label_noise", 0.1),
        ("MERGEFORGE_MEAN_SHIFT_RHO", "0.5", "mean_shift_rho", 0.5),
        ("MERGEFORGE_INPUT_TASK_ID", "true", "input_task_id", True),
        ("MERGEFORGE_INPUT_TASK_ID", "0", "input_task_id", False),
        ("MERGEFORGE_ROTATE_INPUTS", "yes", "rotate_inputs", True),
        ("MERGEFORGE_LR_SCHEDULE", "const", "lr_schedule", "const"),
    ],
)
def test_env_override_scalar_types(monkeypatch, env_key, raw, attr, expected):
    """ENV 覆盖必须转成正确类型——PEP 563 下 f.type 是字符串，
    name-based 映射是唯一可靠路径（MatrixForge 实测坑）。"""
    monkeypatch.setenv(env_key, raw)
    cfg = MergeConfig.from_env()
    value = getattr(cfg, attr)
    assert value == expected
    assert isinstance(value, type(expected)), f"{attr} 类型错误: {type(value)}"


def test_env_override_tuple_types(monkeypatch):
    monkeypatch.setenv("MERGEFORGE_SEEDS", "(1, 2, 3)")
    monkeypatch.setenv("MERGEFORGE_HIDDEN_DIMS", "(8, 8)")
    cfg = MergeConfig.from_env()
    assert cfg.seeds == (1, 2, 3)
    assert cfg.hidden_dims == (8, 8)
    assert all(isinstance(s, int) for s in cfg.seeds)


def test_env_override_bad_int_raises(monkeypatch):
    monkeypatch.setenv("MERGEFORGE_N_TASKS", "abc")
    with pytest.raises(ConfigError, match="无法解析"):
        MergeConfig.from_env()


def test_env_override_bad_literal_raises(monkeypatch):
    monkeypatch.setenv("MERGEFORGE_SEEDS", "1,2,3]")  # 语法非法的字面量
    with pytest.raises(ConfigError, match="无法解析"):
        MergeConfig.from_env()


def test_env_override_then_validate(monkeypatch):
    """覆盖后仍跑 schema 校验：非法值必须炸，不静默通过。"""
    monkeypatch.setenv("MERGEFORGE_N_TASKS", "1")
    with pytest.raises(ConfigError):
        MergeConfig.from_env()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_tasks": 1},
        {"n_classes": 1},
        {"n_train": 16},
        {"label_noise": 0.5},
        {"label_noise": -0.1},
        {"latent_rank": 0},
        {"mean_shift_rho": -1.0},
        {"joint_epochs": 0},
        {"base_epochs": 0},
        {"ft_epochs": 0},
        {"lr": 0.0},
        {"lr_schedule": "linear"},
        {"batch_size": 4},
        {"seeds": ()},
        {"ties_rho_grid": ()},
        {"ta_lambda_grid": (0.5, -0.1)},
        {"hidden_dims": ()},
        {"hidden_dims": (0, 8)},
    ],
)
def test_validate_rejects_bad_config(kwargs):
    with pytest.raises(ConfigError):
        MergeConfig(**kwargs).validate()


def test_validate_latent_rank_upper_bound():
    with pytest.raises(ConfigError):
        MergeConfig(d=4, latent_rank=5).validate()


def test_validate_task_id_regime_requires_identifiability():
    """无任务 ID 时，任务必须可从输入识别（ρ>0 或 rotate 至少其一）——
    否则合并在数学上无意义（DGP 设计不变量）。"""
    with pytest.raises(ConfigError, match="至少取其一"):
        MergeConfig(input_task_id=False, mean_shift_rho=0.0, rotate_inputs=False).validate()
    # 两种补救路径都放行
    MergeConfig(input_task_id=False, mean_shift_rho=0.5, rotate_inputs=False).validate()
    MergeConfig(input_task_id=False, mean_shift_rho=0.0, rotate_inputs=True).validate()
    # 有任务 ID 时无此约束
    MergeConfig(input_task_id=True, mean_shift_rho=0.0, rotate_inputs=False).validate()
