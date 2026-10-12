"""共享测试夹具：小配置 + 小模型（保证整个测试套件秒级完成）。"""

from __future__ import annotations

import pytest

from core.config import MergeConfig
from core.seed import make_rng
from models.mlp import MLP


@pytest.fixture
def tiny_cfg() -> MergeConfig:
    """端到端小配置（结构同规范配置，规模缩小）。"""
    return MergeConfig(
        d=8,
        n_tasks=3,
        n_classes=4,
        n_train=64,
        n_val=32,
        n_test=48,
        label_noise=0.05,
        latent_rank=2,
        prototype_scale=2.0,
        mean_shift_rho=1.0,
        hidden_dims=(8, 8),
        base_epochs=3,
        ft_epochs=3,
        joint_epochs=3,
        batch_size=4096,
        seeds=(7,),
    )


@pytest.fixture
def small_model() -> MLP:
    return MLP(input_dim=6, hidden_dims=(8,), output_dim=3, rng=make_rng(0))


@pytest.fixture
def small_data():
    rng = make_rng(1)
    X = rng.normal(0.0, 1.0, size=(32, 6))
    y = rng.randint(0, 3, size=32)
    return X, y


@pytest.fixture
def merge_triple(small_model):
    """base + 3 个 specialist 扁平参数 + spec（合并测试专用）。"""
    rng = make_rng(2)
    base_flat = small_model.get_flat()
    flats = [base_flat + rng.normal(0.0, 0.05, size=base_flat.shape) for _ in range(3)]
    return base_flat, flats, small_model.param_spec
