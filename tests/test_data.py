"""数据 DGP：可复现性 + 结构不变量（标签噪声机制逐位重放验证）。"""

from __future__ import annotations

import numpy as np

from core.seed import make_rng
from data.tasks import _orthogonal, _sample_split, add_task_id, make_multitask_dataset


def test_dataset_reproducible_bitwise():
    kwargs = {
        "d": 8,
        "n_tasks": 3,
        "n_classes": 4,
        "n_train": 64,
        "n_val": 32,
        "n_test": 48,
        "label_noise": 0.05,
        "latent_rank": 2,
        "prototype_scale": 2.0,
        "mean_shift_rho": 1.0,
        "rotate_inputs": False,
        "input_task_id": False,
    }
    t1 = make_multitask_dataset(seed=7, **kwargs)
    t2 = make_multitask_dataset(seed=7, **kwargs)
    for k in t1:
        for split in ("train", "val", "test"):
            assert np.array_equal(getattr(t1[k], f"X_{split}"), getattr(t2[k], f"X_{split}")), (
                f"task {k} {split} 不可复现"
            )


def test_dataset_varies_with_seed():
    kwargs = {
        "d": 8,
        "n_tasks": 2,
        "n_classes": 3,
        "n_train": 32,
        "n_val": 16,
        "n_test": 16,
        "label_noise": 0.0,
        "latent_rank": 2,
        "prototype_scale": 2.0,
        "mean_shift_rho": 1.0,
        "rotate_inputs": False,
        "input_task_id": False,
    }
    t1 = make_multitask_dataset(seed=7, **kwargs)
    t2 = make_multitask_dataset(seed=8, **kwargs)
    assert not np.array_equal(t1[0].X_train, t2[0].X_train)


def test_dataset_shapes_and_label_range():
    """全局标签空间 K×C：任务 k 的标签严格落在私有块 [k·C, (k+1)·C) 内。"""
    tasks = make_multitask_dataset(
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
        rotate_inputs=False,
        input_task_id=False,
        seed=7,
    )
    for k, t in tasks.items():
        assert t.task_id == k
        assert t.X_train.shape == (64, 8) and t.X_val.shape == (32, 8) and t.X_test.shape == (48, 8)
        for y in (t.y_train, t.y_val, t.y_test):
            assert y.min() >= k * 4 and y.max() < (k + 1) * 4


def test_label_noise_mechanism_replay():
    """用独立同序 rng 重放 _sample_split 的随机流，逐位复现标签噪声。"""
    rng_kwargs = {"n": 512, "label_noise": 0.2, "n_classes": 4}
    M = make_rng(100).normal(0.0, 1.0, size=(8, 2))
    V = make_rng(101).normal(0.0, 1.0, size=(2, 4)) * 2.0
    mu = np.zeros(8)
    rng_a = make_rng(202)
    _X, y = _sample_split(rng_a, M=M, V=V, mu=mu, R=None, class_offset=0, **rng_kwargs)
    # 重放：z -> flip -> randint 的顺序必须与实现一致
    rng_b = make_rng(202)
    z = rng_b.normal(0.0, 1.0, size=(512, M.shape[1]))
    y_clean = (z @ V).argmax(axis=1)
    flip = rng_b.random(512) < 0.2
    n_flip = int(flip.sum())
    new_y = rng_b.randint(0, 3, size=n_flip)
    new_y = np.where(new_y >= y_clean[flip], new_y + 1, new_y)
    y_expected = y_clean.copy()
    y_expected[flip] = new_y
    assert np.array_equal(y, y_expected)


def test_label_noise_rate_approximates_target():
    """噪声率统计逼近设定值（固定种子，5σ 容差）。"""
    M = make_rng(110).normal(0.0, 1.0, size=(8, 2))
    V = make_rng(111).normal(0.0, 1.0, size=(2, 4)) * 2.0
    rng = make_rng(303)
    _, y = _sample_split(
        rng, n=4000, M=M, V=V, mu=np.zeros(8), R=None, label_noise=0.1, n_classes=4, class_offset=0
    )
    # 干净标签可由同序重放得到
    rng2 = make_rng(303)
    z = rng2.normal(0.0, 1.0, size=(4000, 2))
    y_clean = (z @ V).argmax(axis=1)
    rate = float((y != y_clean).mean())
    assert abs(rate - 0.1) < 0.02, f"标签噪声率 {rate} 偏离 0.1"


def test_mean_shift_applied():
    """均值平移使任务质心 ≈ μ（E[o]=0）。"""
    M = make_rng(120).normal(0.0, 1.0, size=(8, 2))
    V = make_rng(121).normal(0.0, 1.0, size=(2, 3)) * 2.0
    mu = np.zeros(8)
    mu[0] = 2.0
    rng = make_rng(404)
    X, _ = _sample_split(
        rng, n=4000, M=M, V=V, mu=mu, R=None, label_noise=0.0, n_classes=3, class_offset=0
    )
    assert np.allclose(X.mean(axis=0), mu, atol=0.15)


def test_orthogonal_rotation_is_orthogonal():
    R = _orthogonal(make_rng(55), 8)
    assert np.allclose(R.T @ R, np.eye(8), atol=1e-10)


def test_add_task_id():
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = add_task_id(X, task_id=2, n_tasks=5)
    assert out.shape == (4, 8)
    assert np.array_equal(out[:, :3], X)
    assert np.array_equal(out[:, 3:], np.tile(np.eye(5, dtype=np.float64)[2], (4, 1)))


def test_input_task_id_flag_changes_width(tiny_cfg):
    """regime B：任务 ID 拼入输入，模型输入宽度 = d + K。"""
    cfg = tiny_cfg
    cfg.input_task_id = True
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
        input_task_id=True,
        seed=7,
    )
    assert tasks[0].X_train.shape[1] == cfg.d  # 数据本身不带 ID
    assert add_task_id(tasks[0].X_train, 0, cfg.n_tasks).shape[1] == cfg.d + cfg.n_tasks
