"""评测指标：统一口径 + 恒等式（集成/障碍/恢复率）。"""

from __future__ import annotations

import numpy as np
import pytest

from core.seed import make_rng
from eval.metrics import (
    accuracy,
    ensemble_per_task_accuracies,
    flat_to_model,
    linear_barrier,
    mean_task_accuracy,
    per_task_accuracies,
    recovery_ratio,
)
from models.mlp import MLP


class _ConstClassifier:
    """永远预测固定类别的桩模型（duck-typed 过 accuracy）。"""

    def __init__(self, cls: int):
        self.cls = cls

    def predict(self, X):
        return np.full(X.shape[0], self.cls, dtype=np.int64)


def test_accuracy_matches_manual():
    y = np.array([0, 1, 1, 2, 2, 2])
    X = np.zeros((6, 3))
    assert accuracy(_ConstClassifier(1), X, y) == pytest.approx(2 / 6)


def test_per_task_covers_all_tasks(tiny_cfg):
    from data.tasks import make_multitask_dataset

    cfg = tiny_cfg
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
        seed=7,
    )
    model = MLP(
        input_dim=cfg.d, hidden_dims=cfg.hidden_dims, output_dim=cfg.n_classes, rng=make_rng(0)
    )
    per = per_task_accuracies(model, tasks, cfg, "test")
    assert sorted(per) == sorted(tasks)
    assert all(0.0 <= v <= 1.0 for v in per.values())
    assert mean_task_accuracy(model, tasks, cfg, "test") == pytest.approx(
        np.mean(list(per.values()))
    )


def test_ensemble_of_identical_models_equals_single(tiny_cfg):
    from data.tasks import make_multitask_dataset

    cfg = tiny_cfg
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
        seed=7,
    )
    model = MLP(
        input_dim=cfg.d, hidden_dims=cfg.hidden_dims, output_dim=cfg.n_classes, rng=make_rng(1)
    )
    single = per_task_accuracies(model, tasks, cfg, "test")
    ens = ensemble_per_task_accuracies([model, model, model], tasks, cfg, "test")
    for k in single:
        assert ens[k] == pytest.approx(single[k])


def test_linear_barrier_identical_flats_zero(tiny_cfg):
    from data.tasks import make_multitask_dataset

    cfg = tiny_cfg
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
        seed=7,
    )
    model = MLP(
        input_dim=cfg.d, hidden_dims=cfg.hidden_dims, output_dim=cfg.n_classes, rng=make_rng(2)
    )
    flat = model.get_flat()
    spec = model.param_spec
    res = linear_barrier(flat, flat, tasks, cfg, spec, model, n_points=5)
    assert res["barrier"] == pytest.approx(0.0, abs=1e-12)
    assert res["end_a"] == res["end_b"]
    assert len(res["accs"]) == 5


def test_linear_barrier_endpoints_match_models(tiny_cfg):
    """路径端点准确率必须等于两个模型自身的准确率。"""
    from data.tasks import make_multitask_dataset

    cfg = tiny_cfg
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
        seed=7,
    )
    m1 = MLP(
        input_dim=cfg.d, hidden_dims=cfg.hidden_dims, output_dim=cfg.n_classes, rng=make_rng(3)
    )
    m2 = m1.copy()
    m2.set_flat(m1.get_flat() + 0.05)
    res = linear_barrier(m1.get_flat(), m2.get_flat(), tasks, cfg, m1.param_spec, m1, n_points=5)
    assert res["end_a"] == pytest.approx(mean_task_accuracy(m1, tasks, cfg, "test"))
    assert res["end_b"] == pytest.approx(mean_task_accuracy(m2, tasks, cfg, "test"))


def test_recovery_ratio_identities():
    assert recovery_ratio(0.2, 0.2, 0.6) == pytest.approx(0.0)  # merged=base
    assert recovery_ratio(0.2, 0.6, 0.6) == pytest.approx(1.0)  # merged=ensemble
    assert recovery_ratio(0.2, 0.4, 0.6) == pytest.approx(0.5)  # 恢复一半
    assert np.isnan(recovery_ratio(0.2, 0.5, 0.2))  # 分母为 0


def test_flat_to_model_matches_source(tiny_cfg):
    """flat_to_model 还原出的模型必须与直接 set_flat 的模型逐位一致。"""
    template = MLP(
        input_dim=tiny_cfg.d,
        hidden_dims=tiny_cfg.hidden_dims,
        output_dim=tiny_cfg.n_classes,
        rng=make_rng(4),
    )
    flat = template.get_flat() + 0.3
    clone = flat_to_model(flat, template.param_spec, template)
    reference = template.copy()
    reference.set_flat(flat)
    X = make_rng(5).normal(0.0, 1.0, size=(16, tiny_cfg.d))
    assert np.array_equal(clone.forward(X), reference.forward(X))
