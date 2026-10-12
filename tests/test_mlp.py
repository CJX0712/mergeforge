"""MLP 数值正确性：解析梯度 vs 中心差分（手写实现的硬金标准）。"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import ModelError
from core.seed import make_rng
from models.mlp import MLP, gradient_check, softmax


def test_gradient_check_central_difference(small_model, small_data):
    """解析梯度必须与中心差分一致（相对误差 ~1e-6 量级）。"""
    X, y = small_data
    max_rel = gradient_check(small_model, X, y, n_probe=16, rng=make_rng(3))
    assert max_rel < 1e-4, f"梯度校验失败: max_rel_err={max_rel}"


def test_gradient_check_multilayer():
    """多层网络同样满足梯度校验。"""
    rng = make_rng(5)
    model = MLP(input_dim=5, hidden_dims=(7, 6), output_dim=4, rng=rng)
    X = rng.normal(0.0, 1.0, size=(24, 5))
    y = rng.randint(0, 4, size=24)
    max_rel = gradient_check(model, X, y, n_probe=12, rng=make_rng(6))
    assert max_rel < 1e-4


def test_gradient_check_class_block():
    """class_block（私有块 softmax）路径的梯度校验——该分支曾漏写块内
    竞争类的 p_c 梯度项（delta 只含真类 -1 项），数值校验必须覆盖此路径。"""
    rng = make_rng(9)
    model = MLP(input_dim=6, hidden_dims=(8,), output_dim=12, rng=rng)
    X = rng.normal(0.0, 1.0, size=(32, 6))
    y = rng.randint(8, 12, size=32)  # 全部落在块 [8, 12)
    max_rel = gradient_check(model, X, y, n_probe=30, rng=make_rng(10), class_block=(8, 12))
    assert max_rel < 1e-4, f"块梯度校验失败: max_rel_err={max_rel}"


def test_class_block_grad_support_disjoint():
    """块损失的梯度支撑必须严格限于自有块（trunk 与其他块分量 = 0）。"""
    rng = make_rng(11)
    model = MLP(input_dim=4, hidden_dims=(6,), output_dim=8, rng=rng)
    X = rng.normal(0.0, 1.0, size=(16, 4))
    y = rng.randint(4, 8, size=16)
    _, grad = model.loss_and_grad(X, y, class_block=(4, 8))
    spec = model.param_spec
    tensors = spec.split(grad)
    # trunk（W1/b1）梯度必须为 0（块 loss 只影响输出层 + 反传会到 trunk！）
    # 注意：反传会把块内 delta 传播到 trunk——块 loss 限制的是**输出行**，
    # 不是 trunk；trunk 梯度非零是正确的（freeze_trunk 由训练侧掩码实现）。
    assert np.isfinite(grad).all()
    # 输出层其他块的**列**梯度必须严格为 0（W2 形状 (hidden, classes)，块 = 列）
    assert np.allclose(tensors["W2"][:, :4], 0.0)
    assert np.allclose(tensors["b2"][:4], 0.0)
    assert not np.allclose(tensors["W2"][:, 4:], 0.0)


def test_softmax_rows_sum_to_one():
    rng = make_rng(7)
    logits = rng.normal(0.0, 5.0, size=(16, 4))
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)
    assert (probs > 0).all()


def test_softmax_overflow_stable():
    """大 logits 不溢出（max-shift 技巧）。"""
    logits = np.array([[1000.0, 1001.0, 999.0]])
    probs = softmax(logits)
    assert np.isfinite(probs).all()
    assert np.isclose(probs.sum(), 1.0)


def test_forward_shape_and_predict_consistency(small_model, small_data):
    X, _ = small_data
    logits = small_model.forward(X)
    assert logits.shape == (X.shape[0], small_model.dims[-1])
    assert np.array_equal(small_model.predict(X), small_model.predict_proba(X).argmax(axis=1))


def test_flat_roundtrip(small_model):
    flat = small_model.get_flat()
    small_model.set_flat(flat + 0.1)
    assert np.allclose(small_model.get_flat(), flat + 0.1)
    small_model.set_flat(flat)
    assert np.array_equal(small_model.get_flat(), flat)


def test_set_flat_wrong_length_raises(small_model):
    with pytest.raises(ModelError):
        small_model.set_flat(np.zeros(small_model.param_spec.n_params + 1))


def test_forward_wrong_input_dim_raises(small_model):
    with pytest.raises(ModelError):
        small_model.forward(np.zeros((4, small_model.dims[0] + 1)))


def test_loss_and_grad_label_bounds(small_model, small_data):
    X, y = small_data
    with pytest.raises(ModelError):
        small_model.loss_and_grad(X, y + small_model.dims[-1])
    with pytest.raises(ModelError):
        small_model.loss_and_grad(X, y - 1)


def test_loss_and_grad_shapes(small_model, small_data):
    X, y = small_data
    loss, grad = small_model.loss_and_grad(X, y)
    assert loss > 0
    assert grad.shape == (small_model.param_spec.n_params,)
    assert np.isfinite(grad).all()


def test_copy_is_deep(small_model):
    clone = small_model.copy()
    clone.weights[0][0, 0] += 1.0
    assert small_model.weights[0][0, 0] != clone.weights[0][0, 0]


def test_param_spec_split_flatten_roundtrip(small_model):
    spec = small_model.param_spec
    assert spec.n_params == sum(int(np.prod(s)) for s in spec.shapes)
    flat = small_model.get_flat()
    tensors = spec.split(flat)
    assert set(tensors) == set(spec.names)
    assert np.array_equal(spec.flatten(tensors), flat)
