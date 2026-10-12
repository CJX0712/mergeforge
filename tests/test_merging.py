"""合并方法代数恒等式（每种方法必须可被独立验证的不变量）。"""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import MergeError
from core.seed import make_rng
from merging.methods import (
    dare_merge,
    greedy_soup,
    task_arithmetic,
    task_vectors,
    tensor_slices,
    ties_merge,
    uniform_soup,
)


def test_task_vectors_identity(merge_triple):
    base_flat, flats, _ = merge_triple
    deltas = task_vectors(base_flat, flats)
    assert deltas.shape == (len(flats), base_flat.shape[0])
    for k, m in enumerate(flats):
        assert np.allclose(deltas[k], m - base_flat)


def test_tensor_slices_tile_full_vector(merge_triple):
    _, _, spec = merge_triple
    slices = tensor_slices(spec)
    covered = np.zeros(spec.n_params, dtype=bool)
    for sl in slices.values():
        assert not covered[sl].any(), "张量区间重叠"
        covered[sl] = True
    assert covered.all(), "张量区间未覆盖整个扁平向量"


def test_uniform_soup_equals_mean(merge_triple):
    base_flat, flats, spec = merge_triple
    soup = uniform_soup(base_flat, flats, spec)
    assert np.allclose(soup, np.mean(np.stack(flats), axis=0))


def test_uniform_soup_permutation_invariant(merge_triple):
    """置换不变性：成员顺序不影响均匀 soup（数学上恒等，数值容差 1e-12）。"""
    base_flat, flats, spec = merge_triple
    soup = uniform_soup(base_flat, flats, spec)
    perm = [2, 0, 1]
    soup_perm = uniform_soup(base_flat, [flats[i] for i in perm], spec)
    assert np.allclose(soup, soup_perm, rtol=0.0, atol=1e-12)


def test_task_arithmetic_lambda_zero_is_base(merge_triple):
    base_flat, flats, _ = merge_triple
    deltas = task_vectors(base_flat, flats)
    assert np.array_equal(task_arithmetic(base_flat, deltas, 0.0), base_flat)


def test_task_arithmetic_lambda_one(merge_triple):
    base_flat, flats, _ = merge_triple
    deltas = task_vectors(base_flat, flats)
    assert np.allclose(task_arithmetic(base_flat, deltas, 1.0), base_flat + deltas.sum(axis=0))


def test_ties_single_model_equals_task_arithmetic(merge_triple):
    """K=1 时 TIES 无冲突可解：trim(ρ=1) + 全 sign 同意 ≡ Task Arithmetic。"""
    base_flat, flats, _ = merge_triple
    deltas = task_vectors(base_flat, flats)[:1]
    ties = ties_merge(base_flat, deltas, rho=1.0, lam=1.0)
    ta = task_arithmetic(base_flat, deltas, 1.0)
    assert np.allclose(ties, ta)


def test_ties_unanimity_equals_scaled_task_arithmetic():
    """全模型符号一致时：disjoint mean = (1/K)ΣΔ，故 lam=K 时 ≡ TA(λ=1)。"""
    rng = make_rng(11)
    base = rng.normal(0.0, 1.0, size=64)
    # 构造全部同号的任务向量（叠加共同方向 + 小扰动保持同号）
    common = np.abs(rng.normal(1.0, 0.2, size=64))
    deltas = np.stack([common + rng.normal(0.0, 0.01, size=64) for _ in range(4)])
    ties = ties_merge(base, deltas, rho=1.0, lam=4.0)
    ta = task_arithmetic(base, deltas, 1.0)
    assert np.allclose(ties, ta, atol=1e-10)


def test_dare_delta_zero_equals_task_arithmetic(merge_triple):
    """δ=0（不丢弃）时 DARE 退化为 Task Arithmetic。"""
    base_flat, flats, _ = merge_triple
    deltas = task_vectors(base_flat, flats)
    dare = dare_merge(base_flat, deltas, delta=0.0, lam=0.7, rng=make_rng(0))
    ta = task_arithmetic(base_flat, deltas, 0.7)
    assert np.array_equal(dare, ta)


def test_dare_drop_rescale_is_unbiased():
    """drop & rescale 的期望保持：E[δ-scaled ΣΔ] = ΣΔ（固定种子下统计验证）。"""
    rng = make_rng(13)
    P, K, delta = 2000, 4, 0.5
    deltas = np.ones((K, P))  # 全 1 向量：期望总量 K*P
    total = dare_merge(np.zeros(P), deltas, delta=delta, lam=1.0, rng=rng).sum()
    expected = K * P
    std = np.sqrt(K * P * delta / (1.0 - delta))
    assert abs(total - expected) < 5.0 * std, f"DARE 期望保持失败: {total} vs {expected}"


def test_dare_respects_seed(merge_triple):
    """同 seed 的 DARE 掩码逐位可复现。"""
    base_flat, flats, _ = merge_triple
    deltas = task_vectors(base_flat, flats)
    a = dare_merge(base_flat, deltas, 0.4, 1.0, rng=make_rng(99))
    b = dare_merge(base_flat, deltas, 0.4, 1.0, rng=make_rng(99))
    assert np.array_equal(a, b)


def test_greedy_soup_returns_mean_of_selected(merge_triple):
    base_flat, flats, spec = merge_triple
    target = np.mean(np.stack(flats), axis=0) + 0.01

    def eval_fn(flat):
        return -float(np.sum((flat - target) ** 2))

    flat, selected = greedy_soup(base_flat, flats, spec, eval_fn)
    assert set(selected) <= set(range(len(flats)))
    assert np.allclose(flat, np.mean(np.stack([flats[i] for i in selected]), axis=0))


def test_greedy_soup_constant_eval_selects_first(merge_triple):
    """常数评分下无改进可做：只选排序首位（确定性 tie-break）。"""
    base_flat, flats, spec = merge_triple
    flat, selected = greedy_soup(base_flat, flats, spec, lambda f: 0.5)
    assert selected == [0]
    assert np.array_equal(flat, flats[0])


def test_greedy_soup_improves_with_more_members():
    """评分随均值趋近目标单调提升时，贪心应选满全部成员。"""
    rng = make_rng(17)
    flats = [rng.normal(0.0, 1.0, size=32) for _ in range(4)]
    target = np.mean(np.stack(flats), axis=0)

    def eval_fn(flat):
        return -float(np.sum((flat - target) ** 2))

    _, selected = greedy_soup(np.zeros(32), flats, None, eval_fn)
    assert len(selected) == 4


# ---- 输入校验 ----
def test_empty_model_flats_raises(merge_triple):
    base_flat, _, spec = merge_triple
    with pytest.raises(MergeError):
        uniform_soup(base_flat, [], spec)


def test_shape_mismatch_raises(merge_triple):
    base_flat, flats, spec = merge_triple
    bad = [*flats, np.zeros(base_flat.shape[0] + 1)]
    with pytest.raises(MergeError):
        uniform_soup(base_flat, bad, spec)


def test_ties_rho_range(merge_triple):
    base_flat, flats, _ = merge_triple
    deltas = task_vectors(base_flat, flats)
    with pytest.raises(MergeError):
        ties_merge(base_flat, deltas, rho=0.0)
    with pytest.raises(MergeError):
        ties_merge(base_flat, deltas, rho=1.5)


def test_dare_delta_range(merge_triple):
    base_flat, flats, _ = merge_triple
    deltas = task_vectors(base_flat, flats)
    with pytest.raises(MergeError):
        dare_merge(base_flat, deltas, delta=1.0)
    with pytest.raises(MergeError):
        dare_merge(base_flat, deltas, delta=-0.1)


def test_deltas_wrong_ndim_raises(merge_triple):
    base_flat, flats, _ = merge_triple
    with pytest.raises(MergeError):
        task_arithmetic(base_flat, flats[0], 1.0)
