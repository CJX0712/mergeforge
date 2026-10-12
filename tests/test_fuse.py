"""旗舰 MergeFuse：非劣守护 + 逐张量 λ 坐标下降行为。"""

from __future__ import annotations

import numpy as np

from core.config import MergeConfig
from core.seed import make_rng
from merging.fuse import merge_fuse
from merging.methods import task_vectors, ties_merge, uniform_soup


def _fuse_cfg() -> MergeConfig:
    return MergeConfig(
        ties_rho_grid=(0.5,),
        layer_lambda_grid=(0.0, 0.5, 1.0),
    )


def test_fuse_constant_eval_equals_ties(merge_triple):
    """常数评分：坐标下降无改进 → 结果 ≡ TIES(ρ=网格首值)。"""
    base_flat, flats, spec = merge_triple
    cfg = _fuse_cfg()
    flat, diag = merge_fuse(base_flat, flats, spec, lambda f: 0.5, cfg)
    deltas = task_vectors(base_flat, flats)
    assert np.allclose(flat, ties_merge(base_flat, deltas, rho=0.5, lam=1.0))
    assert diag["guard_triggered"] is False
    assert diag["rho"] == 0.5
    assert set(diag["lambdas"]) == set(spec.names)


def test_fuse_guard_falls_back_to_soup(merge_triple):
    """评分函数只奖励 soup 本身时，非劣守护必须回退 soup。"""
    base_flat, flats, spec = merge_triple
    soup = uniform_soup(base_flat, flats, spec)

    def eval_fn(flat):
        return -float(np.sum((flat - soup) ** 2))

    flat, diag = merge_fuse(base_flat, flats, spec, eval_fn, _fuse_cfg())
    assert np.array_equal(flat, soup)
    assert diag["guard_triggered"] is True
    assert diag["val_acc"] == diag["soup_val_acc"]


def test_fuse_never_below_soup(merge_triple):
    """任意评分下旗舰验证分不得低于 uniform soup（守护不变量）。"""
    base_flat, flats, spec = merge_triple
    rng = make_rng(23)
    target = base_flat + rng.normal(0.0, 0.05, size=base_flat.shape)

    def eval_fn(flat):
        return -float(np.sum((flat - target) ** 2))

    soup = uniform_soup(base_flat, flats, spec)
    soup_acc = eval_fn(soup)
    _, diag = merge_fuse(base_flat, flats, spec, eval_fn, _fuse_cfg())
    assert diag["val_acc"] >= soup_acc - 1e-12


def test_fuse_diagnostics_complete(merge_triple):
    base_flat, flats, spec = merge_triple
    cfg = _fuse_cfg()
    _, diag = merge_fuse(base_flat, flats, spec, lambda f: 0.5, cfg)
    for key in (
        "guard_triggered",
        "rho",
        "lambdas",
        "val_acc",
        "ties_val_acc",
        "soup_val_acc",
        "n_evals",
    ):
        assert key in diag
    assert diag["n_evals"] >= len(cfg.ties_rho_grid)
    assert all(np.isfinite(v) for v in diag["lambdas"].values())


def test_fuse_lambda_grid_changes_result(merge_triple):
    """逐张量 λ 自由度必须真实生效：网格越细，验证分不低于粗网格。"""
    base_flat, flats, spec = merge_triple
    rng = make_rng(29)
    target = base_flat + rng.normal(0.0, 0.03, size=base_flat.shape)

    def eval_fn(flat):
        return -float(np.sum((flat - target) ** 2))

    coarse = MergeConfig(ties_rho_grid=(0.5,), layer_lambda_grid=(1.0,))
    fine = MergeConfig(ties_rho_grid=(0.5,), layer_lambda_grid=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25))
    _, diag_c = merge_fuse(base_flat, flats, spec, eval_fn, coarse)
    _, diag_f = merge_fuse(base_flat, flats, spec, eval_fn, fine)
    assert diag_f["val_acc"] >= diag_c["val_acc"] - 1e-12
