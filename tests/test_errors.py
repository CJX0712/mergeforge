"""错误目录：查找、编码、重码/重名双检（ValuaForge 静默遮蔽回归）。"""

from __future__ import annotations

import pytest

from core import errors
from core.errors import (
    ERROR_CATALOG,
    ConfigError,
    DataError,
    MergeError,
    MergeForgeError,
    ModelError,
    PipelineError,
    error_class,
)


def test_error_class_lookup():
    assert error_class("E100") is DataError
    assert error_class("E200") is ConfigError
    assert error_class("E300") is ModelError
    assert error_class("E400") is MergeError
    assert error_class("E500") is PipelineError


def test_error_class_unknown_code():
    with pytest.raises(KeyError, match="未知错误码"):
        error_class("E999")


def test_all_errors_share_base():
    for cls in ERROR_CATALOG.values():
        assert issubclass(cls, MergeForgeError)


def test_error_codes_match_catalog():
    for code, cls in ERROR_CATALOG.items():
        assert cls.code == code


def test_duplicate_binding_detected(monkeypatch):
    """同码绑同类必须先炸（重名检查先于重复绑定检查触发，两者都拦）。"""
    monkeypatch.setattr(errors, "ERROR_CATALOG", {"E100": DataError, "E101": DataError})
    with pytest.raises(RuntimeError, match=r"重名|重复类绑定"):
        errors._discover_registry()


def test_duplicate_name_detected(monkeypatch):
    """重名（两个类同名）比重码更隐蔽：后定义者会静默遮蔽前者。"""
    twin = type("DataError", (MergeForgeError,), {"code": "E101"})
    monkeypatch.setattr(errors, "ERROR_CATALOG", {"E100": DataError, "E101": twin})
    with pytest.raises(RuntimeError, match="重名"):
        errors._discover_registry()


def test_registry_is_clean_at_import():
    """当前注册表本身必须通过双检（导入期已跑过一次，这里显式再跑）。"""
    errors._discover_registry()  # 不抛即过
