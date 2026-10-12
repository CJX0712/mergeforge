"""MergeForge 错误目录（E100~E500）。

两类同名错误会互相静默遮蔽（ValuaForge 实测：E103 在 ERROR_CATALOG 里但
``error_class("E103")`` 抛 KeyError）。因此注册表同时检测**重码与重名**，
撞名即 RuntimeError 立即炸，导入期零信号的问题改为导入期暴露。
"""

from __future__ import annotations


class MergeForgeError(Exception):
    """所有 MergeForge 错误的基类。"""

    code: str = "E000"


class DataError(MergeForgeError):
    """E100 系列：数据生成/切分/形状问题。"""

    code = "E100"


class ConfigError(MergeForgeError):
    """E200 系列：配置非法 / ENV 覆盖类型错误。"""

    code = "E200"


class ModelError(MergeForgeError):
    """E300 系列：模型前向/反向/参数形状问题。"""

    code = "E300"


class MergeError(MergeForgeError):
    """E400 系列：合并方法输入/参数问题。"""

    code = "E400"


class PipelineError(MergeForgeError):
    """E500 系列：流水线编排/基准问题。"""

    code = "E500"


ERROR_CATALOG: dict[str, type[MergeForgeError]] = {
    "E100": DataError,
    "E200": ConfigError,
    "E300": ModelError,
    "E400": MergeError,
    "E500": PipelineError,
}


def _discover_registry() -> None:
    """检测重码与重名（撞名比撞码更隐蔽：后定义者静默遮蔽前者）。"""
    seen_names: dict[str, str] = {}
    for code, cls in ERROR_CATALOG.items():
        if cls.__name__ in seen_names:
            raise RuntimeError(
                f"错误类重名: {cls.__name__} 同时绑定 {seen_names[cls.__name__]} 与 {code}"
            )
        seen_names[cls.__name__] = code
    if len(set(ERROR_CATALOG.values())) != len(ERROR_CATALOG):
        raise RuntimeError("ERROR_CATALOG 存在重复类绑定")


_discover_registry()


def error_class(code: str) -> type[MergeForgeError]:
    """按错误码取类；未知码抛 KeyError（带可用码列表）。"""
    try:
        return ERROR_CATALOG[code]
    except KeyError:
        raise KeyError(f"未知错误码 {code!r}，可用: {sorted(ERROR_CATALOG)}") from None
