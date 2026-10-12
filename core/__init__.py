"""MergeForge core: 线程钉死 / 随机种子 / 错误目录 / 配置 / 类型 / 接口。

导入任何 numpy 之前必须先 import 本包（``core/__init__`` 会钉死 BLAS 线程），
否则多线程 BLAS 的浮点累加顺序不固定，确定性门禁会偶发失败。
"""

from core.thread_pin import pin_threads

pin_threads()

__all__ = ["pin_threads"]
