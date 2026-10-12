"""MergeForge 配置（ENV_MERGEFORGE_* 覆盖 + schema 校验）。

坑（MatrixForge 实测）：``from __future__ import annotations`` 使
``f.type`` 为字符串，``target is int`` 永不命中，env 值残留字符串。
因此按**字段名字符串映射**转换，并为每个 int 字段配「覆盖后类型断言」单测。
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass

from core.errors import ConfigError

_ENV_PREFIX = "MERGEFORGE_"

# ENV 可覆盖字段 -> 类型名（name-based，避开 PEP 563 字符串注解问题）
_ENV_TYPES: dict[str, str] = {
    "d": "int",
    "n_classes": "int",
    "n_tasks": "int",
    "n_train": "int",
    "n_val": "int",
    "n_test": "int",
    "label_noise": "float",
    "latent_rank": "int",
    "prototype_scale": "float",
    "mean_shift_rho": "float",
    "rotate_inputs": "bool",
    "input_task_id": "bool",
    "specialist_from": "str",
    "freeze_trunk": "bool",
    "specialist_block_loss": "bool",
    "base_pretext": "str",
    "joint_epochs": "int",
    "hidden_dims": "tuple",
    "base_epochs": "int",
    "ft_epochs": "int",
    "lr": "float",
    "batch_size": "int",
    "lr_schedule": "str",
    "seeds": "tuple",
    "ties_rho_grid": "tuple",
    "ta_lambda_grid": "tuple",
    "dare_delta_grid": "tuple",
    "layer_lambda_grid": "tuple",
    "dare_delta_default": "float",
    "ties_rho_default": "float",
}


@dataclass
class MergeConfig:
    """MergeForge 全量配置。默认值为规范基准（canonical benchmark）配置。"""

    # ---- 数据 DGP ----
    d: int = 16  # 输入维度
    n_classes: int = 4  # 每任务类别数
    n_tasks: int = 6  # 任务数 K
    n_train: int = 900  # 每任务训练样本（joint 参照需 > base 才具上界意义）
    n_val: int = 300  # 每任务验证样本（合并方法校准专用，与 test 不相交）
    n_test: int = 600  # 每任务测试样本
    label_noise: float = 0.05  # 标签噪声（DGP 贝叶斯误差，train/val/test 一致）
    latent_rank: int = 4  # 共享潜因子秩 r（跨任务共享 trunk 结构）
    prototype_scale: float = 3.0  # 任务私有原型尺度（类间可分性）
    mean_shift_rho: float = 0.3  # 任务均值平移半径。A11 规范设定：任务在输入空间
    # 的正交补子空间中可识别（噪声less 任务身份信号）→ pretext base 可学会块路由；
    # 类内判别信息只存在于潜因子 z 中，pretext base 未见过真实类标签 → 块内 ≈ 随机
    # 1/C。「路由共享可学 / 块内任务私有」的结构性分工是合并增益的来源（A11）。
    rotate_inputs: bool = False  # 每任务正交旋转（与 mean_shift 二选一或同用）

    # ---- 模型 ----
    hidden_dims: tuple[int, ...] = (64, 64)
    input_task_id: bool = False  # True=regime B（任务 ID 拼入输入）；False=regime A（从 x 推断）
    specialist_from: str = "base"  # base=从联合训练模型微调（特征基共享，可组合）；
    # init=从共享初始化训练（实证：hidden>>r 时特征旋转歧义导致合并崩溃，见 model_card）
    freeze_trunk: bool = True  # True=专家只训输出层（LoRA/head-adapter 式合并）
    specialist_block_loss: bool = True  # True=专家 softmax 只算自有类别块（delta 零串扰）
    base_pretext: str = "block"  # block=base 学块路由（标签=块内随机类，预训练模型设定，
    # A11 规范）；labels=全局标签训练（A5 前旧设定，消融对照）

    # ---- 训练 ----
    # 算力口径：base_epochs × (K×n_train) = n_tasks × ft_epochs × n_train
    # （base 与 specialist 总量精确对齐）；joint_epochs 为 2× 预算的单模型参照。
    base_epochs: int = 100  # 多任务 base 训练轮数（联合单模型基线）
    ft_epochs: int = 100  # 每任务 specialist 训练轮数（从共享初始化训练）
    joint_epochs: int = 200  # joint 多任务参照训练轮数（2× base 预算的上界参照）
    lr: float = 0.005  # Adam 学习率（base/FT/joint 共用，温和 FT）
    batch_size: int = 8192  # 规范配置为 full-batch（>= base/joint 堆叠行数 n_tasks×n_train）
    lr_schedule: str = "cosine"  # const | cosine

    # ---- 基准 ----
    seeds: tuple[int, ...] = (7, 11, 23)

    # ---- 合并方法网格（在 val 上校准）----
    ties_rho_grid: tuple[float, ...] = (0.2, 0.5, 0.8, 1.0)
    ties_rho_default: float = 1.0
    ta_lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    dare_delta_grid: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6)
    dare_delta_default: float = 0.4
    # 逐张量 λ 网格上限 >1：不相交类别子空间下「求和组合」(λ≈1.0-1.5) 优于
    # 「平均组合」(λ=1/K 稀释)——soup 的稀释损失是旗舰超车空间（A11 实证）。
    layer_lambda_grid: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
    )

    def validate(self) -> None:
        if self.d < 2:
            raise ConfigError(f"d 必须 >= 2，收到 {self.d}")
        if self.n_tasks < 2:
            raise ConfigError(f"n_tasks 必须 >= 2（合并至少需要 2 个模型），收到 {self.n_tasks}")
        if self.n_classes < 2:
            raise ConfigError(f"n_classes 必须 >= 2，收到 {self.n_classes}")
        if self.n_train < 32:
            raise ConfigError(f"n_train 太小（{self.n_train}），batch 无法稳定")
        if not (0.0 <= self.label_noise < 0.5):
            raise ConfigError(f"label_noise 必须在 [0, 0.5)，收到 {self.label_noise}")
        if not (1 <= self.latent_rank <= self.d):
            raise ConfigError(f"latent_rank 必须在 [1, d={self.d}]，收到 {self.latent_rank}")
        if self.mean_shift_rho < 0:
            raise ConfigError(f"mean_shift_rho 必须 >= 0，收到 {self.mean_shift_rho}")
        if not self.input_task_id and self.mean_shift_rho == 0 and not self.rotate_inputs:
            raise ConfigError(
                "input_task_id=False 时 mean_shift_rho>0 或 rotate_inputs=True 至少取其一"
                "（否则任务无法从输入 x 识别，合并在数学上无意义）"
            )
        if self.joint_epochs < 1:
            raise ConfigError("joint_epochs 必须 >= 1")
        if not self.hidden_dims or any(h < 1 for h in self.hidden_dims):
            raise ConfigError(f"hidden_dims 非法: {self.hidden_dims}")
        if self.base_epochs < 1 or self.ft_epochs < 1:
            raise ConfigError("base_epochs/ft_epochs 必须 >= 1")
        if self.lr <= 0:
            raise ConfigError(f"lr 必须 > 0，收到 {self.lr}")
        if self.lr_schedule not in ("const", "cosine"):
            raise ConfigError(f"lr_schedule 必须为 const/cosine，收到 {self.lr_schedule!r}")
        if self.specialist_from not in ("base", "init"):
            raise ConfigError(f"specialist_from 必须为 base/init，收到 {self.specialist_from!r}")
        if self.base_pretext not in ("block", "labels"):
            raise ConfigError(f"base_pretext 必须为 block/labels，收到 {self.base_pretext!r}")
        if self.batch_size < 8:
            raise ConfigError(f"batch_size 必须 >= 8，收到 {self.batch_size}")
        if not self.seeds:
            raise ConfigError("seeds 不能为空")
        for grid_name in (
            "ties_rho_grid",
            "ta_lambda_grid",
            "dare_delta_grid",
            "layer_lambda_grid",
        ):
            grid = getattr(self, grid_name)
            if not grid:
                raise ConfigError(f"{grid_name} 不能为空")
            if any(v < 0 for v in grid):
                raise ConfigError(f"{grid_name} 含负值: {grid}")

    # ---- ENV 覆盖 ----
    @classmethod
    def from_env(cls, **overrides) -> MergeConfig:
        cfg = cls(**overrides)
        for name, type_name in _ENV_TYPES.items():
            env_key = f"{_ENV_PREFIX}{name.upper()}"
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            try:
                if type_name == "int":
                    value = int(raw)
                elif type_name == "float":
                    value = float(raw)
                elif type_name == "bool":
                    value = raw.strip().lower() in ("1", "true", "yes", "on")
                elif type_name == "tuple":
                    value = tuple(ast.literal_eval(raw))
                else:
                    value = raw
            except (ValueError, SyntaxError) as exc:
                raise ConfigError(f"ENV {env_key}={raw!r} 无法解析为 {type_name}: {exc}") from None
            object.__setattr__(cfg, name, value)
        cfg.validate()
        return cfg
