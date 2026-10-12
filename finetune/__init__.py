"""训练包（共享初始化 + base 联合训练 + specialist 从初始化训练）。"""

from finetune.trainer import (
    make_init_model,
    make_model,
    model_input_dim,
    model_output_dim,
    train_base,
    train_specialist,
)

__all__ = [
    "make_init_model",
    "make_model",
    "model_input_dim",
    "model_output_dim",
    "train_base",
    "train_specialist",
]
