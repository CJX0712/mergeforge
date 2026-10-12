"""训练流水线：共享初始化 + pretext base 训练 + 每任务 specialist 训练。

无泄漏保证：
- base 只在 train 折上训练；val 折仅供合并方法校准；test 折只在最终评测出现。
- 每次运行的随机性全部来自种子（init / shuffle / pretext 重标注均无隐式随机）。

专家训练范式（A11 规范设定）：base 以块路由 pretext 训练（预训练模型角色：
见过域、会路由、不会任务分类），specialist 从 base 复制、冻结 trunk、
只训输出层且只算自有类别块——delta 是纯头部增量且不相交，可无损组合。
历史设定（A4 从初始化 / A5 全局标签 base）的失败实证见 docs/model_card.md。
"""

from __future__ import annotations

import numpy as np

from core.seed import make_rng
from core.thread_pin import pin_threads

pin_threads()

from data.tasks import add_task_id  # noqa: E402
from models.mlp import MLP  # noqa: E402
from models.optimizer import Adam  # noqa: E402


def model_input_dim(cfg) -> int:
    return cfg.d + (cfg.n_tasks if cfg.input_task_id else 0)


def model_output_dim(cfg) -> int:
    """全局输出维度 = K×C（私有类别子空间：任务 k 独占类别块 k）。"""
    return cfg.n_tasks * cfg.n_classes


def make_model(cfg, rng: np.random.RandomState) -> MLP:
    return MLP(
        input_dim=model_input_dim(cfg),
        hidden_dims=tuple(cfg.hidden_dims),
        output_dim=model_output_dim(cfg),
        rng=rng,
    )


def _batches(n: int, batch_size: int, rng: np.random.RandomState):
    perm = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = perm[start : start + batch_size]
        yield idx


def head_grad_mask(model: MLP) -> np.ndarray:
    """输出层（头）参数位置为 1、trunk 为 0 的梯度掩码（flat 序 = param_spec）。"""
    spec = model.param_spec
    mask = np.zeros(spec.n_params, dtype=np.float64)
    head_names = {f"W{model.n_layers}", f"b{model.n_layers}"}
    i = 0
    for name, shape in zip(spec.names, spec.shapes, strict=True):
        n = int(np.prod(shape))
        if name in head_names:
            mask[i : i + n] = 1.0
        i += n
    return mask


def _train_loop(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    rng: np.random.RandomState,
    lr_schedule: str = "const",
    grad_mask: np.ndarray | None = None,
    class_block: tuple[int, int] | None = None,
) -> MLP:
    opt = Adam(model.param_spec.n_params, lr=lr)
    n = X.shape[0]
    steps_per_epoch = max(1, (n + batch_size - 1) // batch_size)
    total_steps = epochs * steps_per_epoch
    step = 0
    for _ in range(epochs):
        for idx in _batches(n, batch_size, rng):
            if lr_schedule == "cosine" and total_steps > 1:
                opt.lr = lr * 0.5 * (1.0 + np.cos(np.pi * step / (total_steps - 1)))
            _, grad = model.loss_and_grad(X[idx], y[idx], class_block=class_block)
            if grad_mask is not None:
                grad = grad * grad_mask
            flat = opt.step(model.get_flat(), grad)
            model.set_flat(flat)
            step += 1
    return model


def make_init_model(cfg, seed: int) -> MLP:
    """共享初始化：base 与全部 specialist 的同源起点（预训练模型的角色）。"""
    return make_model(cfg, make_rng(seed + 1))


def train_base(
    tasks: dict,
    cfg,
    seed: int,
    epochs: int | None = None,
    pretext: bool | None = None,
) -> MLP:
    """多任务 base：在全部任务的 train 折上从共享初始化训练。

    base_pretext="block"（规范设定，A11）：标签替换为「块内随机类」
    （y = k·C + rand(C)）——base 只学**块路由**（哪个任务的样本），不学
    块内判别。这是文献「预训练模型 + 任务微调」设定的对应物：base 扮演
    预训练模型（见过域、会路由、不会任务分类，块内 ≈ 随机 1/C），专家
    扮演任务微调（补全各自类别块）。合并 = 路由（base 提供，共享函数）
    + 块内判别（专家提供，不相交子空间）——两类知识可分解，任务向量可组合。
    base_pretext="labels"：全局标签训练（A5 前的旧设定，消融对照）。

    epochs=None 用 cfg.base_epochs；传入 cfg.joint_epochs 即 joint 多任务参照
    （单模型联合训练上界参照，算力口径见 config 注释）。
    """
    model = make_init_model(cfg, seed)
    use_pretext = (
        (getattr(cfg, "base_pretext", "labels") == "block") if pretext is None else pretext
    )
    rng_pretext = make_rng(seed + 900)
    Xs, ys = [], []
    for k in sorted(tasks):
        t = tasks[k]
        Xs.append(add_task_id(t.X_train, k, cfg.n_tasks) if cfg.input_task_id else t.X_train)
        if use_pretext:
            y = k * cfg.n_classes + rng_pretext.randint(0, cfg.n_classes, size=t.y_train.shape[0])
            ys.append(y.astype(np.int64))
        else:
            ys.append(t.y_train)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    rng_train = make_rng(seed + 2)
    return _train_loop(
        model,
        X,
        y,
        cfg.base_epochs if epochs is None else epochs,
        cfg.batch_size,
        cfg.lr,
        rng_train,
        lr_schedule=getattr(cfg, "lr_schedule", "const"),
    )


def train_specialist(
    base_model: MLP,
    task,
    k: int,
    cfg,
    seed: int,
    freeze_trunk: bool = False,
    block_loss: bool = True,
) -> MLP:
    """从 base 复制参数训练任务 k 的 specialist。

    freeze_trunk=True 时只训练输出层（梯度掩码钉死 trunk）——LoRA/head-adapter
    式合并设定：专家不可能发散共享特征基，任务向量是纯头部增量。
    block_loss=True 时 softmax/CE 只计算任务自有的类别块（私有标签子空间的
    自然 per-task 损失）：任务向量的其他块分量严格为 0，多个专家的 delta
    可无损平均；24-way 全程 softmax 会把对其他块的压制写进 delta，平均后互污
    （实证：soup 塌缩到 base 水平，见 model_card）。
    """
    model = base_model.copy()
    X = add_task_id(task.X_train, k, cfg.n_tasks) if cfg.input_task_id else task.X_train
    rng_train = make_rng(seed + 100 + k)
    mask = head_grad_mask(model) if freeze_trunk else None
    block = (k * cfg.n_classes, (k + 1) * cfg.n_classes) if block_loss else None
    return _train_loop(
        model,
        X,
        task.y_train,
        cfg.ft_epochs,
        cfg.batch_size,
        cfg.lr,
        rng_train,
        lr_schedule=getattr(cfg, "lr_schedule", "const"),
        grad_mask=mask,
        class_block=block,
    )
