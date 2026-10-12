"""纯 NumPy MLP（解析梯度 + 中心差分梯度自检）。

- 结构：[Linear → ReLU] × L → Linear → softmax CE。
- 一切计算 float64；参数以「扁平向量」暴露，合并只依赖该表示。
- 初始化：He normal（ReLU 层）/ 小方差（输出层）。
"""

from __future__ import annotations

import numpy as np

from core.errors import ModelError
from core.thread_pin import pin_threads

pin_threads()


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(np.float64)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """多层感知机（分类）。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        output_dim: int,
        rng: np.random.RandomState,
    ) -> None:
        if input_dim < 1 or output_dim < 1:
            raise ModelError(f"非法维度: input={input_dim}, output={output_dim}")
        dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(output_dim)]
        self.dims = dims
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.n_layers = len(dims) - 1
        for i in range(self.n_layers):
            fan_in, fan_out = dims[i], dims[i + 1]
            is_output = i == self.n_layers - 1
            std = 0.01 if is_output else np.sqrt(2.0 / fan_in)
            self.weights.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out, dtype=np.float64))

    # ---- 参数扁平化（合并的唯一入口）----
    @property
    def param_spec(self):
        from core.types import ParamSpec

        names: list[str] = []
        shapes: list[tuple[int, ...]] = []
        for i in range(self.n_layers):
            names.extend([f"W{i + 1}", f"b{i + 1}"])
            shapes.extend([self.weights[i].shape, self.biases[i].shape])
        return ParamSpec(names=names, shapes=shapes)

    def get_flat(self) -> np.ndarray:
        # 顺序必须与 param_spec 严格一致（逐层 W,b 交错）：
        # get_flat 块序 / spec 交错序的错位会让 set_flat 静默重组参数
        # （训练循环每步损坏 + 合并评测全部失真），且不抛任何异常。
        parts: list[np.ndarray] = []
        for w, b in zip(self.weights, self.biases, strict=True):
            parts.extend([w.ravel(), b.ravel()])
        return np.concatenate(parts).astype(np.float64)

    def set_flat(self, flat: np.ndarray) -> None:
        spec = self.param_spec
        if flat.shape != (spec.n_params,):
            raise ModelError(f"扁平参数长度 {flat.shape} != ({spec.n_params},)")
        tensors = spec.split(flat)
        for i in range(self.n_layers):
            self.weights[i] = tensors[f"W{i + 1}"].copy()
            self.biases[i] = tensors[f"b{i + 1}"].copy()

    def copy(self) -> MLP:
        new = MLP.__new__(MLP)
        new.dims = list(self.dims)
        new.n_layers = self.n_layers
        new.weights = [w.copy() for w in self.weights]
        new.biases = [b.copy() for b in self.biases]
        return new

    # ---- 前向 / 损失 / 梯度 ----
    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2 or X.shape[1] != self.dims[0]:
            raise ModelError(f"X 形状 {X.shape} 与输入维度 {self.dims[0]} 不符")
        a = np.asarray(X, dtype=np.float64)
        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            a = relu(z) if i < self.n_layers - 1 else z
        return a  # logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.forward(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=1)

    def loss_and_grad(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_block: tuple[int, int] | None = None,
    ) -> tuple[float, np.ndarray]:
        """返回 (mean CE loss, 扁平解析梯度)。

        class_block=(a, b) 时 softmax/CE 只计算 logits[:, a:b]（私有标签子空间下
        任务专家只对自己的类别块建立判别）：梯度只流入该块对应的输出行，
        任务向量的其他块分量严格为 0——这是多头专家 delta 可无损平均的前提
        （24-way 全程 softmax 会让各专家把其他块大幅压制写进 delta，平均后互相污染）。
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n = X.shape[0]
        if n == 0:
            raise ModelError("空 batch")
        if y.shape != (n,):
            raise ModelError(f"y 形状 {y.shape} != ({n},)")
        if y.min() < 0 or y.max() >= self.dims[-1]:
            raise ModelError(f"标签越界: [{y.min()}, {y.max()}] vs C={self.dims[-1]}")
        if class_block is not None:
            a, b = class_block
            if not (0 <= a < b <= self.dims[-1]):
                raise ModelError(f"class_block {class_block} 非法（C={self.dims[-1]}）")
            if y.min() < a or y.max() >= b:
                raise ModelError(f"标签 [{y.min()}, {y.max()}] 越出 class_block {class_block}")
            y_local = y - a
        else:
            y_local = y

        # 前向（缓存每层输入与预激活）
        acts: list[np.ndarray] = []
        pre: list[np.ndarray] = []
        a = X
        for i in range(self.n_layers):
            acts.append(a)
            z = a @ self.weights[i] + self.biases[i]
            pre.append(z)
            a = relu(z) if i < self.n_layers - 1 else z
        logits = a
        if class_block is not None:
            lo, hi = class_block
            probs = softmax(logits[:, lo:hi])
            logp = np.log(np.clip(probs[np.arange(n), y_local], 1e-300, None))
            loss = float(-logp.mean())
        else:
            probs = softmax(logits)
            logp = np.log(np.clip(probs[np.arange(n), y], 1e-300, None))
            loss = float(-logp.mean())

        # 反向
        grads_w: list[np.ndarray] = [np.empty_like(w) for w in self.weights]
        grads_b: list[np.ndarray] = [np.empty_like(b) for b in self.biases]
        delta = np.zeros_like(logits)
        if class_block is not None:
            # dL/dz_c = p_c - 1{c=y}（c ∈ 块内）；块外列保持 0。
            delta[:, lo:hi] = probs
            delta[np.arange(n), y] -= 1.0
            delta /= n
        else:
            delta = probs.copy()
            delta[np.arange(n), y] -= 1.0
            delta /= n
        for i in range(self.n_layers - 1, -1, -1):
            a_prev = acts[i]
            grads_w[i] = a_prev.T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_grad(pre[i - 1])
        # 梯度扁平序必须与 get_flat / param_spec 一致（逐层 W,b 交错），
        # 否则 Adam 会把 dW2 更新到 b1 上（块序/交错序错位，无声错误）。
        grad_pairs: list[np.ndarray] = []
        for gw, gb in zip(grads_w, grads_b, strict=True):
            grad_pairs.extend([gw.ravel(), gb.ravel()])
        grad_flat = np.concatenate(grad_pairs)
        return loss, grad_flat

    # ---- 数值梯度自检 ----
    def numerical_grad(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-5,
        class_block: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """中心差分数值梯度（用于梯度校验，勿用于训练）。"""
        flat = self.get_flat()
        num = np.zeros_like(flat)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            self.set_flat(flat)
            lp, _ = self.loss_and_grad(X, y, class_block=class_block)
            flat[i] = old - eps
            self.set_flat(flat)
            lm, _ = self.loss_and_grad(X, y, class_block=class_block)
            num[i] = (lp - lm) / (2.0 * eps)
            flat[i] = old
        self.set_flat(flat)
        return num


def gradient_check(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    n_probe: int = 12,
    rng: np.random.RandomState | None = None,
    eps: float = 1e-5,
    class_block: tuple[int, int] | None = None,
) -> float:
    """随机抽取 n_probe 个参数坐标做中心差分校验，返回最大相对误差。"""
    rng = rng or np.random.RandomState(0)
    _, analytic = model.loss_and_grad(X, y, class_block=class_block)
    flat = model.get_flat()
    idx = rng.choice(flat.size, size=min(n_probe, flat.size), replace=False)
    max_rel = 0.0
    for i in idx:
        old = flat[i]
        flat[i] = old + eps
        model.set_flat(flat)
        lp, _ = model.loss_and_grad(X, y, class_block=class_block)
        flat[i] = old - eps
        model.set_flat(flat)
        lm, _ = model.loss_and_grad(X, y, class_block=class_block)
        num = (lp - lm) / (2.0 * eps)
        flat[i] = old
        rel = abs(num - analytic[i]) / max(abs(num), abs(analytic[i]), 1e-8)
        max_rel = max(max_rel, float(rel))
    model.set_flat(flat)
    return max_rel
