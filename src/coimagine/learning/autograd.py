"""A tiny reverse-mode autodiff over NumPy arrays.

Why this exists: the project's earlier models were hand-written MLPs with bespoke backward
passes. That is fine for a fixed architecture and dangerous for anything larger. The world
model needs an encoder, a transition, several decoders and a multi-step rollout objective,
and every change to that graph would otherwise need a new analytic gradient.

This is a miniature version of the same idea as any framework, with only the operations the
world model needs. It is small enough to read in one sitting and every operation is checked
against finite differences in the tests.

Deliberately **not** included: broadcasting magic, in-place ops, GPU, or anything clever.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


class Tensor:
    """A node in the computation graph: an array plus the rule that made it."""

    __slots__ = ("data", "grad", "_backward", "_parents")

    def __init__(
        self,
        data: np.ndarray,
        parents: Sequence["Tensor"] = (),
        backward: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = tuple(parents)
        self._backward = backward

    # -- graph helpers -----------------------------------------------------
    def backward(self) -> None:
        """Accumulate gradients over the whole graph from this node."""

        topo: List[Tensor] = []
        seen = set()

        def build(node: Tensor) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._parents:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._backward is not None:
                node._backward(node.grad)

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape


def _reduce_grad(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Sum a gradient back down to ``shape``.

    Needed whenever an operand was broadcast: a bias of shape ``(d,)`` added to a batch of
    shape ``(n, d)`` receives a gradient of shape ``(n, d)`` and must be summed over the
    batch. Without this the bias gradient silently carries the batch dimension and training
    does something else entirely.
    """

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _binary(
    a: Tensor, b: Tensor, forward, backward_a, backward_b
) -> Tensor:
    out = Tensor(forward(a.data, b.data), (a, b))

    def _backward(grad: np.ndarray) -> None:
        ga = backward_a(grad, a.data, b.data, out.data)
        gb = backward_b(grad, a.data, b.data, out.data)
        if ga is not None:
            # A parameter may appear on both sides (or twice), so accumulate. Broadcasting
            # is undone here so every parameter gets a gradient of its own shape.
            a.grad = a.grad + _reduce_grad(np.asarray(ga), a.data.shape)
        if gb is not None:
            b.grad = b.grad + _reduce_grad(np.asarray(gb), b.data.shape)

    out._backward = _backward
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return _binary(
        a,
        b,
        lambda x, y: x @ y,
        lambda g, x, y, _o: g @ y.T,
        lambda g, x, y, _o: x.T @ g,
    )


def add(a: Tensor, b: Tensor) -> Tensor:
    return _binary(a, b, lambda x, y: x + y, lambda g, *_: g, lambda g, *_: g)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return _binary(
        a, b, lambda x, y: x * y, lambda g, x, y, _o: g * y, lambda g, x, y, _o: g * x
    )


def scale(a: Tensor, factor: float) -> Tensor:
    out = Tensor(a.data * factor, (a,))

    def _backward(grad: np.ndarray) -> None:
        a.grad = a.grad + grad * factor

    out._backward = _backward
    return out


def relu(a: Tensor) -> Tensor:
    out = Tensor(np.maximum(0.0, a.data), (a,))

    def _backward(grad: np.ndarray) -> None:
        a.grad = a.grad + grad * (a.data > 0.0)

    out._backward = _backward
    return out


def tanh(a: Tensor) -> Tensor:
    value = np.tanh(a.data)
    out = Tensor(value, (a,))

    def _backward(grad: np.ndarray) -> None:
        a.grad = a.grad + grad * (1.0 - value * value)

    out._backward = _backward
    return out


def sigmoid(a: Tensor) -> Tensor:
    value = _stable_sigmoid(a.data)
    out = Tensor(value, (a,))

    def _backward(grad: np.ndarray) -> None:
        a.grad = a.grad + grad * value * (1.0 - value)

    out._backward = _backward
    return out


def concat(tensors: Sequence[Tensor], axis: int = 1) -> Tensor:
    out = Tensor(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors))
    sizes = [t.data.shape[axis] for t in tensors]

    def _backward(grad: np.ndarray) -> None:
        start = 0
        for tensor, size in zip(tensors, sizes):
            piece = np.take(grad, range(start, start + size), axis=axis)
            tensor.grad = tensor.grad + piece
            start += size

    out._backward = _backward
    return out


def mse(prediction: Tensor, target: np.ndarray) -> Tensor:
    """Mean squared error against a constant target."""

    n = prediction.data.size
    residual = prediction.data - target
    out = Tensor(np.array((residual ** 2).sum() / n), (prediction,))

    def _backward(grad: np.ndarray) -> None:
        prediction.grad = prediction.grad + grad * 2.0 * residual / n

    out._backward = _backward
    return out


def weighted_mse(prediction: Tensor, target: np.ndarray, weight: np.ndarray) -> Tensor:
    """Mean squared error with a per-element weight.

    The observation is a one-hot occupancy plane that is mostly zeros, so an unweighted MSE is
    dominated by getting the empty cells right and is nearly blind to the few cells that carry
    the object. The weight marks which elements carry signal.
    """

    residual = prediction.data - target
    total_weight = weight.sum()
    out = Tensor(np.array((weight * residual ** 2).sum() / max(1.0, total_weight)), (prediction,))

    def _backward(grad: np.ndarray) -> None:
        prediction.grad = prediction.grad + grad * 2.0 * weight * residual / max(1.0, total_weight)

    out._backward = _backward
    return out


def softmax_cross_entropy(
    logits: Tensor, target: np.ndarray, weight: Optional[np.ndarray] = None
) -> Tensor:
    """Cross entropy with integer class targets, computed stably.

    ``weight`` is an optional per-example weight. It exists because the transitions that carry
    the dynamics are rare: a ninth of the actions move the object, so an unweighted loss is
    minimised by predicting "the object stays put" for every action, which is exactly what the
    first working version of the world model learned.
    """

    z = logits.data - logits.data.max(axis=1, keepdims=True)
    exp = np.exp(z)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    n = logits.data.shape[0]
    per_example = -np.log(probabilities[np.arange(n), target] + 1e-12)
    if weight is None:
        weight = np.ones(n)
    weight = np.asarray(weight, dtype=np.float64)
    total_weight = max(1e-9, weight.sum())
    loss = float((weight * per_example).sum() / total_weight)
    out = Tensor(np.array(loss), (logits,))

    def _backward(grad: np.ndarray) -> None:
        copy = probabilities.copy()
        copy[np.arange(n), target] -= 1.0
        logits.grad = logits.grad + grad * copy * weight[:, None] / total_weight

    out._backward = _backward
    return out


def softmax(logits: Tensor, axis: int = 1) -> Tensor:
    """Row-wise softmax, differentiable.

    The factored world model carries an explicit belief as a probability vector, so it needs a
    softmax that is part of the graph rather than a post-hoc numpy call.
    """

    shifted = logits.data - logits.data.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=axis, keepdims=True)
    out = Tensor(probabilities, (logits,))

    def _backward(grad: np.ndarray) -> None:
        # dL/dz = p * (g - sum(g * p))
        inner = (grad * probabilities).sum(axis=axis, keepdims=True)
        logits.grad = logits.grad + probabilities * (grad - inner)

    out._backward = _backward
    return out


def binary_cross_entropy(logits: Tensor, target: np.ndarray, positive_weight: float = 1.0) -> Tensor:
    """Binary cross entropy on a (n, 1) logit tensor."""

    probabilities = _stable_sigmoid(logits.data)
    weight = np.where(target > 0.5, positive_weight, 1.0)
    n = logits.data.shape[0]
    loss = (
        weight
        * -(target * np.log(probabilities + 1e-12) + (1 - target) * np.log(1 - probabilities + 1e-12))
    ).sum() / n
    out = Tensor(np.array(float(loss)), (logits,))

    def _backward(grad: np.ndarray) -> None:
        d = np.where(target > 0.5, positive_weight * (probabilities - 1.0), probabilities)
        logits.grad = logits.grad + grad * d / n

    out._backward = _backward
    return out


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    out[~positive] = exp / (1.0 + exp)
    return out


# ---------------------------------------------------------------------------
# Layers and optimisation
# ---------------------------------------------------------------------------


@dataclass
class Linear:
    """A dense layer holding its parameters as tensors."""

    input_dim: int
    output_dim: int
    rng: np.random.Generator
    name: str = "linear"
    gain: float = 2.0

    def __post_init__(self) -> None:
        scale = np.sqrt(self.gain / self.input_dim)
        self.weight = Tensor(self.rng.normal(0.0, scale, size=(self.input_dim, self.output_dim)))
        self.bias = Tensor(np.zeros(self.output_dim))

    def __call__(self, x: Tensor) -> Tensor:
        return add(matmul(x, self.weight), self.bias)

    def parameters(self) -> List[Tuple[str, Tensor]]:
        return [(f"{self.name}.weight", self.weight), (f"{self.name}.bias", self.bias)]


@dataclass
class MLP:
    """A stack of dense layers with ReLU activations and an optional linear output."""

    dims: Sequence[int]
    rng: np.random.Generator
    name: str = "mlp"
    #: Applied to the last layer. Defaults to **linear**, which is the correct default for a
    #: regression or decoder head: a ReLU output cannot represent negative targets, and an
    #: earlier version defaulted to ReLU and silently plateaued at the target's variance.
    final_activation: str = "none"

    def __post_init__(self) -> None:
        self.layers = [
            Linear(self.dims[i], self.dims[i + 1], self.rng, name=f"{self.name}.{i}")
            for i in range(len(self.dims) - 1)
        ]

    def __call__(self, x: Tensor) -> Tensor:
        h = x
        for index, layer in enumerate(self.layers):
            h = layer(h)
            last = index == len(self.layers) - 1
            if not last:
                h = relu(h)
            elif self.final_activation == "relu":
                h = relu(h)
            elif self.final_activation == "tanh":
                h = tanh(h)
        return h

    def parameters(self) -> List[Tuple[str, Tensor]]:
        out: List[Tuple[str, Tensor]] = []
        for layer in self.layers:
            out.extend(layer.parameters())
        return out


@dataclass
class Adam:
    """Adam over a list of named tensors. Hand-written, like everything else here."""

    parameters: List[Tuple[str, Tensor]]
    lr: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    step: int = 0
    state: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def zero_grad(self) -> None:
        for _name, tensor in self.parameters:
            tensor.grad = np.zeros_like(tensor.data)

    def update(self) -> None:
        self.step += 1
        for name, tensor in self.parameters:
            m, v = self.state.get(name, (np.zeros_like(tensor.data), np.zeros_like(tensor.data)))
            m = self.beta1 * m + (1 - self.beta1) * tensor.grad
            v = self.beta2 * v + (1 - self.beta2) * (tensor.grad ** 2)
            m_hat = m / (1 - self.beta1 ** self.step)
            v_hat = v / (1 - self.beta2 ** self.step)
            tensor.data = tensor.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            self.state[name] = (m, v)
