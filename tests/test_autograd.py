"""Tests for the miniature autodiff engine.

The engine is hand-written, so every operation and every composition the world model relies on
is checked against central finite differences. If these fail, nothing downstream means
anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from coimagine.learning.autograd import (
    Adam,
    MLP,
    Tensor,
    binary_cross_entropy,
    concat,
    mse,
    softmax_cross_entropy,
)


def _pack(params):
    return np.concatenate([p.data.ravel() for _k, p in params])


def _unpack(params, vec):
    offset = 0
    for _k, p in params:
        size = p.data.size
        p.data = vec[offset : offset + size].reshape(p.data.shape).copy()
        offset += size


def _gradient_check(build, params, *, samples: int = 10, tol: float = 1e-5) -> float:
    """Worst relative error between analytic and finite-difference gradients."""

    tensor = build()
    tensor.backward()
    analytic = np.concatenate([np.asarray(p.grad).ravel() for _k, p in params])
    base = _pack(params)
    rng = np.random.default_rng(7)
    worst = 0.0
    for index in rng.choice(base.size, size=min(samples, base.size), replace=False):
        step = 1e-6
        plus = base.copy(); plus[index] += step; _unpack(params, plus)
        up = float(np.asarray(build().data).reshape(()))
        minus = base.copy(); minus[index] -= step; _unpack(params, minus)
        down = float(np.asarray(build().data).reshape(()))
        _unpack(params, base)
        numeric = (up - down) / (2 * step)
        exact = float(analytic[index])
        worst = max(worst, abs(numeric - exact) / max(1e-9, abs(numeric) + abs(exact)))
    return worst


def test_gradients_of_a_dense_stack() -> None:
    rng = np.random.default_rng(1)
    net = MLP([5, 7, 3], rng, name="m")
    x = np.asarray(rng.normal(size=(4, 5)))
    target = np.asarray(rng.normal(size=(4, 3)))
    assert _gradient_check(lambda: mse(net(Tensor(x)), target), list(net.parameters())) < 1e-5


def test_bias_gradient_is_reduced_over_the_batch() -> None:
    """The broadcasting bug that made the first version silently wrong: a bias added to a
    batch must receive a gradient of its own shape, not the batch's."""

    rng = np.random.default_rng(11)
    net = MLP([3, 4], rng, name="b")
    x = np.asarray(rng.normal(size=(6, 3)))
    target = np.asarray(rng.normal(size=(6, 4)))
    _gradient_check(lambda: mse(net(Tensor(x)), target), list(net.parameters()))
    for name, parameter in net.parameters():
        assert parameter.grad.shape == parameter.data.shape, name


def test_gradients_through_two_heads_and_cross_entropy() -> None:
    rng = np.random.default_rng(2)
    trunk = MLP([4, 6], rng, name="t")
    head = MLP([8, 5], rng, name="h", final_activation="none")
    action = np.asarray(rng.normal(size=(3, 2)))
    labels = np.asarray([0, 3, 4])
    observation = np.asarray(rng.normal(size=(3, 4)))
    build = lambda: softmax_cross_entropy(  # noqa: E731
        head(concat([trunk(Tensor(observation)), Tensor(action)])), labels
    )
    worst = _gradient_check(build, list(trunk.parameters()) + list(head.parameters()))
    assert worst < 1e-5


def test_gradients_through_a_two_step_rollout() -> None:
    """The world model's core graph: a latent carried through the transition twice."""

    rng = np.random.default_rng(5)
    encoder = MLP([4, 6], rng, name="enc")
    transition = MLP([8, 6], rng, name="tr")
    decoder = MLP([6, 4], rng, name="dec", final_activation="none")
    observation = np.asarray(rng.normal(size=(3, 4)))
    first = np.asarray(rng.normal(size=(3, 2)))
    second = np.asarray(rng.normal(size=(3, 2)))
    target = np.asarray(rng.normal(size=(3, 4)))

    def build():
        z = encoder(Tensor(observation))
        z = transition(concat([z, Tensor(first)]))
        z = transition(concat([z, Tensor(second)]))
        return mse(decoder(z), target)

    params = (
        list(encoder.parameters())
        + list(transition.parameters())
        + list(decoder.parameters())
    )
    assert _gradient_check(build, params) < 1e-5


def test_gradients_of_weighted_binary_cross_entropy() -> None:
    rng = np.random.default_rng(3)
    head = MLP([3, 1], rng, name="b", final_activation="none")
    labels = np.asarray([[1.0], [0.0], [1.0]])
    inputs = np.asarray(rng.normal(size=(3, 3)))
    build = lambda: binary_cross_entropy(head(Tensor(inputs)), labels, 3.0)  # noqa: E731
    assert _gradient_check(build, list(head.parameters())) < 1e-5


def test_adam_reduces_a_loss() -> None:
    """A mechanics test for the optimiser, not a function-approximation claim: the target is
    linear, so a small network should drive the loss close to zero."""

    rng = np.random.default_rng(4)
    net = MLP([2, 16, 1], rng, name="f", final_activation="none")
    optimiser = Adam(net.parameters(), lr=1e-2)
    inputs = np.asarray(rng.normal(size=(64, 2)))
    targets = np.asarray((inputs @ np.asarray([1.5, -2.0]))[:, None])
    first = None
    for _ in range(400):
        optimiser.zero_grad()
        loss = mse(net(Tensor(inputs)), targets)
        loss.backward()
        optimiser.update()
        if first is None:
            first = float(loss.data)
    assert float(loss.data) < first * 0.05, (first, float(loss.data))


def test_impossible_observations_do_not_produce_nan() -> None:
    """Cross entropy on a fully saturated logit must stay finite."""

    logits = Tensor(np.asarray([[50.0, -50.0], [-50.0, 50.0]]))
    loss = softmax_cross_entropy(logits, np.asarray([0, 1]))
    loss.backward()
    assert np.isfinite(float(loss.data))
    assert np.all(np.isfinite(logits.grad))
