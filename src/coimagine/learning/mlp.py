"""A small multi-head MLP in NumPy, with explicit gradients.

Why hand-written: the environment is tiny and tabular, the model has ~20k parameters,
and keeping the whole computation inspectable matters more here than framework
convenience. There is no heavy dependency and no hidden autograd.

The architecture implements the design under test:

* a shared trunk encodes the robot's own observation into a **latent**;
* an **action head** consumes the latent plus a candidate action and predicts whether
  that action will work;
* **privileged decoder heads** are attached to the same latent and are supervised by
  the true object placement and the true contact physics.

The decoder heads are what let us ask whether the latent actually carries the contact
mode, rather than only the geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import ACTION_IDS, OBSERVATION_DIM


def _sigmoid(logit: np.ndarray) -> np.ndarray:
    """Numerically stable logistic.

    A plain ``1 / (1 + exp(-logit))`` overflows once the logit is very negative, which
    happens as soon as the positive-class weight is large: the model drives logits to
    -30 and the exponential warns and loses precision.
    """

    out = np.empty_like(logit, dtype=np.float64)
    positive = logit >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp = np.exp(logit[~positive])
    out[~positive] = exp / (1.0 + exp)
    return out


def _he(shape, rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[0]
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


@dataclass
class MultiHeadMLP:
    """Shared latent trunk with a feasibility head and two privileged decoder heads."""

    obs_dim: int = OBSERVATION_DIM
    hidden: int = 64
    action_dim: int = len(ACTION_IDS)
    n_physics: int = 16
    n_cells: int = 36
    seed: int = 0

    #: Number of action-outcome channels, in a fixed order:
    #: 0 = the object moves, 1 = the object makes progress toward the goal,
    #: 2 = the action enables a push that was not possible before.
    n_outcomes: int = 3

    #: Loss weights. Fixed before training and reported; never tuned on the test set.
    feasibility_weight: float = 1.0
    physics_weight: float = 0.3
    pose_weight: float = 0.3
    value_weight: float = 1.0

    #: Divisor applied to value targets so the MSE is on a comparable scale.
    value_scale: float = 20.0

    #: Weight on the force-threshold regression head, and the divisor for its target.
    threshold_weight: float = 1.0
    threshold_scale: float = 2.0

    #: Weight on positive examples in the outcome BCE. The continuous probe's labels are
    #: 3.7% positive, where an unweighted model minimises loss by predicting "never
    #: feasible"; set this to ``(1 - p) / p`` from the training rate to counteract that.
    #: It is a declared, standard reweighting, reported with the result, not tuned on test.
    positive_weight: float = 1.0

    params: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.params:
            return
        rng = np.random.default_rng(self.seed)
        h = self.hidden
        a = self.action_dim
        self.params = {
            "W1": _he((self.obs_dim, h), rng), "b1": np.zeros(h),
            "W2": _he((h, h), rng), "b2": np.zeros(h),
            "Wf1": _he((h + a, h // 2), rng), "bf1": np.zeros(h // 2),
            "Wf2": _he((h // 2, self.n_outcomes), rng), "bf2": np.zeros(self.n_outcomes),
            "Wp": _he((h, self.n_physics), rng), "bp": np.zeros(self.n_physics),
            "Wq": _he((h, self.n_cells), rng), "bq": np.zeros(self.n_cells),
            "Wv1": _he((h + a, h // 2), rng), "bv1": np.zeros(h // 2),
            "Wv2": _he((h // 2, 1), rng), "bv2": np.zeros(1),
            "Wt1": _he((h + a, h // 2), rng), "bt1": np.zeros(h // 2),
            "Wt2": _he((h // 2, 1), rng), "bt2": np.zeros(1),
        }

    # -- forward -----------------------------------------------------------
    def latent(self, obs: np.ndarray) -> np.ndarray:
        p = self.params
        z1 = np.maximum(0.0, obs @ p["W1"] + p["b1"])
        z2 = np.maximum(0.0, z1 @ p["W2"] + p["b2"])
        return z2

    def forward(self, obs: np.ndarray, action: np.ndarray) -> Dict[str, np.ndarray]:
        p = self.params
        z1 = np.maximum(0.0, obs @ p["W1"] + p["b1"])
        z2 = np.maximum(0.0, z1 @ p["W2"] + p["b2"])
        joined = np.concatenate([z2, action], axis=1)
        f1 = np.maximum(0.0, joined @ p["Wf1"] + p["bf1"])
        outcome_logit = f1 @ p["Wf2"] + p["bf2"]
        v1 = np.maximum(0.0, joined @ p["Wv1"] + p["bv1"])
        value = (v1 @ p["Wv2"] + p["bv2"])[:, 0]
        t1 = np.maximum(0.0, joined @ p["Wt1"] + p["bt1"])
        threshold = (t1 @ p["Wt2"] + p["bt2"])[:, 0]
        return {
            "z1": z1, "z2": z2, "joined": joined, "f1": f1,
            "outcome_logit": outcome_logit,
            "v1": v1, "value": value,
            "t1": t1, "threshold": threshold,
            "phys_logit": z2 @ p["Wp"] + p["bp"],
            "pose_logit": z2 @ p["Wq"] + p["bq"],
        }

    # -- loss and gradients -------------------------------------------------
    def loss_and_grads(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        targets: np.ndarray,
        physics_index: Optional[np.ndarray] = None,
        pose_index: Optional[np.ndarray] = None,
        values: Optional[np.ndarray] = None,
        thresholds: Optional[np.ndarray] = None,
        threshold_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, np.ndarray], Dict[str, float]]:
        """Binary cross-entropy on the outcome channels, MSE on the value, plus the
        privileged decoders."""

        if targets.ndim == 1:
            targets = targets[:, None]
        cache = self.forward(obs, action)
        p = self.params
        n = obs.shape[0]
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        logit = cache["outcome_logit"]
        prob = _sigmoid(logit)
        eps = 1e-12
        weight = np.where(targets > 0.5, self.positive_weight, 1.0)
        bce = weight * -(
            targets * np.log(prob + eps) + (1 - targets) * np.log(1 - prob + eps)
        )
        loss = self.feasibility_weight * float(bce.sum()) / n
        metrics = {"feasibility": loss}

        d_logit = (self.feasibility_weight / n) * np.where(
            targets > 0.5, self.positive_weight * (prob - 1.0), prob
        )
        grads["Wf2"] += cache["f1"].T @ d_logit
        grads["bf2"] += d_logit.sum(axis=0)
        d_f1 = (d_logit @ p["Wf2"].T) * (cache["f1"] > 0)
        grads["Wf1"] += cache["joined"].T @ d_f1
        grads["bf1"] += d_f1.sum(axis=0)
        d_z2 = d_f1 @ p["Wf1"].T[:, : self.hidden]

        if values is not None:
            target = values / self.value_scale
            residual = cache["value"] - target
            value_loss = self.value_weight * float((residual ** 2).mean())
            loss += value_loss
            metrics["value"] = value_loss
            d_value = (self.value_weight * 2.0 / n) * residual
            grads["Wv2"] += cache["v1"].T @ d_value[:, None]
            grads["bv2"] += d_value.sum(keepdims=True)
            d_v1 = (d_value[:, None] @ p["Wv2"].T) * (cache["v1"] > 0)
            grads["Wv1"] += cache["joined"].T @ d_v1
            grads["bv1"] += d_v1.sum(axis=0)
            d_z2 = d_z2 + (d_v1 @ p["Wv1"].T[:, : self.hidden])

        if thresholds is not None:
            # Regress the force threshold only where a push is possible at all; the
            # threshold is undefined (infinite) otherwise, so those samples are masked
            # rather than given an arbitrary target.
            if threshold_mask is None:
                threshold_mask = np.ones_like(thresholds)
            mask = threshold_mask.astype(bool)
            if mask.any():
                target_t = thresholds[mask] / self.threshold_scale
                residual = cache["threshold"][mask] - target_t
                threshold_loss = self.threshold_weight * float((residual ** 2).mean())
                loss += threshold_loss
                metrics["threshold"] = threshold_loss
                d_threshold = np.zeros_like(cache["threshold"])
                d_threshold[mask] = (
                    self.threshold_weight * 2.0 / max(1, int(mask.sum()))
                ) * residual
                grads["Wt2"] += cache["t1"].T @ d_threshold[:, None]
                grads["bt2"] += d_threshold.sum(keepdims=True)
                d_t1 = (d_threshold[:, None] @ p["Wt2"].T) * (cache["t1"] > 0)
                grads["Wt1"] += cache["joined"].T @ d_t1
                grads["bt1"] += d_t1.sum(axis=0)
                d_z2 = d_z2 + (d_t1 @ p["Wt1"].T[:, : self.hidden])

        if physics_index is not None:
            phys_logit = cache["phys_logit"]
            shifted = phys_logit - phys_logit.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            prob_p = exp / exp.sum(axis=1, keepdims=True)
            ce = -np.log(prob_p[np.arange(n), physics_index] + eps)
            physics_loss = self.physics_weight * float(ce.sum()) / n
            loss += physics_loss
            metrics["physics"] = physics_loss
            d_phys = (self.physics_weight / n) * prob_p
            d_phys[np.arange(n), physics_index] -= self.physics_weight / n
            grads["Wp"] += cache["z2"].T @ d_phys
            grads["bp"] += d_phys.sum(axis=0)
            d_z2 = d_z2 + d_phys @ p["Wp"].T

        if pose_index is not None:
            pose_logit = cache["pose_logit"]
            shifted = pose_logit - pose_logit.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            prob_q = exp / exp.sum(axis=1, keepdims=True)
            ce = -np.log(prob_q[np.arange(n), pose_index] + eps)
            pose_loss = self.pose_weight * float(ce.sum()) / n
            loss += pose_loss
            metrics["pose"] = pose_loss
            d_pose = (self.pose_weight / n) * prob_q
            d_pose[np.arange(n), pose_index] -= self.pose_weight / n
            grads["Wq"] += cache["z2"].T @ d_pose
            grads["bq"] += d_pose.sum(axis=0)
            d_z2 = d_z2 + d_pose @ p["Wq"].T

        d_z2 = d_z2 * (cache["z2"] > 0)
        grads["W2"] += cache["z1"].T @ d_z2
        grads["b2"] += d_z2.sum(axis=0)
        d_z1 = (d_z2 @ p["W2"].T) * (cache["z1"] > 0)
        grads["W1"] += obs.T @ d_z1
        grads["b1"] += d_z1.sum(axis=0)
        return loss, grads, metrics

    def predict_outcomes(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Probability of each action-outcome channel, shape ``(N, n_outcomes)``."""

        logit = self.forward(obs, action)["outcome_logit"]
        return _sigmoid(logit)

    def predict_feasibility(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Channel 0 only: the probability that the object moves."""

        return self.predict_outcomes(obs, action)[:, 0]

    def predict_threshold(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predicted force threshold, in the same units as the training target."""

        return self.forward(obs, action)["threshold"] * self.threshold_scale

    def predict_value(self, obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predicted action value, in the same units as the training target."""

        return self.forward(obs, action)["value"] * self.value_scale

    def predict_physics(self, obs: np.ndarray) -> np.ndarray:
        logit = self.forward(obs, np.zeros((obs.shape[0], self.action_dim)))["phys_logit"]
        shifted = logit - logit.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class TrainHistory:
    losses: List[float] = field(default_factory=list)
    feasibility: List[float] = field(default_factory=list)
    physics: List[float] = field(default_factory=list)
    pose: List[float] = field(default_factory=list)
    value: List[float] = field(default_factory=list)
    threshold: List[float] = field(default_factory=list)


def train(
    model: MultiHeadMLP,
    obs: np.ndarray,
    action: np.ndarray,
    targets: np.ndarray,
    physics_index: Optional[np.ndarray] = None,
    pose_index: Optional[np.ndarray] = None,
    values: Optional[np.ndarray] = None,
    thresholds: Optional[np.ndarray] = None,
    threshold_mask: Optional[np.ndarray] = None,
    *,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 3e-3,
    seed: int = 0,
    verbose: bool = False,
) -> TrainHistory:
    """Adam training loop over the three heads."""

    rng = np.random.default_rng(seed)
    state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in model.params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    n = obs.shape[0]
    history = TrainHistory()

    for epoch in range(epochs):
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            loss, grads, metrics = model.loss_and_grads(
                obs[idx], action[idx], targets[idx],
                None if physics_index is None else physics_index[idx],
                None if pose_index is None else pose_index[idx],
                None if values is None else values[idx],
                None if thresholds is None else thresholds[idx],
                None if threshold_mask is None else threshold_mask[idx],
            )
            step += 1
            for key, grad in grads.items():
                m, v = state[key]
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                model.params[key] = model.params[key] - lr * m_hat / (np.sqrt(v_hat) + eps)
                state[key] = (m, v)
            epoch_loss += loss
            batches += 1
            history.feasibility.append(metrics.get("feasibility", 0.0))
            history.physics.append(metrics.get("physics", 0.0))
            history.pose.append(metrics.get("pose", 0.0))
            history.value.append(metrics.get("value", 0.0))
            history.threshold.append(metrics.get("threshold", 0.0))
        history.losses.append(epoch_loss / max(1, batches))
        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d} loss {history.losses[-1]:.4f}")
    return history
