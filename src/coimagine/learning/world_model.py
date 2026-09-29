"""Learned world models for the grid stage: encoder, latent transition, decoders.

Two variants, both trained and compared:

* ``LatentWorldModel`` -- an encoder maps the robot's own observation to a latent, a
  transition maps ``(latent, action)`` to the next latent, and decoder heads read the next
  observation and privileged quantities out of the latent. Multi-step rollouts are trained
  through the transition by re-applying it, which is the whole point of having a latent.
* ``DirectWorldModel`` -- predicts the next observation vector directly from
  ``(observation, action)``, with no bottleneck. The observation is only 188 numbers here, so
  this is a serious alternative and not a straw man.

Both get the same data, the same budget and the same privileged supervision, so a difference
between them is attributable to the bottleneck rather than to capacity or data.

The observation is built from the robot's own scan and history only; nothing privileged is an
input. Privileged quantities (the object's cell, the contact physics) appear as decoder targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .autograd import (
    Adam,
    MLP,
    Tensor,
    binary_cross_entropy,
    concat,
    mse,
    softmax_cross_entropy,
    weighted_mse,
)
from .features import ACTION_IDS, OBSERVATION_DIM

N_ACTIONS = len(ACTION_IDS)
#: Roughly one step in thirteen moves the object, so the "did it move" head needs a positive
#: weight or it minimises loss by always answering no -- which is what it did before.
POSITIVE_WEIGHT = 8.0
#: Extra weight on transitions where the object moved, applied to the pose and observation
#: heads. The dynamics live in those rows.
MOVE_WEIGHT = 30.0
N_PHYSICS = 16
N_CELLS = 36


def one_hot_actions(indices: np.ndarray) -> np.ndarray:
    out = np.zeros((len(indices), N_ACTIONS))
    out[np.arange(len(indices)), indices] = 1.0
    return out


@dataclass
class Transition:
    """One simulator step, as the learner sees it."""

    observation: np.ndarray          # the robot's own view at time t
    action: int                      # index into ACTION_IDS
    next_observation: np.ndarray     # the robot's own view at time t+1
    moved: int                       # did the object move
    cost: float                      # the step's action cost
    # Privileged decoder targets. These describe the state **after** the action, because the
    # latent produced by the transition represents that state. An earlier version attached the
    # *current* state's labels to the next latent, which is an off-by-one and makes the target
    # unlearnable whenever the object moves.
    physics: int
    object_cell: int
    next_physics: int = 0
    next_object_cell: int = 0
    #: Whether the object has been released. This is the task's central **hidden** transition:
    #: a successful release leaves the observation byte-identical and only flips this flag, so
    #: without it as a target the model has no way to represent that release did anything.
    next_released: int = 0
    released: int = 0


@dataclass
class Trajectory:
    """A contiguous run of transitions, used for the multi-step rollout loss."""

    transitions: List[Transition] = field(default_factory=list)
    goal_reached: bool = False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class LatentWorldModel:
    latent_dim: int = 32
    hidden: int = 96
    seed: int = 0
    observation_dim: int = OBSERVATION_DIM

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.encoder = MLP([self.observation_dim, self.hidden, self.latent_dim], rng, name="enc")
        self.transition = MLP(
            [self.latent_dim + N_ACTIONS, self.hidden, self.latent_dim], rng, name="trn"
        )
        self.observation_head = MLP(
            [self.latent_dim, self.hidden, self.observation_dim], rng, name="obs"
        )
        self.physics_head = MLP([self.latent_dim, self.hidden, N_PHYSICS], rng, name="phy")
        self.pose_head = MLP([self.latent_dim, self.hidden, N_CELLS], rng, name="pos")
        self.cost_head = MLP([self.latent_dim, self.hidden, 1], rng, name="cost")
        self.moved_head = MLP([self.latent_dim, self.hidden, 1], rng, name="moved")
        self.released_head = MLP([self.latent_dim, self.hidden, 1], rng, name="released")

    # -- parameter access --------------------------------------------------
    def parameters(self):
        out = []
        for module in (
            self.encoder, self.transition, self.observation_head,
            self.physics_head, self.pose_head, self.cost_head, self.moved_head,
            self.released_head,
        ):
            out.extend(module.parameters())
        return out

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Encode one observation or a batch; a single vector is promoted to a batch of one."""

        return self.encoder(Tensor(np.atleast_2d(observation))).data

    def step(self, latent: np.ndarray, action: int) -> np.ndarray:
        """Apply the transition to one latent or a batch."""

        latents = np.atleast_2d(latent)
        action_vector = np.zeros((latents.shape[0], N_ACTIONS))
        action_vector[:, action] = 1.0
        return self.transition(concat([Tensor(latents), Tensor(action_vector)])).data

    def imagine(self, observation: np.ndarray, actions: Sequence[int]) -> List[Dict[str, np.ndarray]]:
        """Imagine forward, returning the same per-step dict as :meth:`predict_next`.

        Both models return this shape so a caller can treat them interchangeably -- and so the
        visualiser can draw every imagined step without knowing which model produced it.
        """

        results: List[Dict[str, np.ndarray]] = []
        latent = self.encode(observation)
        for action in actions:
            latent = self.step(latent, action)
            results.append(
                {
                    "latent": latent,
                    "observation": self.observation_head(Tensor(latent)).data,
                    "physics_logits": self.physics_head(Tensor(latent)).data,
                    "pose_logits": self.pose_head(Tensor(latent)).data,
                    "cost": self.cost_head(Tensor(latent)).data[:, 0],
                    "moved_logit": self.moved_head(Tensor(latent)).data[:, 0],
                }
            )
        return results

    def continue_from(self, predicted: Dict[str, np.ndarray], action: int) -> Dict[str, np.ndarray]:
        """Imagine one more step starting from an already-imagined step.

        The latent is carried forward **in latent space**. Going through the decoded
        observation instead would re-encode the model's own blurry prediction and throw away
        the bottleneck, which is the whole reason the latent exists.
        """

        return self._decode(self.step(predicted["latent"], action))

    def _decode(self, latent: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "latent": latent,
            "observation": self.observation_head(Tensor(latent)).data,
            "physics_logits": self.physics_head(Tensor(latent)).data,
            "pose_logits": self.pose_head(Tensor(latent)).data,
            "cost": self.cost_head(Tensor(latent)).data[:, 0],
            "moved_logit": self.moved_head(Tensor(latent)).data[:, 0],
            "released_logit": self.released_head(Tensor(latent)).data[:, 0],
        }

    def predict_next(self, observation: np.ndarray, action: int) -> Dict[str, np.ndarray]:
        """One imagined step, with every head applied to the predicted latent."""

        return self._decode(self.step(self.encode(observation), action))


@dataclass
class DirectWorldModel:
    hidden: int = 96
    seed: int = 0
    observation_dim: int = OBSERVATION_DIM

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        joined = self.observation_dim + N_ACTIONS
        self.predictor = MLP([joined, self.hidden, self.hidden, self.observation_dim], rng, name="pred")
        self.physics_head = MLP([self.observation_dim, self.hidden, N_PHYSICS], rng, name="phy")
        self.pose_head = MLP([self.observation_dim, self.hidden, N_CELLS], rng, name="pos")
        self.cost_head = MLP([self.observation_dim, self.hidden, 1], rng, name="cost")
        self.moved_head = MLP([self.observation_dim, self.hidden, 1], rng, name="moved")
        self.released_head = MLP([self.observation_dim, self.hidden, 1], rng, name="released")

    def parameters(self):
        out = []
        for module in (
            self.predictor, self.physics_head, self.pose_head,
            self.cost_head, self.moved_head, self.released_head,
        ):
            out.extend(module.parameters())
        return out

    def _joined(self, observation, action: int) -> Tensor:
        """Concatenate an observation (array or graph node) with a one-hot action.

        Accepting either is what lets the same helper build both the training graph, where the
        observation is a prediction we must differentiate through, and the inference path.
        """

        obs_tensor = (
            observation
            if isinstance(observation, Tensor)
            else Tensor(np.atleast_2d(observation))
        )
        count = obs_tensor.data.shape[0]
        action_vector = np.zeros((count, N_ACTIONS))
        action_vector[:, action] = 1.0
        return concat([obs_tensor, Tensor(action_vector)])

    def continue_from(self, predicted: Dict[str, np.ndarray], action: int) -> Dict[str, np.ndarray]:
        """Imagine one more step; without a bottleneck the prediction is the next input."""

        return self.predict_next(predicted["observation"], action)

    def predict_next(self, observation: np.ndarray, action: int) -> Dict[str, np.ndarray]:
        predicted = self.predictor(self._joined(observation, action)).data
        return {
            "observation": predicted,
            "physics_logits": self.physics_head(Tensor(predicted)).data,
            "pose_logits": self.pose_head(Tensor(predicted)).data,
            "cost": self.cost_head(Tensor(predicted)).data[:, 0],
            "moved_logit": self.moved_head(Tensor(predicted)).data[:, 0],
            "released_logit": self.released_head(Tensor(predicted)).data[:, 0],
        }

    def imagine(self, observation: np.ndarray, actions: Sequence[int]) -> List[Dict[str, np.ndarray]]:
        """Feed each prediction back in as the next input -- where drift shows up."""

        results: List[Dict[str, np.ndarray]] = []
        current = observation
        for action in actions:
            predicted = self.predict_next(current, action)
            current = predicted["observation"]
            results.append(predicted)
        return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class WorldModelHistory:
    losses: List[float] = field(default_factory=list)
    #: Per-head loss, averaged over an epoch. Indispensable: a single total can look fine
    #: while one head is stuck at the marginal and another is being sacrificed for it.
    parts: List[Dict[str, float]] = field(default_factory=list)


def _decoder_losses(
    model, representation: Tensor, steps, observation_prediction: Tensor
) -> Tuple[Tensor, Dict[str, float]]:
    """Decoder losses for one imagined step, over a whole batch.

    ``representation`` is what the heads read from -- the latent for the latent model, the
    predicted observation for the direct model -- and it always has one row per example in the
    batch, so every target here is built per row.

    An earlier version zipped the batch-level representation against the batch of transitions
    and therefore computed every example's loss against the *first* example's label. The model
    dutifully learned the marginal, which is what a prediction from noise looks like.
    """

    # Transitions that actually move the object are about a ninth of the data. Without a
    # weight the pose head is minimised by ignoring the action entirely, and the model becomes
    # a state estimator that returns the same next position for every candidate action.
    moved_weight = 1.0 + MOVE_WEIGHT * np.asarray([float(step.moved) for step in steps])
    observation_target = np.stack([step.next_observation for step in steps])
    # Weight the observation loss towards the cells that carry signal: the occupancy plane is
    # mostly zeros, so an unweighted MSE is nearly blind to the few cells holding the object.
    observation_weight = 1.0 + 20.0 * observation_target
    physics_target = np.asarray([step.next_physics for step in steps])
    pose_target = np.asarray([step.next_object_cell for step in steps])
    # Shape (n, 1): the cost head emits one column, so a (n,) target would broadcast to (n, n)
    # and produce a nonsense loss that dominates everything else.
    cost_target = np.asarray([[step.cost] for step in steps])
    moved_target = np.asarray([[float(step.moved)] for step in steps])
    released_target = np.asarray([[float(step.next_released)] for step in steps])

    losses = [
        (
            "observation",
            weighted_mse(
                observation_prediction, observation_target,
                observation_weight * moved_weight[:, None],
            ),
            1.0,
        ),
        ("physics", softmax_cross_entropy(model.physics_head(representation), physics_target), 1.0),
        ("pose", softmax_cross_entropy(model.pose_head(representation), pose_target, moved_weight), 1.0),
        ("cost", mse(model.cost_head(representation), cost_target), 1.0),
        ("moved", binary_cross_entropy(model.moved_head(representation), moved_target, POSITIVE_WEIGHT), 1.0),
        ("released", binary_cross_entropy(model.released_head(representation), released_target, POSITIVE_WEIGHT), 1.0),
    ]
    total = None
    parts: Dict[str, float] = {}
    for name, loss, weight in losses:
        term = loss if weight == 1.0 else _scaled(loss, weight)
        total = term if total is None else _plus(total, term)
        parts[name] = float(loss.data) * weight
    assert total is not None
    return total, parts


def _scaled(tensor: Tensor, factor: float) -> Tensor:
    out = Tensor(tensor.data * factor, (tensor,))

    def _backward(grad: np.ndarray) -> None:
        tensor.grad = tensor.grad + grad * factor

    out._backward = _backward
    return out


def _plus(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data + b.data, (a, b))

    def _backward(grad: np.ndarray) -> None:
        a.grad = a.grad + grad
        b.grad = b.grad + grad

    out._backward = _backward
    return out


def train_world_model(
    model,
    trajectories: Sequence[Trajectory],
    *,
    rollout_length: int = 3,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 0,
    latent_consistency: float = 0.5,
    verbose: bool = False,
) -> WorldModelHistory:
    """Train with the multi-step rollout loss.

    Each example is a window of ``rollout_length`` consecutive transitions. The latent is
    carried through the transition and every decoder is applied at every step, so the model is
    optimised for what a planner actually does -- imagining forward -- rather than for one-step
    accuracy. For the latent model an extra term pulls the imagined latent towards the encoding
    of the true next observation, which keeps the latent from drifting into something the
    transition can predict but the encoders cannot read.
    """

    windows: List[List[Transition]] = []
    for trajectory in trajectories:
        steps = trajectory.transitions
        for start in range(len(steps) - rollout_length + 1):
            windows.append(steps[start : start + rollout_length])
    if not windows:
        raise ValueError("no training windows; collect longer trajectories")

    rng = np.random.default_rng(seed)
    optimiser = Adam(model.parameters(), lr=lr)
    history = WorldModelHistory()
    is_latent = isinstance(model, LatentWorldModel)

    for epoch in range(epochs):
        order = rng.permutation(len(windows))
        epoch_loss = 0.0
        batches = 0
        epoch_parts: Dict[str, float] = {}
        for start in range(0, len(order), batch_size):
            batch = [windows[i] for i in order[start : start + batch_size]]
            loss, parts = _sequential_loss(model, batch, latent_consistency, is_latent)
            optimiser.zero_grad()
            loss.backward()
            optimiser.update()
            epoch_loss += float(loss.data)
            for name, value in parts.items():
                epoch_parts[name] = epoch_parts.get(name, 0.0) + value
            batches += 1
        history.losses.append(epoch_loss / max(1, batches))
        history.parts.append({k: v / max(1, batches) for k, v in epoch_parts.items()})
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d} loss {history.losses[-1]:.5f}")
    return history


def _sequential_loss(
    model, batch: Sequence[Sequence[Transition]], latent_consistency: float, is_latent: bool
) -> Tuple[Tensor, Dict[str, float]]:
    """Loss for a batch of same-length windows.

    Implemented per window because the transition is applied step by step; the batch dimension
    is folded into each call so every step is a matrix operation.
    """

    total: Optional[Tensor] = None
    metrics: Dict[str, float] = {}
    pooled = list(zip(*batch))  # step-major
    if is_latent:
        latent = model.encoder(Tensor(np.stack([step.observation for step in pooled[0]])))
        for index, steps in enumerate(pooled):
            action = np.asarray([step.action for step in steps])
            latent = model.transition(concat([latent, Tensor(one_hot_actions(action))]))
            observation_prediction = model.observation_head(latent)
            parts, head_metrics = _decoder_losses(model, latent, steps, observation_prediction)
            for name, value in head_metrics.items():
                metrics[name] = metrics.get(name, 0.0) + value
            total = parts if total is None else _plus(total, parts)
            if latent_consistency > 0.0:
                # stop-gradient target: the true next observation encoded by the same encoder
                target = model.encoder(Tensor(np.stack([s.next_observation for s in steps]))).data
                total = _plus(total, _scaled(mse(latent, target), latent_consistency))
    else:
        current = np.stack([step.observation for step in pooled[0]])
        for steps in pooled:
            action = np.asarray([step.action for step in steps])
            current = model.predictor(model._joined(current, action))
            parts, head_metrics = _decoder_losses(model, current, steps, current)
            for name, value in head_metrics.items():
                metrics[name] = metrics.get(name, 0.0) + value
            total = parts if total is None else _plus(total, parts)
            current_data = current.data
            current = Tensor(current_data)  # detach between steps: drift is measured, not back-propagated
    assert total is not None
    return total, metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_world_model(model, path: str) -> None:
    """Save parameters plus the architecture description to one ``.npz``."""

    arrays = {name: tensor.data for name, tensor in model.parameters()}
    arrays["__kind__"] = np.asarray(["latent" if isinstance(model, LatentWorldModel) else "direct"])
    arrays["__latent_dim__"] = np.asarray([getattr(model, "latent_dim", 0)])
    arrays["__hidden__"] = np.asarray([model.hidden])
    arrays["__seed__"] = np.asarray([model.seed])
    np.savez(path, **arrays)


def load_world_model(path: str):
    """Rebuild a model and its parameters from :func:`save_world_model`."""

    blob = np.load(path, allow_pickle=False)
    kind = str(blob["__kind__"][0])
    latent_dim = int(blob["__latent_dim__"][0])
    hidden = int(blob["__hidden__"][0])
    seed = int(blob["__seed__"][0])
    model = (
        LatentWorldModel(latent_dim=latent_dim, hidden=hidden, seed=seed)
        if kind == "latent"
        else DirectWorldModel(hidden=hidden, seed=seed)
    )
    for name, tensor in model.parameters():
        tensor.data = np.array(blob[name])
    return model
