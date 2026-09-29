"""A factored world model: an invariant parameter belief, a dynamic state, and an explicit
observation model.

The earlier world models were one undifferentiated latent that had to do three different jobs at
once, and the structure they needed was missing in three specific ways:

1. **No parameter belief.** The hidden contact parameters are fixed for the whole episode, so
   what they need is *inference*, not a transition. Here the invariant part is an explicit
   16-way distribution over the hidden parameter combination, and the transition never touches
   it.
2. **No observation model.** A belief can only be updated through ``P(y | a, s, θ)`` -- and that
   likelihood is **conditioned on the action**, because the same feedback symbol means opposite
   things after different actions (``drove`` after ``push`` means the object moved, so the object
   is not latched; ``drove`` after ``release`` means the release worked, so it is). Without the
   observation model there is no legal belief update at all, which is why parameter estimation
   could not be learned before.
3. **No belief update during imagination.** Planning has to branch on the feedback it imagines
   arriving and update the belief accordingly, or "release then push" collapses into an
   unconditional pair of actions.

The model therefore factors as

    b(θ)      = softmax(theta_head(observation))          invariant, updated only by evidence
    s_t       = softmax(dyn_head(observation))            dynamic: released + object cell
    s_{t+1}   = transition(s_t, a, θ)                     the action changes the dynamic state
    y_{t+1}   = observation_model(a, s_{t+1}, θ)          the action determines the evidence

with the belief update done explicitly by enumerating the 16 parameter settings:

    b'(θ) ∝ P(y | a, s_{t+1}, θ) · b(θ)

Everything above is inspectable: the belief is a distribution over declared parameter values, the
dynamic state is ``released`` plus a distribution over the 36 object cells, and the observation
model is a distribution over the declared feedback and public-event symbols.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..grid.dynamics import FEEDBACK_SYMBOLS
from .autograd import Adam, MLP, Tensor, binary_cross_entropy, concat, mse, softmax, softmax_cross_entropy
from .features import PUBLIC_SYMBOLS
from .world_model import N_ACTIONS, N_CELLS, N_PHYSICS, POSITIVE_WEIGHT, Trajectory, Transition

#: The observation symbols the model must predict: the robot's own private feedback first, then
#: the public event. The public event is observable by both sides; the feedback is private.
OBSERVATION_SYMBOLS: Tuple[str, ...] = tuple(FEEDBACK_SYMBOLS) + tuple(PUBLIC_SYMBOLS)
N_OBSERVATION_SYMBOLS = len(OBSERVATION_SYMBOLS)
#: The dynamic state: released (1) + object cell distribution (36) + both robot positions (4,
#: normalised). The positions matter because whether a push does anything depends on who is
#: standing where: without them the observation model cannot tell contact from no contact and has
#: to hedge, which made it confidently wrong about the constraint mode and the resistance.
N_POSES = 4
DYNAMIC_DIM = 1 + N_CELLS + N_POSES
_CELL_SLICE = (1, 1 + N_CELLS)
_POSE_SLICE = (1 + N_CELLS, DYNAMIC_DIM)
_POSE_SCALE = 5.0


def observation_symbol_index(transition: Transition) -> int:
    """The symbol a real step produced, as an index into :data:`OBSERVATION_SYMBOLS`."""

    if transition.feedback_symbol in FEEDBACK_SYMBOLS:
        return FEEDBACK_SYMBOLS.index(transition.feedback_symbol)
    raise ValueError(transition.feedback_symbol)


def public_symbol_index(transition: Transition) -> int:
    return len(FEEDBACK_SYMBOLS) + (0 if transition.moved else 1)


@dataclass
class BeliefWorldModel:
    """The factored model described in the module docstring."""

    hidden: int = 96
    seed: int = 0
    observation_dim: int = 188

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # invariant: a distribution over the 16 hidden parameter combinations
        self.theta_head = MLP([self.observation_dim, self.hidden, N_PHYSICS], rng, name="theta")
        # dynamic: released plus the object's cell, both read off the current observation
        self.dyn_head = MLP([self.observation_dim, self.hidden, DYNAMIC_DIM], rng, name="dyn")
        # the action changes the dynamic state; the parameters condition how
        self.transition = MLP([DYNAMIC_DIM + N_ACTIONS + N_PHYSICS, self.hidden, DYNAMIC_DIM], rng, name="trn")
        # the action determines the evidence, and the parameters condition what it means
        self.observation_model = MLP(
            [N_ACTIONS + DYNAMIC_DIM + N_PHYSICS, self.hidden, N_OBSERVATION_SYMBOLS],
            rng,
            name="obsmodel",
        )
        self.cost_head = MLP([N_ACTIONS + DYNAMIC_DIM + N_PHYSICS, self.hidden, 1], rng, name="cost")
        self._theta_eye = np.eye(N_PHYSICS)

    # -- parameters --------------------------------------------------------
    def parameters(self):
        out: List[Tuple[str, Tensor]] = []
        for module in (
            self.theta_head, self.dyn_head, self.transition,
            self.observation_model, self.cost_head,
        ):
            out.extend(module.parameters())
        return out

    # -- inference ---------------------------------------------------------
    def parameter_belief(self, observation: np.ndarray) -> np.ndarray:
        """``b(θ)``: a distribution over the 16 hidden parameter combinations."""

        logits = self.theta_head(Tensor(np.atleast_2d(observation))).data
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def dynamic_state(self, observation: np.ndarray) -> np.ndarray:
        """``s_t``: released probability plus the object-cell distribution."""

        logits = self.dyn_head(Tensor(np.atleast_2d(observation))).data
        return _dynamic_from_logits(logits)

    def step(
        self, dynamic: np.ndarray, action: int, belief: np.ndarray
    ) -> np.ndarray:
        """``s_{t+1}`` from ``s_t``, the action and the current parameter belief.

        The transition is evaluated under **every** parameter setting and the results are mixed
        by the belief. Feeding the belief's expectation as if it were a parameter vector would be
        a train/inference mismatch -- the transition is trained on one-hot parameters -- and it
        silently produced a confident zero here: with the true parameters the model predicts
        ``release`` sets ``released`` with probability 1.000, and with an expected-parameter input
        it predicted 0.000.
        """

        dynamic = np.atleast_2d(dynamic)
        belief = np.atleast_2d(belief)
        count = dynamic.shape[0]
        action_vector = np.zeros((count, N_ACTIONS))
        action_vector[:, action] = 1.0
        logits = self.transition(
            concat(
                [
                    Tensor(np.repeat(dynamic, N_PHYSICS, axis=0)),
                    Tensor(np.repeat(action_vector, N_PHYSICS, axis=0)),
                    Tensor(np.tile(self._theta_eye, (count, 1))),
                ]
            )
        ).data.reshape(count, N_PHYSICS, DYNAMIC_DIM)
        states = _dynamic_from_logits(logits.reshape(count * N_PHYSICS, DYNAMIC_DIM))
        states = states.reshape(count, N_PHYSICS, DYNAMIC_DIM)
        return np.einsum("nt,ntd->nd", belief, states)

    def step_actions(
        self, dynamic: np.ndarray, belief: np.ndarray
    ) -> np.ndarray:
        """Every action from one dynamic state, in a single batched forward pass.

        A search evaluates thousands of (state, action) pairs per decision, so doing them one at
        a time in NumPy is the difference between a usable planner and an unusable one. The batch
        is laid out as (action, parameter setting).
        """

        dynamic = np.atleast_2d(dynamic)
        belief = np.atleast_2d(belief)
        action_block = np.concatenate(
            [
                np.repeat(np.eye(N_ACTIONS)[[action]], N_PHYSICS, axis=0)
                for action in range(N_ACTIONS)
            ],
            axis=0,
        )
        dynamic_block = np.tile(
            np.repeat(dynamic, N_PHYSICS, axis=0), (N_ACTIONS, 1)
        )
        theta_block = np.tile(self._theta_eye, (N_ACTIONS, 1))
        logits = self.transition(
            concat([Tensor(dynamic_block), Tensor(action_block), Tensor(theta_block)])
        ).data.reshape(N_ACTIONS, N_PHYSICS, DYNAMIC_DIM)
        states = _dynamic_from_logits(
            logits.reshape(N_ACTIONS * N_PHYSICS, DYNAMIC_DIM)
        ).reshape(N_ACTIONS, N_PHYSICS, DYNAMIC_DIM)
        return np.einsum("nt,atd->ad", belief, states)

    def cost_actions(
        self, states: np.ndarray, belief: np.ndarray
    ) -> np.ndarray:
        """Predicted cost of each action, told the state that action leads to.

        Row ``a`` pairs action ``a`` with ``states[a]``. The cost head is conditioned on the
        parameters like everything else, so it is evaluated under **all 16 settings** and mixed by
        the belief -- feeding the belief's expectation would be the same out-of-distribution
        mistake that made the transition return a confident zero.
        """

        states = np.atleast_2d(states)
        belief = np.atleast_2d(belief)
        count = states.shape[0]
        action_block = np.concatenate(
            [
                np.repeat(np.eye(N_ACTIONS)[[action]], N_PHYSICS, axis=0)
                for action in range(count)
            ],
            axis=0,
        )
        state_block = np.repeat(states, N_PHYSICS, axis=0)
        theta_block = np.tile(self._theta_eye, (count, 1))
        predicted = self.cost_head(
            concat([Tensor(action_block), Tensor(state_block), Tensor(theta_block)])
        ).data[:, 0].reshape(count, N_PHYSICS)
        return predicted @ belief[0]

    def step_per_theta(self, dynamic: np.ndarray, action: int) -> np.ndarray:
        """The next state under **each** parameter setting, one row per setting.

        The transition is conditioned on the parameters, so each hypothesis predicts a different
        next state. Evaluating a likelihood on a state computed under the belief *mixture* is out
        of distribution -- the observation model is trained on states computed under one-hot
        parameters -- and it was the reason the belief saturated at 0.87 after evidence that
        should have ruled a hypothesis out entirely.
        """

        dynamic = np.atleast_2d(dynamic)
        count = dynamic.shape[0]
        action_vector = np.zeros((count * N_PHYSICS, N_ACTIONS))
        action_vector[:, action] = 1.0
        logits = self.transition(
            concat(
                [
                    Tensor(np.repeat(dynamic, N_PHYSICS, axis=0)),
                    Tensor(action_vector),
                    Tensor(np.tile(self._theta_eye, (count, 1))),
                ]
            )
        ).data
        return _dynamic_from_logits(logits)

    def update_belief(
        self,
        belief: np.ndarray,
        prior_state: np.ndarray,
        action: int,
        symbol: int,
    ) -> np.ndarray:
        """The Bayes update, with each hypothesis predicting its own next state.

        ``b'(θ) ∝ P(y | a, s_θ', θ) b(θ)`` where ``s_θ' = T(s, a, θ)``. Conditioning every
        hypothesis on a single mixture state instead is the mistake described in
        :meth:`step_per_theta`.
        """

        belief = np.atleast_2d(belief)
        prior_state = np.atleast_2d(prior_state)
        count = prior_state.shape[0]
        next_states = self.step_per_theta(prior_state, action)   # (count*16, D)
        action_vector = np.zeros((count * N_PHYSICS, N_ACTIONS))
        action_vector[:, action] = 1.0
        logits = self.observation_model(
            concat(
                [
                    Tensor(action_vector),
                    Tensor(next_states),
                    Tensor(np.tile(self._theta_eye, (count, 1))),
                ]
            )
        ).data.reshape(count, N_PHYSICS, N_OBSERVATION_SYMBOLS)
        likelihood = _softmax_rows(
            logits.reshape(count * N_PHYSICS, N_OBSERVATION_SYMBOLS)
        ).reshape(count, N_PHYSICS, N_OBSERVATION_SYMBOLS)[:, :, symbol]
        posterior = likelihood * belief
        total = posterior.sum(axis=1, keepdims=True)
        safe = np.where(total > 1e-12, total, 1.0)
        posterior = posterior / safe
        return np.where(total > 1e-12, posterior, belief)

    def observe(
        self, dynamic: np.ndarray, action: int, belief: np.ndarray
    ) -> np.ndarray:
        """``P(y | a, s, θ)`` marginalised over the belief:

        row ``i`` of the result is the distribution over observation symbols for the ``i``-th
        parameter setting, so the caller can both marginalise and condition.
        """

        dynamic = np.atleast_2d(dynamic)
        count = dynamic.shape[0]
        action_vector = np.zeros((count, N_ACTIONS))
        action_vector[:, action] = 1.0
        # evaluate the likelihood under every parameter setting at once
        logits = self.observation_model(
            concat(
                [
                    Tensor(np.repeat(action_vector, N_PHYSICS, axis=0)),
                    Tensor(np.repeat(dynamic, N_PHYSICS, axis=0)),
                    Tensor(np.tile(self._theta_eye, (count, 1))),
                ]
            )
        ).data
        return _softmax_rows(logits).reshape(count, N_PHYSICS, N_OBSERVATION_SYMBOLS)

    def predict_outcome(
        self, dynamic: np.ndarray, action: int, belief: np.ndarray
    ) -> np.ndarray:
        """``P(y | a, s, b)``, the belief-marginalised prediction of the next symbol."""

        per_theta = self.observe(dynamic, action, belief)
        belief = np.atleast_2d(belief)
        return np.einsum("nt,ntk->nk", belief, per_theta)


@dataclass
class BeliefTrainingHistory:
    losses: List[float] = field(default_factory=list)
    parts: List[Dict[str, float]] = field(default_factory=list)


def train_belief_model(
    model: BeliefWorldModel,
    trajectories: Sequence[Trajectory],
    *,
    rollout_length: int = 3,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 1e-2,
    seed: int = 0,
    verbose: bool = False,
) -> BeliefTrainingHistory:
    """Train all four parts jointly, with the transition and observation model rolled forward.

    The transition and the observation model are conditioned on the **true** parameter one-hot
    during training (teacher forcing). That is deliberate: the point is to learn the transition
    and the likelihood *given* the parameters, and a wrong belief at training time would blur
    exactly the structure being learned. The belief is evaluated separately, by whether the Bayes
    update built on the learned likelihood concentrates on the truth.
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
    history = BeliefTrainingHistory()
    theta_eye = np.eye(N_PHYSICS)

    for epoch in range(epochs):
        order = rng.permutation(len(windows))
        epoch_parts: Dict[str, float] = {}
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            batch = [windows[i] for i in order[start : start + batch_size]]
            pooled = list(zip(*batch))
            first = pooled[0]
            observation = np.stack([step.observation for step in first])
            theta_true = theta_eye[np.asarray([step.physics for step in first])]

            losses = []
            metrics: Dict[str, float] = {}
            # invariant belief and current dynamic state, straight from the observation
            theta_logits = model.theta_head(Tensor(observation))
            losses.append(("theta", softmax_cross_entropy(
                theta_logits, np.asarray([step.physics for step in first])), 1.0))
            dyn_logits = model.dyn_head(Tensor(observation))
            losses.append(("released", binary_cross_entropy(
                logits_slice(dyn_logits, 0, 1),
                np.asarray([[float(step.released)] for step in first]), POSITIVE_WEIGHT), 1.0))
            losses.append(("cell", softmax_cross_entropy(
                logits_slice(dyn_logits, _CELL_SLICE[0], _CELL_SLICE[1]),
                np.asarray([step.object_cell for step in first])), 1.0))
            losses.append(("pose", mse(
                logits_slice(dyn_logits, _POSE_SLICE[0], _POSE_SLICE[1]),
                np.asarray([step.robot_poses for step in first]) / _POSE_SCALE), 1.0))

            dynamic = _dynamic_tensor(dyn_logits)
            for steps in pooled:
                action = np.asarray([step.action for step in steps])
                action_vector = np.zeros((len(steps), N_ACTIONS))
                action_vector[np.arange(len(steps)), action] = 1.0
                dynamic = model.transition(
                    concat([dynamic, Tensor(action_vector), Tensor(theta_true)])
                )
                released_next = np.asarray([[float(step.next_released)] for step in steps])
                cell_next = np.asarray([step.next_object_cell for step in steps])
                symbols = np.asarray([observation_symbol_index(step) for step in steps])
                costs = np.asarray([[step.cost] for step in steps])
                losses.extend(
                    [
                        ("released_next", binary_cross_entropy(
                            logits_slice(dynamic, 0, 1), released_next, POSITIVE_WEIGHT), 1.0),
                        ("cell_next", softmax_cross_entropy(
                            logits_slice(dynamic, _CELL_SLICE[0], _CELL_SLICE[1]), cell_next), 1.0),
                        ("pose_next", mse(
                            logits_slice(dynamic, _POSE_SLICE[0], _POSE_SLICE[1]),
                            np.asarray([step.next_robot_poses for step in steps]) / _POSE_SCALE), 1.0),
                    ]
                )
                # The observation model and the cost head must consume the **processed** dynamic
                # state, the same thing inference feeds them. Training them on the raw transition
                # logits is a silent train/inference mismatch: the loss looks perfect and the
                # model is queried with different numbers.
                dynamic = _dynamic_tensor(dynamic)
                losses.extend(
                    [
                        ("observation", softmax_cross_entropy(
                            model.observation_model(concat([
                                Tensor(action_vector), dynamic, Tensor(theta_true)
                            ])), symbols), 1.0),
                        ("cost", mse(
                            model.cost_head(concat([
                                Tensor(action_vector), dynamic, Tensor(theta_true)
                            ])), costs), 1.0),
                    ]
                )

            total = None
            for name, loss, weight in losses:
                term = loss if weight == 1.0 else _scaled(loss, weight)
                total = term if total is None else _plus(total, term)
                metrics[name] = float(loss.data) * weight
            optimiser.zero_grad()
            total.backward()
            optimiser.update()
            epoch_loss += float(total.data)
            for name, value in metrics.items():
                epoch_parts[name] = epoch_parts.get(name, 0.0) + value
            batches += 1
        history.losses.append(epoch_loss / max(1, batches))
        history.parts.append({k: v / max(1, batches) for k, v in epoch_parts.items()})
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d} loss {history.losses[-1]:.3f} "
                  + " ".join(f"{k}={v:.3f}" for k, v in sorted(history.parts[-1].items())))
    return history


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def logits_slice(tensor: Tensor, start: int, end: int) -> Tensor:
    """A differentiable column slice, so one head can supply several distributions."""

    out = Tensor(tensor.data[:, start:end], (tensor,))

    def _backward(grad: np.ndarray) -> None:
        full = np.zeros_like(tensor.data)
        full[:, start:end] = grad
        tensor.grad = tensor.grad + full

    out._backward = _backward
    return out


def _dynamic_from_logits(logits: np.ndarray) -> np.ndarray:
    """Split a raw head output into (released probability, cell distribution, robot positions)."""

    released = _sigmoid(logits[:, :1])
    cells = _softmax_rows(logits[:, _CELL_SLICE[0] : _CELL_SLICE[1]])
    poses = logits[:, _POSE_SLICE[0] : _POSE_SLICE[1]]
    return np.concatenate([released, cells, poses], axis=1)


def _dynamic_tensor(logits: Tensor) -> Tensor:
    """The same split, differentiable, for use inside the training graph."""

    from .autograd import sigmoid

    released = sigmoid(logits_slice(logits, 0, 1))
    cells = softmax(logits_slice(logits, _CELL_SLICE[0], _CELL_SLICE[1]))
    poses = logits_slice(logits, _POSE_SLICE[0], _POSE_SLICE[1])
    return concat([released, cells, poses])


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    out[~positive] = exp / (1.0 + exp)
    return out


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


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_belief_model(model: BeliefWorldModel, path: str) -> None:
    arrays = {name: tensor.data for name, tensor in model.parameters()}
    arrays["__kind__"] = np.asarray(["belief"])
    arrays["__hidden__"] = np.asarray([model.hidden])
    arrays["__seed__"] = np.asarray([model.seed])
    np.savez(path, **arrays)


def load_belief_model(path: str) -> BeliefWorldModel:
    blob = np.load(path, allow_pickle=False)
    model = BeliefWorldModel(hidden=int(blob["__hidden__"][0]), seed=int(blob["__seed__"][0]))
    for name, tensor in model.parameters():
        tensor.data = np.array(blob[name])
    return model
