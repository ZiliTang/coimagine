"""Dataset for the deterministic grid point-prediction baseline (CONTRACT B2).

Sampling policy (fixed in advance, never conditioned on model failures):

* half of the initial states are ordinary legal stances (no robot in contact with the
  object), half are legal contact stances (at least one robot is push-qualified);
* half of the actions are uniform over all 81 joint actions, half come from fixed templates
  (single push with the other waiting, same-direction double push, move actions);
* short real trajectories are rolled out from each initial state, so states are reached by
  the true dynamics rather than concatenated from invalid positions;
* samples are deduplicated on ``(map, state, joint_action)`` and capped at the budget;
* train/val/test split by a fixed hash of ``(map, state)`` canonical key, 80/10/10, so all
  actions of one state live in one split.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .contract import (
    DIRECTION,
    MOVE_ACTIONS,
    N_JOINT_ACTIONS,
    PUSH_ACTIONS,
    WAIT_ACTION,
    GridMap,
    GridState,
    all_valid_states,
    encode_joint,
    is_goal,
    object_cells,
)
from .dynamics import transition_exact

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DataConfig:
    budget: int = 30_000
    seed: int = 202_610_12
    trajectory_length: int = 6
    uniform_action_fraction: float = 0.5
    #: template probabilities among the non-uniform half: single push, double push, move
    template_probabilities: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    contact_fraction: float = 0.5
    split_seed: int = 202_610_12

    def as_dict(self) -> Dict[str, object]:
        return {
            "budget": self.budget,
            "seed": self.seed,
            "trajectory_length": self.trajectory_length,
            "uniform_action_fraction": self.uniform_action_fraction,
            "template_probabilities": list(self.template_probabilities),
            "contact_fraction": self.contact_fraction,
            "split_seed": self.split_seed,
        }


@dataclass(frozen=True)
class TransitionSample:
    map_name: str
    state: GridState
    joint_action: int
    next_state: GridState
    object_moved: bool
    robot_moved: bool
    rollback: bool
    invalid_actions: int


@dataclass
class Dataset:
    samples: List[TransitionSample] = field(default_factory=list)
    state_split: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    config: Optional[DataConfig] = None
    witness: Dict[str, object] = field(default_factory=dict)

    def split_of(self, map_name: str, state: GridState) -> str:
        return self.state_split[state.key(map_name)]


def canonical_key(map_name: str, state: GridState) -> str:
    return state.key(map_name)


def split_for(map_name: str, state: GridState, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{map_name}:{state.key(map_name)}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def has_object_contact(state: GridState) -> bool:
    obj = object_cells(state.object_anchor)
    for position in state.robots():
        if position in obj:
            continue
        if any((position[0] + d[0], position[1] + d[1]) in obj for d in DIRECTION.values()):
            return True
    return False


def qualified_pushers(state: GridState) -> List[Tuple[int, Tuple[int, int]]]:
    """(robot_id, direction-delta) pairs for which ``push_d`` is geometrically qualified."""

    obj = object_cells(state.object_anchor)
    out: List[Tuple[int, Tuple[int, int]]] = []
    for robot_id, position in enumerate(state.robots()):
        if position in obj:
            continue
        for d in DIRECTION.values():
            if (position[0] + d[0], position[1] + d[1]) in obj:
                out.append((robot_id, d))
    return out


def _push_action(direction: Tuple[int, int]) -> int:
    return {DIRECTION[a]: a for a in PUSH_ACTIONS}[direction]


def sample_joint_action(
    state: GridState, rng: random.Random, config: DataConfig
) -> int:
    if rng.random() < config.uniform_action_fraction:
        return rng.randrange(N_JOINT_ACTIONS)
    pushers = qualified_pushers(state)
    single_p, double_p, _move_p = config.template_probabilities
    roll = rng.random()
    if roll < single_p and pushers:
        robot_id, direction = rng.choice(pushers)
        action = _push_action(direction)
        return encode_joint(action if robot_id == 0 else WAIT_ACTION, action if robot_id == 1 else WAIT_ACTION)
    if roll < single_p + double_p and pushers:
        _robot_id, direction = rng.choice(pushers)
        action = _push_action(direction)
        return encode_joint(action, action)
    # move template: each robot moves with probability 0.5, otherwise waits
    actions = []
    for _ in (0, 1):
        if rng.random() < 0.5:
            actions.append(rng.choice(MOVE_ACTIONS))
        else:
            actions.append(WAIT_ACTION)
    return encode_joint(actions[0], actions[1])


def _sample_initial_states(
    grid_map: GridMap, count: int, rng: random.Random, config: DataConfig
) -> List[GridState]:
    candidates = [state for state in all_valid_states(grid_map) if not is_goal(state, grid_map)]
    contact = [state for state in candidates if has_object_contact(state)]
    normal = [state for state in candidates if not has_object_contact(state)]
    n_contact = int(round(count * config.contact_fraction))
    n_normal = count - n_contact
    picks: List[GridState] = []
    if normal:
        picks.extend(rng.choice(normal) for _ in range(n_normal))
    if contact:
        picks.extend(rng.choice(contact) for _ in range(n_contact))
    if not picks:
        picks = [rng.choice(candidates)]
    rng.shuffle(picks)
    return picks


def generate_dataset(
    maps: Sequence[GridMap], config: Optional[DataConfig] = None
) -> Dataset:
    config = config or DataConfig()
    rng = random.Random(config.seed)
    seen: Dict[Tuple[str, str, int], TransitionSample] = {}
    state_split: Dict[str, str] = {}
    # enough initials that the trajectory product can reach the budget
    per_map = max(1, (config.budget // max(1, config.trajectory_length)) // max(1, len(maps)) + 1)
    attempts = 0
    while len(seen) < config.budget and attempts < 4:
        attempts += 1
        for grid_map in maps:
            initials = _sample_initial_states(grid_map, per_map, rng, config)
            for initial in initials:
                if len(seen) >= config.budget:
                    break
                state = initial
                for _ in range(config.trajectory_length):
                    if len(seen) >= config.budget or is_goal(state, grid_map):
                        break
                    joint = sample_joint_action(state, rng, config)
                    next_state, info = transition_exact(state, joint, grid_map)
                    key = (grid_map.name, state.key(grid_map.name), joint)
                    if key not in seen:
                        seen[key] = TransitionSample(
                            map_name=grid_map.name,
                            state=state,
                            joint_action=joint,
                            next_state=next_state,
                            object_moved=info.object_moved,
                            robot_moved=info.robots_moved > 0,
                            rollback=info.rollback,
                            invalid_actions=info.invalid_actions,
                        )
                        state_split.setdefault(
                            state.key(grid_map.name), split_for(grid_map.name, state, config.split_seed)
                        )
                    state = next_state
        if len(seen) >= config.budget:
            break

    samples = list(seen.values())
    stats: Dict[str, Dict[str, float]] = {split: {"transitions": 0.0, "states": 0.0,
                                                  "object_move": 0.0, "robot_move": 0.0,
                                                  "rollback": 0.0, "invalid_actions": 0.0}
                                          for split in SPLITS}
    states_by_split: Dict[str, set] = {split: set() for split in SPLITS}
    for sample in samples:
        split = state_split[sample.state.key(sample.map_name)]
        row = stats[split]
        row["transitions"] += 1
        states_by_split[split].add(sample.state.key(sample.map_name))
        row["object_move"] += int(sample.object_moved)
        row["robot_move"] += int(sample.robot_moved)
        row["rollback"] += int(sample.rollback)
        row["invalid_actions"] += int(sample.invalid_actions > 0)
    for split in SPLITS:
        row = stats[split]
        row["states"] = float(len(states_by_split[split]))
        n = max(1.0, row["transitions"])
        row["object_move_rate"] = row.pop("object_move") / n
        row["robot_move_rate"] = row.pop("robot_move") / n
        row["rollback_rate"] = row.pop("rollback") / n
        row["invalid_action_rate"] = row.pop("invalid_actions") / n

    return Dataset(samples=samples, state_split=state_split, stats=stats, config=config)


def save_dataset(dataset: Dataset, directory: Path) -> Dict[str, str]:
    """Persist config, stats, split list and a compact sample table; return SHA256s."""

    import hashlib
    import json

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(dataset.config.as_dict() if dataset.config else {}, indent=2))
    (directory / "stats.json").write_text(json.dumps(dataset.stats, indent=2))
    with (directory / "splits.jsonl").open("w") as handle:
        for key, split in sorted(dataset.state_split.items()):
            handle.write(json.dumps({"state_key": key, "split": split}) + "\n")
    hashes: Dict[str, str] = {}
    with (directory / "samples.jsonl").open("w") as handle:
        for sample in dataset.samples:
            handle.write(json.dumps({
                "map": sample.map_name,
                "state": {"robot0": sample.state.robot0, "robot1": sample.state.robot1, "object": sample.state.object_anchor},
                "joint": sample.joint_action,
                "next": {"robot0": sample.next_state.robot0, "robot1": sample.next_state.robot1, "object": sample.next_state.object_anchor},
                "object_moved": sample.object_moved,
                "robot_moved": sample.robot_moved,
                "rollback": sample.rollback,
                "invalid_actions": sample.invalid_actions,
            }) + "\n")
    for name in ("config.json", "stats.json", "splits.jsonl", "samples.jsonl"):
        digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        hashes[name] = digest
    (directory / "manifest.json").write_text(json.dumps(hashes, indent=2))
    return hashes


def load_dataset(directory: Path) -> Dataset:
    import json

    config = DataConfig(**json.loads((directory / "config.json").read_text()))
    dataset = generate_dataset(default_maps_local(), config)
    return dataset


def default_maps_local():
    from .contract import default_maps

    return default_maps()
