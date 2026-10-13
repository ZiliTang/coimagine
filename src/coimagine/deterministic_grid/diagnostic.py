"""Fixed-candidate selection-bias diagnostic (CONTRACT sections 4-6).

The truth for a (map, state, depth, candidate seed) tuple is computed once and reused by all
model seeds. Candidate sets are nested prefixes of one fixed order, so increasing K only adds
candidates. The exact reference (``model_seed = -1``) runs the same candidates through the
true transition and the same scoring entry, as a control that regret is zero when the model
is perfect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .candidates import CandidateConfig, candidate_sequences
from .dynamics import transition_exact
from .contract import GridMap, GridState, is_goal
from .data import Dataset
from .model import PointModel, decode, model_step
from .scoring import ScoreComponents, rollout, rollout_exact, score_trajectory

EXACT_SEED = -1


@dataclass(frozen=True)
class DiagnosticConfig:
    depths: Tuple[int, ...] = (1, 2, 4)
    sizes: Tuple[int, ...] = (16, 64, 256)
    candidate_seeds: Tuple[int, ...] = (0, 1, 2)
    model_seeds: Tuple[int, ...] = (0, 1, 2)
    eval_states: int = 20
    states_per_map: int = 10
    teacher_horizon: int = 4
    teacher_candidates: int = 16

    def as_dict(self) -> Dict[str, object]:
        return {
            "depths": list(self.depths), "sizes": list(self.sizes),
            "candidate_seeds": list(self.candidate_seeds), "model_seeds": list(self.model_seeds),
            "eval_states": self.eval_states, "states_per_map": self.states_per_map,
            "teacher_horizon": self.teacher_horizon, "teacher_candidates": self.teacher_candidates,
        }


def select_eval_states(
    dataset: Dataset, maps: Dict[str, GridMap], config: DiagnosticConfig
) -> List[Tuple[str, GridState]]:
    """Test-split legal non-terminal states by stable id, balanced across maps."""

    by_map: Dict[str, List[GridState]] = {name: [] for name in maps}
    for key, split in dataset.state_split.items():
        if split != "test":
            continue
        map_name, rest = key.split("|", 1)
        if map_name not in by_map:
            continue
        fields = dict(part.split("=") for part in rest.split("|"))
        state = GridState(
            tuple(int(v) for v in fields["r0"].split(",")),  # type: ignore[arg-type]
            tuple(int(v) for v in fields["r1"].split(",")),  # type: ignore[arg-type]
            tuple(int(v) for v in fields["o"].split(",")),  # type: ignore[arg-type]
        )
        if is_goal(state, maps[map_name]):
            continue
        by_map[map_name].append(state)
    selected: List[Tuple[str, GridState]] = []
    for name in maps:
        states = sorted(by_map[name], key=lambda state: state.key(name))
        selected.extend((name, state) for state in states[: config.states_per_map])
    if len(selected) < config.eval_states:
        seen = {(name, state.key(name)) for name, state in selected}
        for name in maps:
            for state in sorted(by_map[name], key=lambda state: state.key(name)):
                if (name, state.key(name)) in seen:
                    continue
                selected.append((name, state))
                seen.add((name, state.key(name)))
                if len(selected) >= config.eval_states:
                    break
            if len(selected) >= config.eval_states:
                break
    return selected[: config.eval_states]


def selection_metrics(candidate_ids: Sequence[int], j_hat: np.ndarray, j_ref: np.ndarray) -> Dict[str, float]:
    j_hat = np.asarray(j_hat, dtype=np.float64)
    j_ref = np.asarray(j_ref, dtype=np.float64)
    best = max(range(len(j_hat)), key=lambda index: (j_hat[index], -candidate_ids[index]))
    chosen = int(best)
    selected_bias = float(j_hat[chosen] - j_ref[chosen])
    candidate_mean_bias = float(np.mean(j_hat - j_ref))
    return {
        "chosen_index": chosen,
        "chosen_candidate_id": int(candidate_ids[chosen]),
        "selected_bias": selected_bias,
        "candidate_mean_bias": candidate_mean_bias,
        "selection_excess": selected_bias - candidate_mean_bias,
        "candidate_regret": float(np.max(j_ref) - j_ref[chosen]),
        "chosen_true_return": float(j_ref[chosen]),
        "uniform_candidate_return": float(np.mean(j_ref)),
        "chosen_model_return": float(j_hat[chosen]),
        "candidate_best_true_return": float(np.max(j_ref)),
    }


def _rollout_with_model(
    map_name: str, grid_map: GridMap, initial: GridState, sequence: Sequence[int],
    model: Optional[PointModel], exact: bool,
):
    if exact:
        result = rollout_exact(initial, sequence, grid_map)
        return score_trajectory(result), result
    assert model is not None
    result = rollout(initial, sequence, grid_map, model_step(model, grid_map))
    return score_trajectory(result), result


def run_diagnostic(
    dataset: Dataset,
    maps: Dict[str, GridMap],
    models: Dict[int, PointModel],
    *,
    config: Optional[DiagnosticConfig] = None,
    checkpoint_hashes: Optional[Dict[int, str]] = None,
) -> Dict[str, object]:
    config = config or DiagnosticConfig()
    started = time.time()
    eval_states = select_eval_states(dataset, maps, config)

    records: List[Dict[str, object]] = []
    chosen_traces: List[Dict[str, object]] = []
    per_run: List[Dict[str, object]] = []

    for map_name, state in eval_states:
        grid_map = maps[map_name]
        for depth in config.depths:
            for candidate_seed in config.candidate_seeds:
                sequences = candidate_sequences(
                    state, grid_map, depth, max(config.sizes), candidate_seed,
                    CandidateConfig(seed=candidate_seed, max_sequences=max(config.sizes)),
                )
                # truth once per candidate; reused by every model seed
                truth: List[ScoreComponents] = []
                for sequence in sequences:
                    score, _ = _rollout_with_model(map_name, grid_map, state, sequence, None, True)
                    truth.append(score)
                model_scores: Dict[int, List[ScoreComponents]] = {}
                trace_by_seed: Dict[int, List] = {}
                for model_seed in (EXACT_SEED, *config.model_seeds):
                    scores: List[ScoreComponents] = []
                    traces: List = []
                    for sequence in sequences:
                        score, result = _rollout_with_model(
                            map_name, grid_map, state, sequence,
                            None if model_seed == EXACT_SEED else models[model_seed],
                            model_seed == EXACT_SEED,
                        )
                        scores.append(score)
                        traces.append(result)
                    model_scores[model_seed] = scores
                    trace_by_seed[model_seed] = traces

                for size in config.sizes:
                    k = min(size, len(sequences))
                    if k <= 0:
                        continue
                    ids = list(range(k))
                    j_ref = np.asarray([truth[i].J for i in range(k)])
                    for model_seed, scores in model_scores.items():
                        j_hat = np.asarray([scores[i].J for i in range(k)])
                        metrics = selection_metrics(ids, j_hat, j_ref)
                        chosen = int(metrics["chosen_index"])
                        invalid_rate = float(np.mean([scores[i].invalid_prediction for i in range(k)]))
                        per_run.append({
                            "map": map_name,
                            "state_key": state.key(map_name),
                            "state": {"robot0": state.robot0, "robot1": state.robot1, "object": state.object_anchor},
                            "H": depth,
                            "K": size,
                            "actual_K": k,
                            "candidate_seed": candidate_seed,
                            "model_seed": model_seed,
                            "checkpoint_sha256": (checkpoint_hashes or {}).get(model_seed, "exact" if model_seed == EXACT_SEED else ""),
                            **metrics,
                            "chosen_true_success": truth[chosen].success,
                            "chosen_true_steps": truth[chosen].executed_steps,
                            "chosen_true_invalid": truth[chosen].invalid_prediction,
                            "candidate_invalid_rate": invalid_rate,
                            "candidate_success_rate": float(np.mean([truth[i].success for i in range(k)])),
                        })
                        if model_seed != EXACT_SEED:
                            chosen_traces.append({
                                "map": map_name,
                                "state_key": state.key(map_name),
                                "H": depth,
                                "K": size,
                                "candidate_seed": candidate_seed,
                                "model_seed": model_seed,
                                "chosen_candidate_id": metrics["chosen_candidate_id"],
                                "sequence": list(sequences[chosen]),
                                "true_states": [
                                    [list(s.robot0), list(s.robot1), list(s.object_anchor)]
                                    for s in trace_by_seed[EXACT_SEED][chosen].states
                                ],
                                "predicted_states": [
                                    [list(s.robot0), list(s.robot1), list(s.object_anchor)]
                                    for s in trace_by_seed[model_seed][chosen].states
                                ],
                                "true_J": truth[chosen].J,
                                "model_J": scores[chosen].J,
                            })

                for candidate_id, sequence in enumerate(sequences):
                    record = {
                        "map": map_name,
                        "state_key": state.key(map_name),
                        "H": depth,
                        "candidate_seed": candidate_seed,
                        "candidate_id": candidate_id,
                        "sequence": list(sequence),
                        "J_ref": truth[candidate_id].J,
                        "J_ref_components": truth[candidate_id].as_dict(),
                        "J_hat": {str(seed): model_scores[seed][candidate_id].J for seed in (EXACT_SEED, *config.model_seeds)},
                        "J_hat_components": {str(seed): model_scores[seed][candidate_id].as_dict() for seed in (EXACT_SEED, *config.model_seeds)},
                    }
                    records.append(record)

    teacher = teacher_forced_vs_recursive(dataset, maps, models, eval_states, config)
    return {
        "config": config.as_dict(),
        "eval_states": [
            {"map": name, "state_key": state.key(name),
             "state": {"robot0": state.robot0, "robot1": state.robot1, "object": state.object_anchor},
             "split": dataset.state_split[state.key(name)]}
            for name, state in eval_states
        ],
        "per_run": per_run,
        "candidate_records": records,
        "chosen_traces": chosen_traces,
        "teacher_forced_vs_recursive": teacher,
        "wall_clock": time.time() - started,
    }


def _position_error(predicted: GridState, actual: GridState) -> float:
    return float(
        abs(predicted.robot0[0] - actual.robot0[0]) + abs(predicted.robot0[1] - actual.robot0[1])
        + abs(predicted.robot1[0] - actual.robot1[0]) + abs(predicted.robot1[1] - actual.robot1[1])
        + abs(predicted.object_anchor[0] - actual.object_anchor[0])
        + abs(predicted.object_anchor[1] - actual.object_anchor[1])
    )


def teacher_forced_vs_recursive(
    dataset: Dataset,
    maps: Dict[str, GridMap],
    models: Dict[int, PointModel],
    eval_states: Sequence[Tuple[str, GridState]],
    config: DiagnosticConfig,
) -> Dict[str, object]:
    """Small fixed contrast: recursive vs teacher-forced error by step."""

    depth = config.teacher_horizon
    out: Dict[str, object] = {}
    for model_seed, model in models.items():
        recursive_state_error = [0.0] * depth
        recursive_position_error = [0.0] * depth
        teacher_state_error = [0.0] * depth
        teacher_position_error = [0.0] * depth
        counts = [0] * depth
        for map_name, state in eval_states:
            grid_map = maps[map_name]
            sequences = candidate_sequences(
                state, grid_map, depth, config.teacher_candidates, config.candidate_seeds[0],
                CandidateConfig(seed=config.candidate_seeds[0], max_sequences=config.teacher_candidates),
            )
            for sequence in sequences:
                true_state = state
                predicted_state = state
                for step, joint in enumerate(sequence):
                    true_next, _info = transition_exact(true_state, joint, grid_map)
                    # recursive: feed the model's own decoded state
                    decoded = decode(model, grid_map, predicted_state, joint)
                    if _state_legal(decoded, grid_map):
                        predicted_state = decoded
                    recursive_state_error[step] += float(decoded != true_next)
                    recursive_position_error[step] += _position_error(decoded, true_next)
                    # teacher-forced: decode from the true current state
                    teacher_decoded = decode(model, grid_map, true_state, joint)
                    teacher_state_error[step] += float(teacher_decoded != true_next)
                    teacher_position_error[step] += _position_error(teacher_decoded, true_next)
                    counts[step] += 1
                    true_state = true_next
        out[str(model_seed)] = {
            "recursive_state_error": [value / max(1, count) for value, count in zip(recursive_state_error, counts)],
            "recursive_position_error": [value / max(1, count) for value, count in zip(recursive_position_error, counts)],
            "teacher_state_error": [value / max(1, count) for value, count in zip(teacher_state_error, counts)],
            "teacher_position_error": [value / max(1, count) for value, count in zip(teacher_position_error, counts)],
            "counts": counts,
        }
    return out


def _state_legal(state: GridState, grid_map: GridMap) -> bool:
    from .contract import state_is_valid

    return state_is_valid(state, grid_map)
