from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

from coimagine.core.types import FactorId, FactorType, LocalOption, RobotId


@dataclass(frozen=True)
class Factor:
    factor_id: FactorId
    factor_type: FactorType
    options_by_robot: Mapping[RobotId, Tuple[LocalOption, ...]]

    @property
    def incident_robots(self) -> tuple[RobotId, ...]:
        return tuple(sorted(self.options_by_robot))


class FactorBuilder:
    def build(self, candidate_sets: Mapping[RobotId, Sequence[LocalOption]]) -> Mapping[FactorId, Factor]:
        buckets: Dict[FactorId, Dict[RobotId, list[LocalOption]]] = {}
        types: Dict[FactorId, FactorType] = {}

        for robot_id, options in candidate_sets.items():
            for option in options:
                for entity_id in option.signature.entity_ids:
                    factor_id = f"entity:{entity_id}"
                    buckets.setdefault(factor_id, {}).setdefault(robot_id, []).append(option)
                    types[factor_id] = FactorType.ENTITY
                for token in option.signature.resource_tokens:
                    factor_id = f"resource:{token}"
                    buckets.setdefault(factor_id, {}).setdefault(robot_id, []).append(option)
                    types[factor_id] = FactorType.RESOURCE
                for token in option.signature.temporal_tokens:
                    factor_id = f"sync:{token}"
                    buckets.setdefault(factor_id, {}).setdefault(robot_id, []).append(option)
                    types[factor_id] = FactorType.SYNCHRONIZATION

        return {
            factor_id: Factor(
                factor_id=factor_id,
                factor_type=types[factor_id],
                options_by_robot={
                    robot_id: tuple(sorted(options, key=lambda option: option.stable_key))
                    for robot_id, options in per_robot.items()
                },
            )
            for factor_id, per_robot in sorted(buckets.items())
        }
