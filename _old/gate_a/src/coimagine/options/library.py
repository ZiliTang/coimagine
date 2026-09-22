from __future__ import annotations

from dataclasses import dataclass

from coimagine.core.types import OptionKind


@dataclass(frozen=True)
class OptionTemplate:
    kind: OptionKind
    description: str


UNIVERSAL_OPTION_LIBRARY = (
    OptionTemplate(OptionKind.REPOSITION, "Move to a currently observed free-space anchor."),
    OptionTemplate(OptionKind.APPROACH, "Move to an observation-grounded preparation/contact pose."),
    OptionTemplate(OptionKind.PUSH, "Apply an entity-relative push effect; contact may be prepared first."),
    OptionTemplate(OptionKind.HOLD, "Maintain a task-specific synchronized hold role."),
    OptionTemplate(OptionKind.OBSERVE, "Execute a task-specific sensing action."),
    OptionTemplate(OptionKind.IDLE, "Remain stationary for one option boundary."),
)
