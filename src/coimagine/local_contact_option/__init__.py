"""Local-contact control and option-final-state validation (round: local_contact_option_v1).

The round asks whether local contact-surface feedback, with the robot's own global pose known
but no object-truth feedback to the controller, produces joint skill transitions with more
continuous contact, a clearer push duration and a smaller, less perturbation-sensitive final
state. See ``docs/local_contact_option_v1/``.
"""

from .contract import ARM_DESCRIPTIONS, Exit, OptionConfig, Stage
from .skill import LocalSurfacePush

__all__ = ["ARM_DESCRIPTIONS", "Exit", "OptionConfig", "Stage", "LocalSurfacePush"]
