"""Public command units; shared by CAP validation and Option RL candidates."""
CONTROL_DT = 0.05
MIN_TICKS = 1
MAX_TICKS = 40


def duration_seconds(ticks):
    if type(ticks) is not int or not MIN_TICKS <= ticks <= MAX_TICKS:
        raise ValueError('ticks outside 1..40')
    return CONTROL_DT*ticks
