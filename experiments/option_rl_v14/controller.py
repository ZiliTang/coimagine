"""Reuse the frozen v13 action menu and execution controllers without edits."""
from experiments.option_rl_v13.controller import ACTIONS, CONFIG, ROTATION_CONFIG, VERSION, mask


def make_env(instance, *, context='extended'):
    from experiments.option_rl_v13.scenes import geometry_audit
    from experiments.option_rl_v14.mode_env import LearningEnv
    if not geometry_audit(instance)['legal']:
        raise ValueError('illegal initial geometry')
    return LearningEnv(instance, context=context)
