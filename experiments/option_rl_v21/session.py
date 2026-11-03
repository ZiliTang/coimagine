"""Testbed V2 session; the archived V1 remains independently reproducible."""
from experiments.option_rl_v19.session import TestbedSession as OldSession
from experiments.option_rl_v21.system import OptionSystemEnv
from experiments.option_rl_v21 import spec as S


class TestbedSession(OldSession):
    __test__ = False

    def __init__(self, instance, *, overrides=None, controller_version='option-v2'):
        super().__init__(instance, overrides=overrides, controller_version='option-v2',
                         action_spec=S, env_factory=OptionSystemEnv, install_controller=False)
