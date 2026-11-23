import numpy as np
from experiments.planar_tasks.authoring_v2.cap_trace_value_ppo import advantages


def test_gae_finite_horizon_success_failure_and_terminal_mask():
    for reward in (0.,1.):
        values=np.array([.2,.4,.7])
        terminal=reward-.7
        middle=.7-.4+.95*terminal
        first=.4-.2+.95*middle
        np.testing.assert_allclose(advantages([0.,0.,reward],values),[first,middle,terminal])
