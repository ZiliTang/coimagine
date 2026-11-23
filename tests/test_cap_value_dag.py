import numpy as np

from tools import cap_value_dag as dag


def test_same_cluster_and_early_terminal_propagate_both_moments():
    root=(3,3,5,5,'small','A')
    after=(3,3,4,5,'small','AA')
    later=(3,3,3,5,'toward','AA')
    def case(states,distance,success):
        y=dag.terminal(dict(gap=distance,success=success))
        return dict(branches={'AA':dict(states=states,y=y)})
    early=case([root],0.,1)
    loop=case([root,after,later],2.,0)
    book={'physical_history':dag.backups([early,loop],'physical_history')}
    q=dag.forecasts(book,'physical_history','AA',loop['branches']['AA'])
    np.testing.assert_allclose(q['root'][:5],[1.,2.,.5,.5,.5])
    assert np.isclose(dag.variance(q['root']),1.)
    assert np.isclose(q['root'][6],0.)  # branches are individually deterministic
    # A root-time forecast cannot read the observed future or test label.
    np.testing.assert_allclose(q['root'],dag.forecasts(book,'physical_history','AA',early['branches']['AA'])['root'])
    assert q['observed_mid'][0]==2.
