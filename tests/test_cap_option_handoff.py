from tools import cap_option_handoff_report as report


def test_preparation_failure_and_unswitched_rooms_count_toward_the_whole_policy():
    def branch(root, mid, gain, win):
        return dict(root=root, mid=mid, edges=[{}, dict(gain=gain)], result=dict(success=win))
    before = dict(node=2,flags=[])
    reached = dict(node=5,flags=['touch'],distinct=True)
    stayed = dict(node=2,flags=[],distinct=True)
    arms = {k:branch(before,reached if k[0]=='A' else stayed,
                     .03 if k=='AB' else 0.,int(k=='AB')) for k in ['AA','AB','BA','BB']}
    common = {k:branch(None,None,0.,1) for k in arms}
    cases = [dict(key='fork',arms=arms,single_b=dict(success=0)),
             dict(key='early',arms=common,single_b=dict(success=1))]
    rows = report.rows_for(cases,'any','touch')
    s = report.summarize(rows)
    assert (s['formation']['n'],s['formation']['first'],s['formation']['second'])==(1,1,0)
    assert (s['local']['plus'],s['local']['n'])==(1,1)
    assert (s['terminal']['n'],s['terminal']['first'],s['terminal']['second'])==(2,2,1)
    assert (s['both_failed'],s['both_rescued'])==(1,1)
    assert s['vs_bprep']['plus']==1
    assert report.summarize(report.rows_for(cases,'node5','touch'))['active']==0
