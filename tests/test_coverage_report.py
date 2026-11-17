import numpy as np
from experiments.planar_tasks.authoring_v2.coverage_report import overlap_matrices, serial


def test_coverage_counts_direction_and_jaccard_excludes_shared_failures():
    x=np.array([[1,1,0,0,0],[1,0,1,1,0],[0,0,0,0,0],[np.nan]*5])
    m=overlap_matrices(x)
    assert m['intersection'][0,1]==1
    assert m['jaccard'][0,1]==.25
    assert m['rescue'][0,1]==2 and m['rescue'][1,0]==1
    assert m['containment'][0,1]==.5
    assert m['containment'][1,0]==1/3
    assert np.isnan(m['jaccard'][2,2])
    assert np.isnan(m['containment'][2]).all()
    assert np.isnan(m['rescue'][3]).all()
    assert serial(m)['jaccard'][3][0] is None
