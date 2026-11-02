import pytest
from experiments.option_rl_v11.plateau_watch import plateau
BASE=dict(outside_success=.08,progress=.15,stable50=2,stable80=1)

def test_requires_no_gain_in_both_complete_windows():
 assert plateau(BASE,[dict(BASE),dict(BASE)])
 for field,value in [('outside_success',.10),('progress',.18),('stable50',3),('stable80',2)]:
  for index in [0,1]:
   ws=[dict(BASE),dict(BASE)];ws[index][field]=value
   assert not plateau(BASE,ws)

def test_rejects_incomplete_or_nonfinite_evidence():
 with pytest.raises(ValueError):plateau(BASE,[dict(BASE)])
 with pytest.raises(ValueError):plateau(BASE,[dict(BASE),dict(BASE,progress=float('nan'))])
