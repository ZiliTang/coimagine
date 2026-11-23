"""Complete the two single-change candidates on the remainder of the existing panel."""
from tools import cap_lidar_static_gate as study

ARMS = (('world','fixed'),('world','stagger'),('local','fixed'))


def evaluate(job):
    key,fixture = job
    rows = {c+'_'+s:study.episode(fixture,study.MODULES[c],s) for c,s in ARMS}
    assert rows['world_fixed']['success']==study.e.read(study.s.t.OUT/'confirmation'/(key+'.json'))['A']
    study.e.save(study.OUT/'remaining'/(key+'.json'),rows)


if __name__=='__main__':
    initial=study.initialize()
    first={k for panel,k,f in initial if panel=='natural'}
    natural=study.s.t.d.rooms(study.s.t.FRESH)
    jobs=[(k,f) for k,f in natural.items() if k not in first]
    assert len(jobs)==136 and len(first)==64
    study.e.save(study.OUT/'extension.json',dict(arms=ARMS,
        rooms={k:study.e.s.phase.input_hash(f) for k,f in jobs},
        reason='After first 64 natural outcomes: test fixed/local and stagger/world separately on remaining 136.'))
    study.e.pool(evaluate,jobs,16)
