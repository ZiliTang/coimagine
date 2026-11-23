"""Closed-loop held-panel evaluation of both frozen fastest-teacher actors."""
import torch

from tools import cap_regional_fastest_actor as ex

study = ex.study
OUT = ex.OUT/'held'
PANELS = {'fresh':200, 'near':90}
CHECKPOINTS = {'polish':'polish.pt', 'fixed':'actor.pt'}


def run():
    fixtures = study.read(study.OUT/'held.json')
    frozen = study.read(study.OUT/'frozen.json')
    assert study.ext.old.digest(fixtures) == frozen['held']
    assert study.read(ex.OUT/'polish.json')['selected_extra'] == 4500
    hashes = {key:study.sha(ex.OUT/name) for key,name in CHECKPOINTS.items()}
    assert hashes['fixed'] == study.read(ex.DOCS/'metrics.json')['actor_sha256']
    OUT.mkdir(exist_ok=True)
    protocol = dict(seed=17, primary='polish', reference='fixed', sha256=hashes,
                    panels=PANELS, held_sha256=frozen['held'], sids=frozen['sids'])
    if (OUT/'protocol.json').exists():
        assert study.read(OUT/'protocol.json') == protocol
    else:
        study.save(OUT/'protocol.json',protocol)
    torch.set_num_threads(1)
    study.init_models(frozen['sids'])
    for key,name in CHECKPOINTS.items():
        study.dq.OUT = OUT/key
        study.dq.OUT.mkdir(exist_ok=True)
        model = ex.actor()
        model.load_state_dict(torch.load(ex.OUT/name,map_location='cpu',weights_only=True))
        study.dq.MODEL = model.eval()
        for panel,n in PANELS.items():
            jobs = [(cid,f,key,None,None,panel) for cid,f in sorted(fixtures.items()) if cid.startswith(panel+'_')]
            assert len(jobs) == n
            rows = study.dq.batch(jobs,16,'evaluate_'+panel)
            assert {r['case_id'] for r in rows} == {job[0] for job in jobs}
            print(key,panel,len(rows),sum(r['success'] for r in rows),flush=True)


if __name__ == '__main__':
    run()
