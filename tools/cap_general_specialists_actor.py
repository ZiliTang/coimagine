"""Fastest-success action imitation on the frozen five-CAP library."""
import csv

import numpy as np
import torch

from tools import cap_general_specialists as campaign
from tools import cap_regional_directq as dq
from tools import cap_regional_fastest_actor as imitation
from tools.cap_regional_evaluate import paired

BASE = campaign.OUT/'directq'
OUT = campaign.OUT/'fastest_action_actor_s17'


def run():
    torch.set_num_threads(1)
    if (OUT/'FINISHED.json').is_file():
        return
    state = campaign.read(campaign.STATE)
    assert state['stage'] == 'frozen'
    assert campaign.base.host.RoomCampaign(campaign.CAM).verify_frozen() == state['seal']
    sids = [r['sid'] for r in state['rounds']]
    protocol = campaign.read(campaign.OUT/'protocol.json')
    dev = campaign.read(campaign.OUT/'development.json')
    extra = campaign.read(campaign.OUT/'extra.json')
    held = campaign.read(campaign.OUT/'held.json')
    assert campaign.base.digest(dev) == protocol['pool'] and campaign.base.digest(held) == protocol['held']
    assert not (dev.keys() & extra.keys() or held.keys() & (dev.keys() | extra.keys()))
    fixtures = dev | extra
    root_ids = [int(k.split('_')[1])*100+int(k.split('_')[2]) for k in fixtures]
    assert len(set(root_ids)) == len(fixtures)
    OUT.mkdir(exist_ok=True)
    settings = dict(sids=sids, development=protocol['pool'], extra=campaign.base.digest(extra),
                    held=protocol['held'], seal=state['seal'], seed=imitation.SEED, steps=imitation.STEPS,
                    batch=64, lr=.0003, sampling='uniform teacher room then boundary',
                    target='fastest successful trajectory by executed ticks, tie by CAP ID, merged action hard label',
                    normalizers=campaign.sha(BASE/'normalizers.npz'), library_inputs='GA without trace')
    if (OUT/'protocol.json').exists():
        assert campaign.read(OUT/'protocol.json') == settings
    campaign.save(OUT/'protocol.json', settings)
    if (OUT/'actor.pt').exists():
        model = imitation.actor(BASE)
        model.load_state_dict(torch.load(OUT/'actor.pt', weights_only=True))
    else:
        teachers, data = imitation.teacher_data(fixtures, sids, BASE)
        campaign.save(OUT/'teachers.json', teachers)
        model = imitation.fit(data, base=BASE, out=OUT)
        del data
    dq.MODEL = model.eval()
    dq.MODULES = [dq.load_native((campaign.CAM/'semantics'/(sid+'.py')).read_text())[0] for sid in sids]
    dq.OUT = OUT
    jobs = [(cid, f, 'actor', None, None, 'fresh') for cid, f in sorted(held.items())]
    results = dq.batch(jobs, 12, 'evaluate_fresh')
    by_room = {r['case_id']:r for r in results}
    with (campaign.OUT/'held_results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert set(by_room) == set(held) == {r['room'] for r in rows} and len(rows) == len(held)
    for row in rows:
        episode = by_room[row['room']]
        assert episode['termination'] in ('task_budget_exhausted', 'success') or (
            not episode['success'] and (episode['boundaries'] == 300 or episode['ticks'] == 1800))
        row['actor'] = int(episode['success'])
    with (OUT/'per_scene.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def values(key):
        return np.asarray([int(r[key]) for r in rows])
    reference = campaign.read(campaign.OUT/'FINISHED.json')
    best = reference['best_dev_sid']
    union = values('union').astype(bool)
    summary = dict(rooms=len(rows), teacher_rooms=len(campaign.read(OUT/'teachers.json')),
                   fit=campaign.read(OUT/'fit.json')[-1], checkpoint=campaign.sha(OUT/'actor.pt'),
                   vs_directq=paired(values('mix'),values('actor'),values('objects')),
                   vs_best=paired(values(best),values('actor'),values('objects')), best_sid=best,
                   vs_general=paired(values(sids[0]),values('actor'),values('objects')),
                   union=int(union.sum()), union_retention={
                       k:dict(inside=int(values(col)[union].sum()),outside=int(values(col)[~union].sum()))
                       for k,col in [('directq','mix'),('actor','actor')]})
    campaign.save(OUT/'FINISHED.json',summary)
    print(summary,flush=True)


if __name__ == '__main__':
    run()
