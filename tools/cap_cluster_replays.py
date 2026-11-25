"""Render paired full-start executions for all changed target-cluster failures."""
import csv
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from math import dist
from pathlib import Path
from unittest.mock import patch

from tools import cap_cluster_module as exp
from tools.cap_cluster_module_report import DOC

prefix = exp.prefix


class Recorder(exp.BranchMember):
    def choose(self, row, obs, preview=False):
        evaluating_module = self.phase in ('once', 'full') and len(row['proposals']) == 3
        action = super().choose(row, obs, preview)
        # A pending replacement is the last accepted preview. A command with no
        # pending replacement calls propose() again before boundary().
        self.owner = '恢复模块' if evaluating_module and action is not None else '母程序'
        return action


def record(fixture, key, mode, anchor, expected):
    member = Recorder(exp.isolated.BROKER, mode, anchor)
    frames, events = [], []
    scene = {}
    original_world = prefix.s.phase.world

    @contextmanager
    def with_frames(f):
        with original_world(f) as (world, other):
            scene.update(world.replay()['scene'])
            backend = world.env.env.backend
            ids = [scene['primary_id'], *[o['label'] for o in scene['movable_objects']]]
            bodies = [backend.bodies[k] for k in ids] + list(backend.robot_bodies)
            def frame():
                assert len(frames) == world.env.total_periods
                frames.append([round(float(v), 5) for body in bodies
                               for v in (*body.position, body.angle)])
            update = world.detector.update
            def capture(packet):
                result = update(packet)
                frame()
                return result
            frame()
            world.detector.update = capture
            yield world, other

    def boundary(world, obs, action, pending):
        action = member.boundary(world, obs, action, pending)
        events.append(dict(t=obs['tick'], a=action, h=world.history.sha256, owner=member.owner,
                           contacts=[bool(r['contacting_object']) for r in obs['robots']],
                           segments=[], caps=[member.owner]))
        return action

    with patch.object(prefix.s.phase, 'world', with_frames):
        _, row = prefix.run_member((key, fixture, False), member, exp.preview, boundary)
    result = row['result']
    assert result == expected['result'] and len(frames) == result['ticks'] + 1
    assert member.changed == expected['changed']
    for i, ev in enumerate(events):
        ev['end'] = events[i+1]['t'] if i+1 < len(events) else result['ticks']
        assert ev['end'] > ev['t']
    return scene, dict(events=events, frames=frames, ticks=result['ticks'],
                       success=result['success'], reason=result['termination'] or 'decision_budget',
                       history=result['history'])


def compare(job):
    key, fixture, anchor, parent, full = job
    scene, left = record(fixture,key,'parent',anchor,parent)
    other_scene, right = record(fixture,key,'full',anchor,full)
    assert scene == other_scene
    # A preview can interrupt an otherwise identical running command early.
    # In that case the first differing physical time is its earlier end.
    for a,b in zip(left['events'],right['events']):
        assert a['t']==b['t'] and a['h']==b['h']
        if a['a']!=b['a']:
            split=a['t']
            break
        if a['end']!=b['end']:
            split=min(a['end'],b['end'])
            break
    else:
        raise ValueError('expected a divergence in '+key)
    assert left['frames'][:split+1] == right['frames'][:split+1]
    spans=[]
    for ev in right['events']:
        if ev['owner']=='恢复模块':
            if spans and spans[-1][1]==ev['t']:
                spans[-1][1]=ev['end']
            else:
                spans.append([ev['t'],ev['end']])
    assert spans
    summary=[dict(start=a,end=b,returned=b<right['ticks'],
                  displacement=round(dist(right['frames'][a][:2],right['frames'][b][:2]),3))
             for a,b in spans]
    return dict(cid=key,title=f"{key[-3:]} · "+('救回' if right['success'] else '接手后仍失败'),
                description='同一新房间从初态运行。左侧全程使用母程序；右侧由第 9 版模块自行决定何时接手，可在交还后再次介入。',
                labels=['母程序 P','P ＋恢复模块'],marker=spans[0][0],first_split=split,
                spans=summary,scene=scene,runs=[left,right])


def main():
    lock=exp.read(exp.OUT/'locked.json')
    campaign=exp.OUT/'trials'/lock['name']/'campaign'
    exp.author.host.RoomCampaign(campaign).verify_frozen()
    source=(campaign/'composed.py').read_text()
    assert exp.author.sha(source)==lock['sha256']
    fixtures=exp.read(exp.OUT/'fixtures.json')
    folder=campaign.parent/'confirmation'
    with (DOC/'rooms.csv').open() as stream:
        cases=[r for r in csv.DictReader(stream) if r['parent']=='0' and r['cluster']=='1' and r['history_changed']=='1']
    cases.sort(key=lambda r:(-int(r['module']),r['room']))
    jobs=[]
    for r in cases:
        key=r['room']
        original=exp.read(exp.OUT/'baseline'/(key+'.json'))
        parent=exp.read(folder/'parent'/(key+'.json'))
        assert parent['result']==original['result']
        jobs.append((key,fixtures[key],original['anchor'],parent,exp.read(folder/'full'/(key+'.json'))))
    with ProcessPoolExecutor(7,mp_context=mp.get_context('fork'),initializer=exp.isolated.initialize,initargs=(source,)) as pool:
        pairs=list(pool.map(compare,jobs))
    template=Path(__file__).with_name('cap_regional_advantage_replays.html').read_text()
    page=template.replace('首次分歧 · 同步物理回放','恢复模块 · 同步物理回放')
    page=page.replace('<section><header><label for="cases">','<p>冻结验证面板中，母程序有 21 间遇到目标故障后失败。其中 7 间在启用模块后执行历史改变，以下收录全部 7 间。交还是连续采用模块指令后，首次改为执行母程序指令的边界。</p><section><header><label for="cases">')
    page=page.replace('虚线标出干预，点击曲线区域可跳转。','浅金色区间为右侧执行模块指令的时段，场地图中的金色轨迹是该时段内物体中心的移动。点击曲线可跳转。')
    page=page.replace('/* EMBED_REPLAYS */',json.dumps(pairs,ensure_ascii=False,separators=(',',':')))
    (DOC/'replays.html').write_text(page)
    audit=[dict(room=p['cid'],divergence=p['first_split'],interventions=p['spans'],
                final=[dict(success=r['success'],tick=r['ticks'],history=r['history']) for r in p['runs']]) for p in pairs]
    exp.save(exp.OUT/'paired_replay_audit.json',audit)
    print([(p['cid'],p['spans']) for p in pairs])


if __name__=='__main__':
    main()
