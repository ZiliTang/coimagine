"""Matched decision forks with per-tick physical witnesses on the 16 development rooms."""
import argparse
from collections import Counter
import csv
from functools import partial
import hashlib
from itertools import zip_longest
import json
from math import dist
from statistics import median
import struct
from unittest.mock import patch

from experiments.planar_tasks.relational_search.contract import digest, validate_action
from experiments.planar_tasks.semantic_rl.phase_aware_cap import translating
from tools import cap_safe_nominal_study as study
from tools.cap_entry_residual_study import immutable

OUT = study.OUT / 'forks'
WINDOW = 120
MIN_ADVANCE = .02
BRAKE = dict(a0=44, a1=44, ticks=4, target_id=None)
COMPOSITIONS = {'base':'P绕行.v1','safe':'P保守推进.v1','brake':'P绕行.v1（一次 4 tick 双机制动）'}
PREFIX_FIELDS = ('tick','kind','history','observation','physical_prefix','running','left')


def protocol():
    fixtures = study.dev_rooms()
    immutable(OUT/'protocol.json', dict(rooms={k:study.content_hash(f) for k,f in fixtures.items()},
        selection="all 16 author-development rooms, fixed before the final source or paired outcomes",
        anchor="First unequal issued decision (including command duration) at a common ordinary boundary or actual combined in-flight review. No truth-based selection. Each source runs independently from reset.",
        equality=list(PREFIX_FIELDS), window_ticks=WINDOW, advance_threshold_m=MIN_ADVANCE,
        next_push="First outer target PUSH beginning strictly after the fork, or at/after the end of the 120 tick window for the later endpoint. Qualifies if an actually commanded translating pusher touched the task target and whole-command true goal distance decreased by at least 0.02m.",
        brake="At a matched fork, issue one legal 4 tick joint BRAKE and resume old P after that host command. Report terminal and unobservable cases.",
        scope="Development mechanism diagnostic. Actual choices and contacts have no physical unreachability interpretation. Chance, branch search and policy-specific counterfactual optima are not estimated."))
    return fixtures


def same_context(first, second):
    return all(first[k] == second[k] for k in PREFIX_FIELDS)


class Recorder:
    def __init__(self, anchor):
        self.anchor = anchor
        self.braked = False
        self.decisions = []
        self.frames = []
        self.chain = hashlib.sha256()

    def frame(self, world, codes):
        env = world.env
        backend = env.env.backend
        target = env.primary_id
        touched = 0
        static = False
        for contact in backend.contacts():
            if contact['entity_id'] != target:
                continue
            if contact['pair'] == 'robot_object':
                touched |= 1 << contact['robot_id']
            if contact['pair'] == 'object_wall':
                static = True
        pusher = 0
        if codes is not None and env.active_target == target:
            for i in translating(dict(a0=codes[0], a1=codes[1])):
                pusher |= 1 << i
        body = backend.bodies[target]
        actual = codes or [None,None]
        self.frames.append([*body.position,touched,static,pusher,*actual,env.active_target])
        self.target = target
        # Include every physical control/binding and every moving body's pose and velocity.
        self.chain.update(json.dumps([env.active_target,codes]).encode())
        for obj in [*backend.robot_bodies,*[b for _,b in sorted(backend.bodies.items())]]:
            self.chain.update(struct.pack('!6d',*obj.position,obj.angle,*obj.velocity,obj.angular_velocity))
        if env.total_periods + 1 != len(self.frames):
            raise ValueError('missing physical tick')

    def decide(self, world, kind, obs, suggestion, running=None, left=None):
        row = dict(tick=obs['tick'],kind=kind,history=world.history.sha256,observation=digest(obs),
                   physical_prefix=self.chain.hexdigest(),running=running,left=left,choice=suggestion,
                   seen=any(e['id']==obs['goal']['target_id'] and e['points'] and
                            e['last_seen_tick']==obs['tick'] for e in obs['entities']))
        if self.anchor and not self.braked and same_context(row,self.anchor):
            suggestion = validate_action(BRAKE,obs)
            row['choice'] = suggestion
            self.braked = True
        self.decisions.append(row)
        return suggestion

    def tick(self, original, world, action, *args, **kwargs):
        if not self.frames:
            self.frame(world,None)
        action = self.decide(world,'boundary',study.S.observe(world),action)
        step = world.step
        joint_step = world.env.joint_step
        def sample(a0,a1,*a,**kw):
            result = joint_step(a0,a1,*a,**kw)
            self.frame(world,[a0,a1])
            return result

        def observe_review(request, remaining_ticks=None, *, review=None):
            if review is None:
                return step(request,remaining_ticks)
            public = world.env.public
            def reviewed(feedback,left):
                # Each real review obtains a full public packet only when its own
                # public trigger fires. Observe the same call without adding reads.
                observed = []
                def record(*a, **kw):
                    obs = public(*a,**kw)
                    observed.append(obs)
                    return obs
                with patch.object(world.env,'public',record):
                    next_action = review(feedback,left)
                if observed:
                    next_action = self.decide(world,'push_review',observed[0],next_action,request,left)
                return next_action
            return step(request,remaining_ticks,review=reviewed)

        with patch.object(world.env,'joint_step',sample), patch.object(world,'step',observe_review):
            return original(world,action,*args,**kwargs)


def replay(job):
    room,fixture,source,anchor = job
    recorder = Recorder(anchor)
    engine = study.S
    with patch.object(engine,'tick_action',partial(recorder.tick,engine.tick_action)):
        _, row = study.host.episode((room,fixture,True,None,'boundary',False,study.host.enabled(source),None))
    if anchor and not recorder.braked:
        raise ValueError(f'{room}: brake anchor not reproduced')
    tags = {(e['tick'],e['kind']) for e in row['events'] if e['kind']!='boundary'}
    for event in recorder.decisions:
        if event['kind']=='push_review':
            event['triggers'] = sorted(reason for tick,reason in tags if tick==event['tick']) or ['single_push_review']
    ends = row['result']['ticks']
    if ends + 1 != len(recorder.frames):
        raise ValueError(f'{room}: final physical sample count')
    return room,dict(result=row['result'],frames=recorder.frames,decisions=recorder.decisions,
                     goal=[fixture['goal'][k] for k in ('x','y')], target=recorder.target)


def find_fork(a,b):
    for x,y in zip_longest(a['decisions'],b['decisions']):
        if x is None or y is None or not same_context(x,y):
            return 'unmatched_prefix', None
        if x['choice']!=y['choice']:
            return 'matched_fork', (x,y)
    if a['result'] != b['result'] or a['frames']!=b['frames']:
        return 'unmatched_outcome', None
    return 'identical_run', None


def effects(data, tick):
    frames = data['frames']
    terminal = len(frames)-1
    end = min(terminal,tick+WINDOW)
    result = data['result']
    goal = data['goal']
    def gap(t):
        return dist(frames[t][:2],goal)
    commands = [e for e in data['decisions'] if e['kind']=='boundary']
    effective = []
    for command,next_command in zip(commands,[*commands[1:],{'tick':terminal}]):
        start,stop = command['tick'],next_command['tick']
        if start<=tick or command['choice']['target_id']!=data['target'] or not translating(command['choice']):
            continue
        if gap(start)-gap(stop)<MIN_ADVANCE or not any(r[2] & r[4] for r in frames[start+1:stop+1]):
            continue
        effective.append(dict(start=start,end=stop,action=command['choice'],advance_m=gap(start)-gap(stop)))
    later = next((v for v in effective if v['start']>=end),None)
    following = next((c for c in commands if c['tick']>=end),None)
    first_end = next((c['tick'] for c in commands if c['tick']>tick),terminal)
    first_frame = frames[tick+1]
    first_different = next((t-1 for t in range(tick+2,first_end+1)
                            if frames[t][5:8]!=first_frame[5:8]), first_end)
    return dict(window_end=end, window_ticks=end-tick,goal_advance_m=gap(tick)-gap(end),
                first_executed=dict(a0=first_frame[5],a1=first_frame[6],bound_target=first_frame[7],
                    uninterrupted_ticks=first_different-tick),
                first_command_end=first_end,first_command_advance_m=gap(tick)-gap(first_end),
                object_translation_m=dist(frames[tick][:2],frames[end][:2]),
                pusher_target_contact_ticks=sum(bool(r[2]&r[4]) for r in frames[tick+1:end+1]),
                object_static_ticks=sum(int(r[3]) for r in frames[tick+1:end+1]),
                effective_push_within_window=any(v['end']<=end for v in effective),
                next_effective_push=next(iter(effective),None),first_command_after_window=following and
                dict(tick=following['tick'],action=following['choice']),
                next_effective_after_window=later,terminal_success=int(result['success']),
                terminal_tick=terminal,termination=result['termination'],
                followup_status='terminated_in_window' if terminal==end else
                    'effective_observed' if later else 'no_qualifying_later_command')


def summarize(results,fixtures):
    counts = Counter(v['status'] for v in results)
    matched = [v for v in results if v['status']=='matched_fork']
    overview = dict(panel='16 development rooms',n=len(fixtures),statuses=dict(counts),
                    matched_forks=len(matched),without_visible_target=sum(not v['base_anchor']['seen'] for v in matched),
                    reset_forks=sum(v['base_anchor']['tick']==0 for v in matched),
                    prefix_types=dict(Counter(v['base_anchor']['kind'] for v in matched)),
                    choice_change=dict(Counter(v['choice_change'] for v in matched)),
                    observed_control_difference=sum(v['first_actual_control_difference_at'] is not None for v in matched),
                    compared_whole_room_rescues=sum(v['base_success']<v['safe_success'] for v in results),
                    compared_whole_room_losses=sum(v['base_success']>v['safe_success'] for v in results))
    for arm in COMPOSITIONS:
        rows=[v[arm] for v in matched if arm in v]
        overview[arm]=dict(composition=COMPOSITIONS[arm],n=len(rows),
            terminal_success=sum(v['terminal_success'] for v in rows),
            positive_2cm_window=sum(v['goal_advance_m']>=MIN_ADVANCE for v in rows),
            median_window_goal_advance_m=median(v['goal_advance_m'] for v in rows) if rows else None,
            median_pusher_contact_ticks=median(v['pusher_target_contact_ticks'] for v in rows) if rows else None,
            with_static_contact_in_window=sum(v['object_static_ticks']>0 for v in rows),
            next_effective_push=sum(v['next_effective_push'] is not None for v in rows),
            next_effective_push_in_window=sum(v['effective_push_within_window'] for v in rows),
            after_window=dict(Counter(v['followup_status'] for v in rows)))
    def paired_change(field):
        signs=Counter()
        for v in matched:
            vals=[v[a][field] is not None for a in ['base','safe']]
            signs['gained' if vals==[False,True] else 'lost' if vals==[True,False] else 'same']+=1
        return dict(signs)
    overview['paired_next_effective_after_window']=paired_change('next_effective_after_window')
    overview['matched_terminal_pairs']=dict(Counter(f"{v['base_success']}->{v['safe_success']}" for v in matched))
    study.save(OUT/'paired.json',results)
    study.save(OUT/'summary.json',overview)
    with (OUT/'rooms.csv').open('w') as f:
        fields=['room','status','tick','kind','public_history','observation','physical_prefix',
                'arm','composition','success','advance_120_m','pusher_contact_ticks','object_static_ticks',
                'first_effective_tick','next_effective_after_window_tick','followup_status']
        writer=csv.DictWriter(f,fields)
        writer.writeheader()
        for row in results:
            anchor=row.get('base_anchor',{})
            for arm in COMPOSITIONS:
                if arm=='brake' and arm not in row:
                    continue
                v=row.get(arm,{})
                writer.writerow(dict(room=row['room'],status=row['status'],tick=anchor.get('tick'),kind=anchor.get('kind'),
                    public_history=anchor.get('history'),observation=anchor.get('observation'),physical_prefix=anchor.get('physical_prefix'),
                    arm=arm,composition=COMPOSITIONS[arm],success=v.get('terminal_success',row.get(arm+'_success')),
                    advance_120_m=v.get('goal_advance_m'),pusher_contact_ticks=v.get('pusher_target_contact_ticks'),
                    object_static_ticks=v.get('object_static_ticks'),first_effective_tick=(v.get('next_effective_push') or {}).get('start'),
                    next_effective_after_window_tick=(v.get('next_effective_after_window') or {}).get('start'),
                    followup_status=v.get('followup_status')))
    lines=['# 16 个开发房间的同前缀分叉','',
        f"两份冻结 P 分别从同一初态运行。总计 {len(results)} 间，共 {len(matched)} 间在相同公开历史、观测和逐 tick 物理前缀上首次分歧。"
        f"其中 {overview['reset_forks']} 间在初态分歧，{overview['without_visible_target']} 间在分歧点没有目标新回波。"
        f"从初态至终局完全一致 {counts['identical_run']} 间，前缀或终点无法核对 {counts['unmatched_prefix']+counts['unmatched_outcome']} 间。",'',
        f"分歧中改控制的有 {overview['choice_change'].get('changed_control',0)} 间，只改时长的有 {overview['choice_change'].get('duration_only',0)} 间。"
        f"在两个后缀都仍在运行期间，逐 tick 发出的动作编号或目标绑定实际不同的有 {overview['observed_control_difference']} 间。"
        f"16 间自由运行共救回 {overview['compared_whole_room_rescues']} 间，丢失 {overview['compared_whole_room_losses']} 间。",'',
        '| 真实执行的后缀 | 分母 | 120 ticks 内末帧净靠近 ≥2 cm | 净靠近中位数（m） | 窗内目标碰静态物 | 分歧后有下一次有效推 | 窗末后有下一次有效推 | 窗内终局 | 终局成功 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in COMPOSITIONS:
        v=overview[arm]
        md=v['median_window_goal_advance_m']
        lines.append(f"| {COMPOSITIONS[arm]} | {v['n']} | {v['positive_2cm_window']} | {round(md,3) if md is not None else '—'} | "
                     f"{v['with_static_contact_in_window']} | {v['next_effective_push']} | {v['after_window'].get('effective_observed',0)} | "
                     f"{v['after_window'].get('terminated_in_window',0)} | {v['terminal_success']} |")
    lines.extend(['','每个窗口从共同分歧点算起，遇到终局就取末帧。有效推动要求实际控制目标物体的平移推手至少一 tick 接触目标，且整条外层指令期间的真实目标距离净减少至少 2 cm。'
        '下一次要求在分歧后新下达指令，另列在窗口末端及以后才新下达的推动。末端前已开始的推不会重复计入后一项。'
        '第三组从同一前缀执行一次上限 4 tick 的双机制动后由旧 P 续跑。两份 P 的后缀差异包含其后历次决策。'
        '这里统计了三种实际续跑；其他可行走法、受力因果和物理不可达性仍未知。','',
        '逐房间导出见 [rooms.csv](rooms.csv)。[逐间分歧与物理效果](paired.json) 记录了首条动作、窗后实际接续的动作和首个达标推动。'
        '逐 tick 箱体坐标、控制和接触保存在 traces/，列顺序见 [manifest.json](manifest.json)。'])
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(overview,ensure_ascii=False),flush=True)


def execute(workers):
    fixtures=protocol()
    lock=study.read(study.OUT/'locked.json')
    final=study.read(study.PUBLIC/'FINAL.json')
    if lock['name']!=final['name'] or study.sha(study.PUBLIC/'FINAL.json')!=lock['final_sha256']:
        raise ValueError('author FINAL differs from lock')
    programs={}
    for arm,name,sha in [('base','P',study.BASE_SHA),('safe',lock['name'],lock['sha256'])]:
        path=study.OUT/'sealed'/name
        study.Campaign(path).verify_frozen()
        source=(path/'composed.py').read_text()
        if study.host.digest(source)!=sha or source!=(study.PUBLIC/(name+'.py')).read_text():
            raise ValueError(f'{arm}: source differs from freeze')
        programs[arm]=source
    immutable(OUT/'manifest.json',dict(arm_sha256={a:study.host.digest(p) for a,p in programs.items()},
        compositions=COMPOSITIONS,lock_sha256=study.sha(study.OUT/'locked.json'),
        protocol_sha256=study.sha(OUT/'protocol.json'),forks_sha256=study.sha(study.ROOT/'tools/cap_safe_nominal_forks.py'),
        runtime_sources=study.runtime_sources(),workers=workers,
        frame_columns=['true_target_x','true_target_y','target_touching_robot_bitmask','true_target_static_contact',
            'target_translation_pusher_bitmask','executed_a0','executed_a1','bound_target']))
    for arm,source in programs.items():
        jobs=[(k,f,source,None) for k,f in fixtures.items()]
        study.host.old.collect(source,jobs,replay,workers,OUT/'traces'/arm)
    results=[];brakes=[]
    for room,f in fixtures.items():
        a,b=[study.read(OUT/'traces'/arm/(room+'.json')) for arm in ('base','safe')]
        for name,episode in [('P',a),(lock['name'],b)]:
            reference=study.OUT/'runs/dev'/name/'full'/(room+'.json')
            if reference.exists() and episode['result']!=study.read(reference)['result']:
                raise ValueError(f'{room} {name}: instrumented episode differs from development run')
        status,pair=find_fork(a,b)
        row=dict(room=room,status=status,base_success=a['result']['success'],safe_success=b['result']['success'])
        results.append(row)
        if pair:
            x,y=pair
            row.update(base_anchor=x,safe_anchor=y,base=effects(a,x['tick']),safe=effects(b,y['tick']))
            old = x['choice'] or x['running']
            safe = y['choice'] or y['running']
            row['choice_change'] = ('duration_only' if study.host.old.prefix.event.same_control(old,safe)
                                    else 'changed_control')
            row['first_actual_control_difference_at'] = next((j-1 for j in range(
                x['tick']+1,min(len(a['frames']),len(b['frames'])))
                if a['frames'][j][5:8]!=b['frames'][j][5:8]),None)
            brakes.append((room,f,programs['base'],x))
    if brakes:
        study.host.old.collect(programs['base'],brakes,replay,min(workers,len(brakes)),OUT/'traces/brake')
    for row in results:
        if row['status']=='matched_fork':
            data=study.read(OUT/'traces/brake'/(row['room']+'.json'))
            t=row['base_anchor']['tick']
            event=next(e for e in data['decisions'] if same_context(e,row['base_anchor']))
            if event['choice']!=BRAKE:
                raise ValueError('standard brake was not issued')
            row['brake']=effects(data,t)
            physical = row['brake']['first_executed']
            if physical['a0']!=BRAKE['a0'] or physical['a1']!=BRAKE['a1'] or physical['uninterrupted_ticks']<1:
                raise ValueError('standard brake was not physically executed')
    summarize(results,fixtures)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['protocol','run'])
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.phase=='protocol':
        protocol()
    else:
        execute(args.workers)
