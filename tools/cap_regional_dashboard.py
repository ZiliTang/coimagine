"""Read-only localhost dashboard for the three 6.1 Sol High authors."""
import ast
import csv
import hashlib
import json
import re
import time
from datetime import datetime
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/semantic_rl/cap_regional_five_20261119_01'
FINDINGS=ROOT/'docs/research/key_findings/2026-11-19_regional_cap_authoring'
ARMS=('regular','regional','single')
def read(p):
    return json.loads(p.read_text())


@lru_cache(maxsize=8192)
def history(path):
    return read(Path(path))


def general_coverage():
    root=OUT.parent/'cap_general_specialists_62_20261120_01'
    book=read(root/'state.json')
    original=sorted(history(str(root/'development.json')))
    panels=[dict(name='160 个原始开发场景',ids=original)]
    rounds=book['rounds']+([book['active']] if book['active'] else [])
    round_of={}
    for r in rounds:
        if r['n']>1:
            panels.append(dict(name=f"第 {r['n']} 轮困难场景组",ids=r['ids']))
        for trial in r['trials']:
            round_of.setdefault(trial['sid'],r['n'])
    activity=read(root/'activity.json') if (root/'activity.json').exists() else {}
    if activity.get('sid'):
        round_of.setdefault(activity['sid'],activity.get('round'))
    committed={r['sid']:r['n'] for r in book['rounds']}
    cases=list(dict.fromkeys(k for panel in panels for k in panel['ids']))
    # Include augmented cases when they are in a declared development family.
    # Frozen evaluation rooms never enter the coverage display.
    wanted=set(cases)
    candidates=[]
    for sub in read(root/'author/campaign.json')['submissions']:
        sid=sub['id']
        statuses={}
        for p in (root/'runs'/sid).glob('*.json'):
            if p.stem in wanted:
                row=history(str(p))
                statuses[p.stem]=int(bool(row['success'])) if row['complete'] and not row.get('error') else -1
        candidates.append(dict(sid=sid,name=sub['summary'],round=round_of.get(sid,committed.get(sid)),
                               committed=sid in committed,results=statuses))
    with (ROOT/'docs/research/key_findings/2026-11-19_cap_geometry/scenes.csv').open() as f:
        points={r['case_id']:[float(r['pc1']),float(r['pc2'])] for r in csv.DictReader(f) if r['case_id'] in original}
    return dict(panels=panels,candidates=candidates,points=points)


def original_pool_coverage(coverage):
    rooms = set(coverage['panels'][0]['ids'])
    union = set()
    full_union = set()
    best = None
    for candidate in coverage['candidates']:
        results = candidate['results']
        wins = {room for room in rooms if results.get(room) == 1}
        union.update(wins)
        if rooms and all(results.get(room) in (0, 1) for room in rooms):
            full_union.update(wins)
            if best is None or len(wins) > best['score']:
                best = dict(sid=candidate['sid'], score=len(wins))
    return dict(total=len(rooms), union=len(union), full_union=len(full_union), best=best)


@lru_cache(maxsize=1)
def frozen_pool_coverage(seal):
    return original_pool_coverage(general_coverage())


def general_downstream(root):
    path = root/'post_progress.json'
    if not path.exists():
        return None
    status = read(path)
    phase = status['stage']
    dq = root/'directq'
    if phase in ('开发轨迹补齐', '新房间单成员闭环'):
        sources = ['development.json', 'extra.json'] if phase == '开发轨迹补齐' else ['held.json']
        rooms = {k for name in sources for k in history(str(root/name))}
        done = sum(p.stem in rooms for p in (root/'runs'/status['sid']).glob('*.json'))
        status.update(label=f"{status['sid']} · 成员 {status['member']} / {status['total']} · 已返回房间", done=done, limit=len(rooms))
    elif phase in ('真实轨迹重放与动作校验', '新房间混合闭环'):
        replay = phase == '真实轨迹重放与动作校验'
        progress = read(dq/'progress.json') if (dq/'progress.json').exists() else {}
        expected = 'replay' if replay else 'evaluate_fresh'
        status.update(label='已校验轨迹' if replay else '已返回房间',
                      done=progress.get('completed',0) if progress.get('stage') == expected else 0,
                      limit=status['episodes'] if replay else status['rooms'])
    elif phase == '拟合 GA directQ':
        path = dq/'models/GA_s17/progress.json'
        fit = read(path) if path.exists() else {}
        q = fit.get('stage') == 'Q_init'
        status.update(label='Q 更新' if q else 'V 更新', done=fit.get('updates',0),
                      limit=status['q_updates'] if q else status['v_updates'])
    return status


def general_actor(root):
    directory = root/'fastest_action_actor_s17'
    if not (directory/'protocol.json').exists():
        return None
    if (directory/'FINISHED.json').exists():
        return dict(result=history(str(directory/'FINISHED.json')))
    if (directory/'actor.pt').exists():
        return dict(label='配对闭环房间', done=sum(1 for _ in (directory/'evaluation/fresh').glob('*.json')),
                    limit=len(history(str(root/'held.json'))))
    fit = read(directory/'fit.json')[-1] if (directory/'fit.json').exists() else {}
    return dict(label='最近一次拟合检查（更新数）', done=fit.get('step',0),
                limit=history(str(directory/'protocol.json'))['steps'], fit=fit)


def general_actor_scale(root):
    directory = root/'action_actor_800_s17'
    if not (directory/'stage.json').exists():
        return None
    stage = read(directory/'stage.json')
    if (directory/'FINISHED.json').exists():
        return dict(result=history(str(directory/'FINISHED.json'))['vs_330'])
    phase = stage['stage']
    if phase in ('自然房间基线采集','新增开发场景采集','新增轨迹已齐'):
        done = sum(1 for _ in (directory/'runs').glob('*/*.json'))
        return dict(label='新增开发执行 · '+stage.get('sid',''), done=done, limit=2350)
    if phase == '拟合 800 场景 actor':
        fit = read(directory/'fit.json')[-1] if (directory/'fit.json').exists() else {}
        return dict(label='800 场景 actor 更新', done=fit.get('step',0), limit=stage['updates'])
    progress = read(directory/'progress.json') if (directory/'progress.json').exists() else {}
    return dict(label=phase, done=progress.get('completed',0) if progress.get('stage')==phase else 0,
                limit=stage.get('total',0))


def general_set_actor(root):
    directory = root/'action_actor_800_sets_s17'
    if (directory/'FINISHED.json').is_file():
        return dict(result=history(str(directory/'FINISHED.json'))['vs_hard'])
    if (directory/'progress.json').is_file():
        p = read(directory/'progress.json')
        done = p.get('completed',0)
        if p['stage']=='拟合集合监督' and (directory/'fit.json').exists():
            done = read(directory/'fit.json')[-1]['step']
        return dict(label=p['stage'], done=done, limit=p.get('total',0))


def general_fit_diagnostic(root):
    directory=root/'action_actor_800_fit_diagnostic_s17'
    if not (directory/'progress.json').is_file():
        return None
    names={'init29':'只换初始化','order29':'只换抽样顺序','lr1e4':'学习率 0.0001',
           'layernorm':'加入层归一化','state_scale':'共享输入缩小 16 倍','continue':'旧权重追加硬标签更新'}
    curves=[]
    def row(name,step,correct,total,held=None):
        return dict(name=name,step=step,hits=correct,total=total,held=held)
    base=directory/'baseline.json'
    if base.exists():
        metrics=read(base)
        curves.append(row('原硬标签',24000,metrics['train']['correct'],metrics['train']['multi'],metrics['held']))
    for arm,name in names.items():
        path=directory/(arm+'_curve.json')
        if path.is_file():
            m=read(path)[-1]
            result=directory/(arm+'_metrics.json')
            step=m['step']+(24000 if arm=='continue' else 0)
            curves.append(row(name,step,m['informative_matched'],m['informative'],read(result)['held'] if result.exists() else None))
    closed=directory/'closed_loop'
    evaluation=read(closed/'progress.json') if (closed/'progress.json').is_file() else None
    if evaluation and (closed/'FINISHED.json').is_file():
        evaluation['paired']=read(closed/'FINISHED.json')['paired']['actor800']
    forks=closed/'forks'
    if evaluation and (forks/'progress.json').is_file():
        evaluation['forks']=read(forks/'progress.json')
        evaluation['forks']['later']=sum(1 for _ in (forks/'later').glob('*.json'))
        evaluation['forks']['teachers']=sum(1 for _ in (forks/'teacher_disagreements').glob('*.json'))
        availability=forks/'availability'/'progress.json'
        if availability.is_file():
            evaluation['availability']=read(availability)
            summary=availability.with_name('FINISHED.json')
            if summary.is_file():
                evaluation['availability']['first']=read(summary)['primary']['first']
            sweep=availability.parent/'candidate_sweep'/'progress.json'
            if sweep.is_file():
                evaluation['candidate_sweep']=read(sweep)
                if sweep.with_name('FINISHED.json').is_file():
                    evaluation['candidate_sweep']['failed']=read(sweep.with_name('FINISHED.json'))['failed']
    return dict(stage=read(directory/'progress.json')['stage'],curves=curves,evaluation=evaluation)


def actor_value_rollout(root):
    folder=root/'actor_value_rollout_20261121_s17'
    if not (folder/'progress.json').is_file():
        return None
    result=read(folder/'progress.json')
    if (folder/'FINISHED.json').is_file():
        data=read(folder/'FINISHED.json')
        result['success']=[data['actor_h1']['first'],data['actor_h1']['second'],data['actor_h4']['second']]
        result['rooms']=data['n']
    return result


def actor_inputs(root):
    folder=root/'actor_inputs_20261121_s17'
    if not (folder/'protocol.json').is_file():
        return None
    result={}
    for arm in read(folder/'protocol.json')['arms']:
        curve=folder/(arm+'_curve.json')
        metrics=folder/(arm+'_fit.json')
        result[arm]=dict(step=read(curve)[-1]['step'] if curve.exists() else 0,
                        fit=read(metrics) if metrics.exists() else None)
    progress=folder/'eval_progress.json'
    finished=folder/'FINISHED.json'
    return dict(arms=result,evaluation=read(progress) if progress.exists() else {},
                success={a:v['success'] for a,v in read(finished)['arms'].items()} if finished.exists() else {})


def actor_replan(root):
    folder=root/'actor_replan_20261121_s17'
    if not (folder/'progress.json').is_file():
        return None
    result=read(folder/'progress.json')
    if (folder/'FINISHED.json').is_file():
        data=read(folder/'FINISHED.json')
        result['success']={arm:r['success'] for arm,r in data['arms'].items()}
    matched=folder/'matched_start'
    if (matched/'progress.json').is_file():
        result['matched']=read(matched/'progress.json')
    if (matched/'FINISHED.json').is_file():
        result['matched']['pairs']={k:[v['frequency']['first'],v['frequency']['second']]
                                    for k,v in read(matched/'FINISHED.json').items()}
    comparison=folder/'norm_comparison'
    if (comparison/'progress.json').is_file():
        result['norm_comparison']=read(comparison/'progress.json')
    if (comparison/'FINISHED.json').is_file():
        data=read(comparison/'FINISHED.json')
        result['norm_comparison']['success']=[data['plain330']['first'],data['plain800']['first'],data['plain800']['second']]
    return result


def state():
    with (ROOT/'docs/research/key_findings/2026-11-19_cap_geometry/scenes.csv').open() as f:
        points={r['case_id']:[float(r['pc1']),float(r['pc2'])] for r in csv.DictReader(f) if re.fullmatch(r'dev_[2-5]_\d+',r['case_id'])}
    assert len(points)==160
    result=dict(updated=time.time(),points=points,arms={})
    focus=OUT.parent/'cap_focus_recovery_five_20261120_01'
    if (focus/'state.json').is_file():
        ledger=read(focus/'state.json')
        active=ledger.get('active') or {}
        result['focus_recovery']=dict(center=active.get('center'),
            rounds=[{k:r[k] for k in ('number','sid','center','wins')} for r in ledger['rounds']])
    for arm in ARMS:
        folder=OUT/arm
        s=read(folder/'study.json');c=read(folder/'campaign.json')
        sweeps=s['sweeps']
        latest=c['submissions'][-1] if c['submissions'] else {}
        completed=errors=ticks=0;partial={};last_key=None;last_sid=None
        for log in sorted((folder/'logs').glob('*')):
            if not log.is_dir():continue
            seen={}
            for p in log.glob('*.json'):
                row=history(str(p))
                completed+=1;ticks+=row['ticks']
                errors+=int(not row['complete'])
                seen[row['id']]=int(row['success']) if row['complete'] else -1
            if log.name.endswith('_'+latest.get('id','___')):
                partial.update(seen)
                last_key=log.name
                last_sid=latest.get('id')
        current=s['active']
        active_done=len(list((folder/'logs'/current['key']).glob('*.json'))) if current else 0
        best=max(sweeps,key=lambda v:len(v['wins']),default=None)
        wins=sorted(set().union(*(v['wins'] for v in ([best] if arm=='single' and best else sweeps))))
        bad=[]
        for f in sorted((folder/'author_tools').glob('*submit.json'))[-3:]:
            attempt=read(f)
            if attempt['status']=='error':bad.append(str(attempt.get('error',''))[:250])
        result['arms'][arm]=dict(latest=latest.get('id'),summary=latest.get('summary'),sha=latest.get('sha256'),
            subs=[dict(sid=v['id'],name=v['summary'],sha=v['sha256']) for v in c['submissions']],
            sweeps=sweeps,wins=wins,partial=partial,active=current,active_done=active_done,latest_activity=last_key,
            probes=s['probes'],probes_this_round=s['probes_this_round'],completed=completed,errors=errors,ticks=ticks,
            attempts=c['used']['attempts'],reserved_ticks=c['used']['execution_ticks']-ticks,
            features=c['used']['feature_batches'],deadline=c['deadline'],created=c['created'],end=(folder/'study.json').stat().st_mtime if s['final'] else None,final=s['final'],
            best=best['sid'] if best else None,submit_errors=bad)
    result['tokens']=read(OUT/'token_meter.json') if (OUT/'token_meter.json').exists() else {}
    if (OUT/'eval_progress.json').exists():
        ep=read(OUT/'eval_progress.json')
        ep['completed']=sum(1 for p in (OUT/'evaluation').glob('*/*.json'))
        ep['errors']=read(OUT/'evaluation_errors.json') if (OUT/'evaluation_errors.json').exists() else None
        ep['finished']=(OUT/'FINISHED.json').exists()
        if ep['finished']:
            ep['metrics']=history(str(FINDINGS/'summary.json'))['metrics']
            ep['report_ready']=(FINDINGS/'README.md').is_file()
        result['evaluation']=ep
    else:result['evaluation']=None
    dq=OUT/'directq'
    if (dq/'progress.json').is_file():
        progress=read(dq/'progress.json')
        progress['shards']=sum(1 for _ in (dq/'shards').glob('*/*.json'))
        fit=dq/'models/GA_s17/progress.json'
        progress['fit']=read(fit) if fit.is_file() else None
        panels={}
        for panel,total in [('dev',160),('fresh',200),('near',160)]:
            rows=[history(str(p)) for p in (dq/'evaluation'/panel).glob('*.json')]
            panels[panel]=dict(completed=len(rows),total=total,success=sum(r['success'] for r in rows))
        progress['panels']=panels
        progress['results']=history(str(dq/'FINISHED.json'))['metrics'] if (dq/'FINISHED.json').exists() else None
        result['directq']=progress
    ext=OUT.parent/'cap_regional_extension_20261119_01'
    if (ext/'protocol.json').is_file():
        live={}
        for key,file in [('budget','budget.json'),('process','pipeline.json'),('fit','directq/models/GA_s17/progress.json')]:
            if (ext/file).is_file():live[key]=read(ext/file)
        bfile=ext/'baseline.json'
        if bfile.is_file():
            baseline=read(bfile);covered=set().union(*baseline.values())
            live['neighbors_precovered']=sum(k.startswith('aug_') for k in covered)
        if (ext/'author_state.json').is_file():
            author=read(ext/'author_state.json');active=author['active']
            live['sweeps']=[{k:v for k,v in r.items() if k!='wins'} for r in author['sweeps']]
            live['probes']=author['probes']
            subs=read(ext/'author/campaign.json')['submissions']
            if subs:live['latest']={'sid':'n_'+subs[-1]['id'],'summary':subs[-1]['summary']}
            live['added_rooms']=sorted(set().union(*(r['wins'] for r in author['sweeps']))-covered)
            live['original_gaps']=read(ext/'protocol.json')['parents']
            if active:live['active']=dict(kind=active['kind'],sid=active['sid'],total=active['total'],done=sum(1 for _ in Path(active['directory']).glob('*.json')))
            live['original_union']=115+sum(r['added_original'] for r in author['sweeps'])
        live['replayed']=sum(1 for _ in (ext/'directq/shards').glob('*/*.json'))
        splits={'dev':160,'aug':55,'fresh':200,'near':90}
        live['q_panels']={}
        for k,total in splits.items():
            panel={'total':total}
            for name in ['directq','oldq']:
                files=(ext/name/'evaluation'/k).glob('*.json')
                rows=[history(str(p)) for p in files]
                panel[name]=dict(completed=len(rows),success=sum(r['success'] for r in rows))
            live['q_panels'][k]=panel
        cap_files=list((ext/'evaluation').glob('*/*.json'))
        live['expert_done']=sum(history(str(p))['complete'] for p in cap_files)
        live['closedloops']=live['expert_done']+sum(r[name]['completed'] for r in live['q_panels'].values() for name in ('directq','oldq'))
        live['fit_complete']=(ext/'directq/models/GA_s17/offline_complete.json').is_file()
        live['finished']=(ext/'FINISHED.json').is_file()
        if live['finished']:
            live['metrics']=history(str(FINDINGS.parent/'2026-11-19_regional_extension/results.json'))['metrics']
        result['extension']=live
    inverse=OUT.parent/'cap_regional_inverse_v_20261120_01'
    if (inverse/'stage.json').exists():
        model=inverse/'models/GA_s17'
        live=read(inverse/'stage.json')
        if (model/'progress.json').is_file():live['fit']=read(model/'progress.json')
        live['fitted']=(model/'offline_complete.json').is_file()
        live['completed']=sum(1 for _ in (inverse/'evaluation').glob('*/*.json'))
        result['inverse_v']=live
    fastest=OUT.parent/'cap_regional_fastest_actor_20261120_01'
    if (fastest/'fit.json').is_file():
        result['fastest_actor']=dict(fit=read(fastest/'fit.json')[-1],
            lookup=sum(1 for _ in (fastest/'lookup/evaluation').glob('*/*.json')),
            actor=sum(1 for _ in (fastest/'actor/evaluation').glob('*/*.json')),
            finished=(fastest/'FINISHED.json').is_file())
        if (fastest/'held/protocol.json').is_file():
            result['fastest_actor']['held']={
                arm:sum(1 for _ in (fastest/'held'/arm/'evaluation').glob('*/*.json'))
                for arm in ('polish','fixed')}
            result['fastest_actor']['held_finished']=(fastest/'held/FINISHED.json').is_file()
        if (fastest/'polish.json').is_file():
            extra=read(fastest/'polish.json')
            result['fastest_actor']['polish']=extra.get('series',[None])[-1]
            result['fastest_actor']['polish_best']=next((r for r in extra.get('series',[]) if r['extra']==extra.get('selected_extra')),None)
            result['fastest_actor']['polish_done']=sum(1 for _ in (fastest/'polish/evaluation').glob('*/*.json'))
    soft=OUT.parent/'cap_regional_soft_actor_20261120_01'
    if (soft/'fit.json').is_file():
        result['soft_actor']=dict(fit=read(soft/'fit.json')[-1],labels=read(soft/'label_stats.json'),
            completed=sum(1 for _ in soft.glob('*/evaluation/*/*.json')),finished=(soft/'FINISHED.json').is_file())
    identity=OUT.parent/'cap_regional_id_actor_20261120_01'
    if (identity/'fit.json').is_file():
        result['id_actor']=dict(fit=read(identity/'fit.json')[-1],
            completed=sum(1 for _ in identity.glob('*/evaluation/*/*.json')),
            finished=(identity/'FINISHED.json').is_file())
    ppo=OUT.parent/'cap_regional_id_ppo_20261120_01'
    if (ppo/'stage.json').is_file():
        rounds=read(ppo/'history.json') if (ppo/'history.json').is_file() else []
        monitor=[]
        for path in sorted((ppo/'monitor').glob('r*/results.json')):
            rows=history(str(path))
            monitor.append([int(path.parent.name[1:])*50,sum(r['success'] for r in rows)/len(rows)])
        backend=read(ppo/'cpp_backend.json') if (ppo/'cpp_backend.json').is_file() else None
        result['id_ppo']=dict(stage=read(ppo/'stage.json'),rounds=rounds,monitor=monitor,backend=backend,
            finished=(ppo/'FINISHED.json').is_file())
    once=OUT.parent/'cap_regional_once_20261120_01/progress.json'
    if once.is_file():
        result['single_override']=read(once)
    iteration=OUT.parent/'cap_regional_actor_iteration_20261120_01/progress.json'
    if iteration.is_file():
        result['actor_iteration']=read(iteration)
    replay=OUT.parent/'cap_regional_replay_20261120_01'
    if (replay/'progress.json').is_file():
        result['actor_replay']=read(replay/'progress.json')
        if (replay/'expert_suffix/progress.json').is_file():
            result['actor_replay']['suffix']=read(replay/'expert_suffix/progress.json')
    return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p=urlparse(self.path)
        if p.path=='/':
            data=Path(__file__).with_suffix('.html').read_bytes(); mime='text/html'
        elif p.path in ('/research-journal','/research-journal-data') or p.path.startswith('/journal-file/'):
            from tools.cap_research_journal import respond
            answer=respond(p.path)
            if answer is None:return self.send_error(404)
            data,mime=answer
        elif p.path=='/general62-coverage':
            data=Path(__file__).with_name('cap_general_specialists_coverage.html').read_bytes()
            mime='text/html'
        elif p.path=='/general62-coverage-data':
            data=json.dumps(general_coverage(),ensure_ascii=False).encode();mime='application/json'
        elif p.path=='/general62-journal':
            from tools.cap_general_specialists_journal import page
            data=page().encode();mime='text/html'
        elif p.path=='/general62-diff':
            from tools.cap_general_specialists_journal import diff
            args=parse_qs(p.query)
            try:data=diff(args.get('sid',[''])[0],args.get('parent',[''])[0]).encode()
            except KeyError:return self.send_error(404)
            mime='text/plain'
        elif p.path in ('/general62-data','/general62-source'):
            root=OUT.parent/'cap_general_specialists_62_20261120_01'
            if not (root/'state.json').exists():return self.send_error(404)
            book=read(root/'state.json')
            submissions=read(root/'author/campaign.json')['submissions']
            if p.path.endswith('-source'):
                sid=parse_qs(p.query).get('sid',['latest'])[0]
                if sid=='latest' and submissions:sid=submissions[-1]['id']
                if sid not in {r['id'] for r in submissions}:return self.send_error(404)
                data=(root/'author/semantics'/(sid+'.py')).read_bytes();mime='text/plain'
            else:
                from tools.cap_regional_token_meter import tail_usage
                active=book['active'] or {}
                if book['stage'] == 'frozen':
                    meter = dict(total_tokens=book['end_tokens'])
                else:
                    meter=tail_usage(Path(book['telemetry'])) if book.get('telemetry') else None
                fields=('sid','kind','score','total','eligible','finished')
                trials=[t for r in book['rounds'] for t in r['trials']]+active.get('trials',[])
                full=[t for t in trials if t['kind']=='sweep' and t['total']==160 and t['eligible']]
                names={r['id']:r['summary'] for r in submissions}
                def summary(t):
                    return dict(**{k:t[k] for k in fields},name=names.get(t['sid']))
                current=read(root/'activity.json') if (root/'activity.json').exists() else {}
                running=None
                if current.get('stage')=='running':
                    ids=set(current['ids'])
                    finished=[history(str(f)) for f in (root/'runs'/current['sid']).glob('*.json') if f.stem in ids]
                    running=dict(sid=current['sid'],done=sum(r['complete'] for r in finished),total=len(ids),
                        errors=sum(not r['complete'] for r in finished),success=sum(bool(r['success']) for r in finished if r['complete']))
                response=dict(stage=book['stage'],active=active.get('n'),topic=active.get('topic'),time_left=max(0,int(active.get('deadline',0)-time.time())),
                    tokens=(meter or {}).get('total_tokens',0),round_cap=active.get('token_end',0)-active.get('token_start',0),
                    start=book['start'],latest=submissions[-1]['id'] if submissions else None,running=running,
                    last_sweep=summary(full[-1]) if full else None,
                    candidate_coverage=(frozen_pool_coverage(str(book['seal'])) if book['stage']=='frozen'
                                        else original_pool_coverage(general_coverage())),
                    best_sweep=summary(max(full,key=lambda t:t['score'])) if full else None,
                    general_sweeps=[summary(t) for t in full],round_tokens=max(0,(meter or {}).get('total_tokens',0)-active.get('token_start',0)) if active else 0,
                    trials=[summary(t) for t in active.get('trials',[])],
                    members=[dict(sid=r['sid'],n=r['n'],topic=r['topic'],success=len(r['wins']),added=r['added'],union=r['union']) for r in book['rounds']],
                    downstream=general_downstream(root),
                    action_actor=general_actor(root),
                    actor_scale=general_actor_scale(root),
                    set_actor=general_set_actor(root),
                    fit_diagnostic=general_fit_diagnostic(root),
                    actor_value_rollout=actor_value_rollout(root),
                    actor_replan=actor_replan(root),
                    actor_inputs=actor_inputs(root),
                    result=read(root/'FINISHED.json') if (root/'FINISHED.json').exists() else None)
                data=json.dumps(response,ensure_ascii=False).encode();mime='application/json'
        elif p.path=='/focus-progress':
            file=OUT.parent/'cap_focus_recovery_five_20261120_01/live/index.html'
            if not file.is_file():return self.send_error(404)
            data=file.read_bytes();mime='text/html'
        elif p.path=='/census.csv':
            file=OUT.parent/'cap_focus_recovery_five_20261120_01/live/census.csv'
            if not file.is_file():return self.send_error(404)
            data=file.read_bytes();mime='text/csv'
        elif p.path=='/focus-source':
            author=OUT.parent/'cap_focus_recovery_five_20261120_01/author'
            sid=parse_qs(p.query).get('sid',['latest'])[0]
            if sid=='latest':
                entries=read(author/'campaign.json')['submissions']
                if not entries:return self.send_error(404)
                sid=entries[-1]['id']
            if not re.fullmatch(r's[0-9]{3,}',sid):return self.send_error(404)
            file=author/'semantics'/(sid+'.py')
            if not file.is_file():return self.send_error(404)
            data=file.read_bytes();mime='text/plain'
        elif p.path in ('/advantage-replays','/advantage-report','/advantage-chart.png'):
            filename, mime = {'/advantage-replays': ('replays.html','text/html'),
                              '/advantage-report': ('README.md','text/plain'),
                              '/advantage-chart.png': ('diagnostic.png','image/png')}[p.path]
            file = FINDINGS.parent/'2026-11-20_regional_advantage'/filename
            if not file.is_file():return self.send_error(404)
            data = file.read_bytes()
        elif p.path=='/data':
            data=json.dumps(state(),ensure_ascii=False).encode();mime='application/json'
        elif p.path=='/report' and (FINDINGS/'README.md').is_file():
            data=(FINDINGS/'README.md').read_bytes();mime='text/plain'
        elif p.path=='/extension-source':
            sid=parse_qs(p.query).get('sid',[''])[0]
            if not re.fullmatch(r'n_s[0-9]{3}',sid):return self.send_error(404)
            file=OUT.parent/'cap_regional_extension_20261119_01/author/semantics'/(sid[2:]+'.py')
            if not file.is_file():return self.send_error(404)
            data=file.read_bytes();mime='text/plain'
        elif p.path in ('/actor-replay-report','/actor-suffix-report'):
            file=FINDINGS.parent/'2026-11-20_regional_actor_replay'/('README.md' if p.path=='/actor-replay-report' else 'suffix.md')
            if not file.is_file():return self.send_error(404)
            data=file.read_bytes();mime='text/plain'
        elif p.path=='/actor-iteration-report':
            file=FINDINGS.parent/'2026-11-20_regional_actor_iteration/README.md'
            if not file.is_file():return self.send_error(404)
            data=file.read_bytes();mime='text/plain'
        elif p.path in ('/single-override-chart.png','/single-override-pairs.csv'):
            filename='progress.png' if p.path.endswith('png') else 'acceptance_pairs.csv'
            file=FINDINGS.parent/'2026-11-20_regional_once'/filename
            if not file.is_file():return self.send_error(404)
            data=file.read_bytes();mime='image/png' if filename.endswith('png') else 'text/csv'
        elif p.path=='/single-override-report' and (FINDINGS.parent/'2026-11-20_regional_once/README.md').is_file():
            data=(FINDINGS.parent/'2026-11-20_regional_once/README.md').read_bytes();mime='text/plain'
        elif p.path=='/id-ppo-report' and (FINDINGS.parent/'2026-11-20_regional_id_ppo/README.md').is_file():
            data=(FINDINGS.parent/'2026-11-20_regional_id_ppo/README.md').read_bytes();mime='text/plain'
        elif p.path=='/id-actor-report' and (FINDINGS.parent/'2026-11-20_regional_id_actor/README.md').is_file():
            data=(FINDINGS.parent/'2026-11-20_regional_id_actor/README.md').read_bytes();mime='text/plain'
        elif p.path=='/soft-actor-report' and (FINDINGS.parent/'2026-11-20_regional_soft_actor/README.md').is_file():
            data=(FINDINGS.parent/'2026-11-20_regional_soft_actor/README.md').read_bytes();mime='text/plain'
        elif p.path=='/fastest-held-report' and (FINDINGS.parent/'2026-11-20_regional_fastest_actor/held.md').is_file():
            data=(FINDINGS.parent/'2026-11-20_regional_fastest_actor/held.md').read_bytes();mime='text/plain'
        elif p.path=='/fastest-actor-report' and (FINDINGS.parent/'2026-11-20_regional_fastest_actor/README.md').is_file():
            data=(FINDINGS.parent/'2026-11-20_regional_fastest_actor/README.md').read_bytes();mime='text/plain'
        elif p.path=='/inverse-v-report' and (FINDINGS.parent/'2026-11-20_regional_inverse_v/README.md').is_file():
            data=(FINDINGS.parent/'2026-11-20_regional_inverse_v/README.md').read_bytes();mime='text/plain'
        elif p.path=='/extension-report' and (FINDINGS.parent/'2026-11-19_regional_extension/README.md').is_file():
            data=(FINDINGS.parent/'2026-11-19_regional_extension/README.md').read_bytes();mime='text/plain'
        elif p.path=='/directq-report' and (FINDINGS.parent/'2026-11-19_regional_directq/README.md').is_file():
            data=(FINDINGS.parent/'2026-11-19_regional_directq/README.md').read_bytes();mime='text/plain'
        elif p.path in ('/regional_member_maps.png','/regional_member_scenes.csv','/regional_member_counts.json',
                        '/regional_selection_geometry.png','/regional_selection_geometry.json'):
            file=FINDINGS/p.path[1:]
            if not file.is_file():return self.send_error(404)
            data=file.read_bytes()
            mime={'.png':'image/png','.csv':'text/csv','.json':'application/json'}[file.suffix]
        elif p.path=='/source':
            args=parse_qs(p.query);arm=args.get('arm',[''])[0];sid=args.get('sid',[''])[0]
            if arm not in ARMS or not re.fullmatch(r's\d{3,}',sid):return self.send_error(404)
            path=OUT/arm/'semantics'/(sid+'.py')
            if not path.exists():return self.send_error(404)
            data=path.read_bytes();mime='text/plain'
        else:return self.send_error(404)
        self.send_response(200);self.send_header('Cache-Control','no-store');self.send_header('Content-Type',mime+'; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass


if __name__=='__main__':
    ThreadingHTTPServer(('127.0.0.1',8775),Handler).serve_forever()
