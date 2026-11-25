"""Run the matched, three-round graph-versus-random CAP authoring experiment."""
import argparse
import ast
from pathlib import Path
import time
import traceback

from tools import cap_authored_handoff as origin
from tools import cap_fault_curriculum_author as author
from tools import cap_fault_curriculum_prefix as prefix
from tools.cap_fault_curriculum_rooms import freeze

OUT = origin.OUT.parent / "cap_fault_curriculum_20261124_01"
read, save = origin.read, origin.save
INTERFACE = """Each dispatch calls the frozen mother for base_action, then the previously
accepted modules in creation order, then your new module. First non-None owns that
boundary. Your local dict initially {} persists only via module_update at actual
feedback boundaries. In module_update, memory is the mother's updated public
state, event contains observation/action/executed_ticks (initial dt can be None).
In module_act all inputs are read-only; return None to defer, otherwise return a
new legal complete joint command (target_id, a0, a1, ticks). Roots of all
mother geometry helpers are available. Keep local small, JSON serializable,
with explicit entry, progress/timeout and release. The executor can re-query
on single PUSH contact and every 0.20m progress; preview updates are on a copy.
The global 48 KiB source limit includes the mother and earlier modules. Tests run
from each room's original start; a suggested takeover is an example, not a gate.
Use development feedback, never rely on scenario IDs, ordinal or outcome."""
ARMS = ('random', 'graph')
NROUNDS = 3


def compose(mother, modules):
    if not modules:
        return mother
    code = mother
    for name in ('initial_memory','update_memory','prepare_context'):
        pattern='def '+name+'('
        assert code.count(pattern)==1
        code=code.replace(pattern,'def fc_mother_'+name+'(')
    extras=[]
    for i, snippet in enumerate(modules):
        author.check_response(dict(source=snippet,rationale=''))
        tree=ast.parse(snippet)
        for fn in tree.body:
            fn.name=f'fc{i}_{fn.name}'
        extras.append(ast.unparse(tree))
    initial='def initial_memory() -> dict:\n    m = fc_mother_initial_memory()\n    m["fc_states"] = ['+', '.join('{}' for _ in modules)+']\n    return m'
    update=['def update_memory(memory: dict, event: dict) -> dict:',
            '    slots = memory.pop("fc_states")','    m = fc_mother_update_memory(memory, event)']
    prep=['def prepare_context(memory: dict, observation: dict) -> dict:',
          '    slots = memory.pop("fc_states")','    ctx = fc_mother_prepare_context(memory, observation)',
          '    base = ctx["proposals"][0]','    act = None']
    for i in range(len(modules)):
        update.append(f'    slots[{i}] = fc{i}_module_update(slots[{i}], fc_copy(event), fc_copy(m))')
        prep.extend(['    if act is None:',f'        act = fc{i}_module_act(slots[{i}], fc_copy(memory), fc_copy(observation), dict(base))'])
    update.extend(['    m["fc_states"] = slots','    return m'])
    prep.extend(['    if act is not None:', '        ctx["proposals"][0] = act',
                 '        tid = act["target_id"]','        j = next((j for j,e in enumerate(observation["entities"]) if e["id"]==tid), -1)',
                 '        ctx["values"] = [j,act["a0"],act["a1"],act["ticks"]]','    return ctx'])
    full='\n\n'.join([code,'from copy import deepcopy as fc_copy',*extras,initial,'\n'.join(update),'\n'.join(prep)])+'\n'
    from experiments.planar_tasks.semantic_rl.typed_compiler import validate_source
    validate_source(full)
    return full


def count(rows):
    return sum(v['result']['success'] for v in rows.values())


def scan(source, fixtures, trace):
    from tools.cap_fault_curriculum_isolated import scan_panel
    return scan_panel(source,fixtures,workers=10,trace=trace)


def protocol():
    cfg=dict(version=1,arms=list(ARMS),rounds=NROUNDS,experts='common frozen s003 fixed72-local',
             rules=INTERFACE,development=160,held=200,failed_cases=8,recovery_cases=4,
             model=author.MODEL,revisions=len(author.REVISIONS),seconds_per_call=author.CALL_SECONDS,
             effort=author.EFFORT,submit_bytes=author.SOURCE_BYTES,selection_seed_base=662411,
             selection='packet successes then minimum ticks then earlier revision; full development net gain >0')
    if (OUT/'protocol.json').is_file():
        assert read(OUT/'protocol.json')==cfg
    else:
        save(OUT/'protocol.json',cfg)


def collect_mother():
    dev=freeze(OUT)['dev']
    raw=(origin.WORK/'basis.py').read_text()
    assert origin.base.digest(raw)==origin.A
    source=origin.perception.source_for_profile(raw)
    target=OUT/'mother'
    target.mkdir(exist_ok=True)
    (target/'policy.py').write_text(source)
    author.seal(target,source,dict(list(dev.items())[:4]))
    if not (target/'dev.json').exists():
        save(target/'dev.json',prefix.scan_panel(source,dev,workers=22))
    return source


def arm(name):
    dev=freeze(OUT)['dev']
    mother=(OUT/'mother/policy.py').read_text()
    rows=read(OUT/'mother/dev.json')
    modules=[]
    for i in range(1,NROUNDS+1):
        d=OUT/name/f'r{i}';d.mkdir(parents=True,exist_ok=True)
        marker=d/'selected.json'
        if marker.is_file():
            rec=read(marker)
            if rec['admitted']:
                modules.append((d/f'candidate{rec["revision"]}'/'module.py').read_text())
                rows=read(d/'winner_dev.json')
            continue
        packet,audit=prefix.select_packet(rows,name,seed=662411+i,failed=8,recovered=4)
        save(d/'packet.json',packet);save(d/'audit.json',audit)
        before=count(rows)
        members=dict(mother=mother,**{f'accepted_{j+1}':v for j,v in enumerate(modules)})
        cases={case['room']:dev[case['room']] for case in packet['cases']}
        baseline=count({k:rows[k] for k in cases})
        feedback=None
        ranking=[]
        for rev in author.REVISIONS:
            cd=d/f'candidate{rev}';cd.mkdir(exist_ok=True)
            try:
                snippet=author.propose(d,packet,INTERFACE,members,rev,feedback)
            except (ValueError,SyntaxError) as exc:
                if not (d/f'call_{rev}/response.json').exists():
                    raise
                feedback=dict(error=str(exc),revision=rev)
                save(cd/'feedback.json',feedback)
                continue
            (cd/'module.py').write_text(snippet)
            try:
                source=compose(mother,[*modules,snippet])
            except (ValueError,SyntaxError) as exc:
                feedback=dict(error=str(exc),previous_source=snippet)
                save(cd/'feedback.json',feedback)
                continue
            author.seal(cd,source,dev_subset(cases,dev))
            try:
                trial=scan(source,cases,False)
            except (ValueError,RuntimeError,KeyError,TypeError,IndexError,ArithmeticError) as exc:
                feedback=dict(error=str(exc)[:2000],previous_source=snippet,revision=rev)
                save(cd/'feedback.json',feedback)
                continue
            save(cd/'packet_runs.json',trial)
            feedback=dict(revision=rev,previous_source=snippet,success=count(trial),baseline=baseline,
                          cases=[dict(room=k,baseline=rows[k]['result']['success'],
                                      success=v['result']['success'],ticks=v['result']['ticks']) for k,v in trial.items()])
            save(cd/'feedback.json',feedback)
            ranking.append((count(trial),-sum(v['result']['ticks'] for v in trial.values()),-rev))
        admitted=False;rev=None;tested=None
        if ranking:
            rev=-max(ranking)[2];cd=d/f'candidate{rev}';snippet=(cd/'module.py').read_text()
            source=compose(mother,[*modules,snippet])
            author.seal(cd,source,dev_subset(cases,dev))
            try:
                trial=scan(source,dev,True)
            except (ValueError,RuntimeError,KeyError,TypeError,IndexError,ArithmeticError) as exc:
                save(d/'dev_error.json',dict(error=str(exc)))
            else:
                tested=count(trial)
                save(d/'winner_dev.json',trial)
                if tested>before:
                    admitted=True;modules.append(snippet);rows=trial
        (d/'policy.py').write_text(compose(mother,modules))
        save(marker,dict(round=i,admitted=admitted,revision=rev,dev_before=before,
                         candidate_dev=tested,dev_after=count(rows),valid_candidates=len(ranking)))
        print(name,read(marker),flush=True)
    save(OUT/name/'locked.json',dict(sha=[author.sha((OUT/name/f'r{i}/policy.py').read_text()) for i in range(1,NROUNDS+1)]))


def dev_subset(cases,dev):
    subset=dict(cases)
    for k in dev:
        if len(subset)>=4:
            break
        subset[k]=dev[k]
    return subset


def finish():
    from tools.cap_fault_curriculum_report import publish
    rooms=freeze(OUT)
    mother=(OUT/'mother/policy.py').read_text()
    author.seal(OUT/'mother',mother,dict(list(rooms['dev'].items())[:4]))
    known={author.sha(mother)}
    policies={}
    for a in ARMS:
        policies[a]=[]
        locked=read(OUT/a/'locked.json')['sha']
        for i in range(1,NROUNDS+1):
            d=OUT/a/f'r{i}';rec=read(d/'selected.json')
            source=(d/'policy.py').read_text()
            assert author.sha(source)==locked[i-1]
            if rec['admitted']:
                cases={c['room']:rooms['dev'][c['room']] for c in read(d/'packet.json')['cases']}
                author.seal(d/f'candidate{rec["revision"]}',source,dev_subset(cases,rooms['dev']))
                known.add(author.sha(source))
            assert author.sha(source) in known
            policies[a].append(source)
    def eval_one(source):
        path=OUT/'held'/f'{author.sha(source)}.json'
        if not path.exists():
            save(path,scan(source,rooms['held'],False))
        rows=read(path)
        assert rows.keys()==rooms['held'].keys()
        return rows
    baseline=eval_one(mother)
    histories={a:[eval_one(src) for src in sources] for a,sources in policies.items()}
    publish(OUT,baseline,histories)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('mother',*ARMS,'finish'))
    phase=p.parse_args().phase
    try:
        if phase=='mother':
            protocol();collect_mother()
        elif phase in ARMS:
            assert (OUT/'protocol.json').is_file()
            arm(phase)
        else:
            finish()
    except Exception:
        save(OUT/f'{phase}_error.json',dict(traceback=traceback.format_exc(),at=time.time()))
        raise
