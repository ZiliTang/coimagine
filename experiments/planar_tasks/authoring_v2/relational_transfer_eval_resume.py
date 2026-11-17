"""Resume frozen transfer evaluation with bounded, summary-only IPC.

No training, author calls, policy changes or new test cases. Full trajectories are
saved in the worker before returning a small summary; completed futures are dropped.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,multiprocessing as mp,shutil,subprocess,sys,time,traceback
from . import relational_transfer_study as R
from .relational_transfer_recovery import policy_failure,RECOVERY,ORIGINAL
S=R.S;T=R.T
OLD=R.EVAL
OUT=S.BASE/'rel_transfer_evaluation_20261117_02'


def durable_summary_worker(job):
    path=Path(job['artifact_path'])
    S.save(path/'worker_started.json',dict(time=time.time(),case_id=job['case_id'],condition=job['condition']))
    try:
        result=R.worker(job)
        S.pack(path/'result.json.gz',result,compresslevel=1)
        # Never send the replay/public history through the ProcessPool pipe.
        return dict(summary=result['summary'],fixture_sha256=result['fixture_sha256'],source_sha256=result['source_sha256'])
    except BaseException:
        S.save(path/'worker_error.json',dict(traceback=traceback.format_exc()));raise


def recover():
    for p in [R.AUTHOR,R.PARENT,RECOVERY,R.DATA,R.FIT]:S.Campaign(p).verify_frozen()
    original=S.read(OLD/'campaign.json');saved=[];missing=[]
    for req in original['runs']:
        p=OLD/'runs'/req['id']/'result.json.gz'
        if not p.exists():missing.append(req);continue
        result=S.unpack(p)
        assert result['fixture_sha256']==req['fixture_sha256']
        row=dict(req,**result['summary'],imported_from=str(p),outcome_class='cap_invalid_action' if policy_failure(result['summary']) else 'execution')
        saved.append(row)
        del result
    assert len(saved)+len(missing)==560 and len({r['id'] for r in saved+missing})==560
    S.save(R.HOST/'EVALUATION_RECOVERY_PROTOCOL.json',dict(original_state=original,saved=len(saved),missing=len(missing),no_retraining=True,no_new_cases=True,no_checkpoint_selection=True,workers=R.WORKERS,max_inflight=R.WORKERS,ipc='summary only; full replay persisted by worker',root_cause='original futures retained full rollout results; parent SIGKILL coincided with system OOM',accounting='original submitted reservations do not establish execution; 178 durable outcomes known at initial audit; other worker completion unknown',source_artifact=str(R.PARENT),model_artifact=str(R.FIT)))
    c=S.Campaign(OLD);saved_by_id={r['id']:r for r in saved}
    with c.locked():
        for i,row in enumerate(c.state['runs']):
            rid=row['id']
            if rid in saved_by_id:
                actual=saved_by_id[rid]
                if rid in c.state['reservations']:c.state['used']['execution_ticks']-=1800-actual['ticks']
                c.state['runs'][i]=actual
            else:row.update(status='interrupted_execution_unknown',result_available=False,reserved_ticks_upper_bound=1800)
        c.state['reservations']={};c.state['accounting_note']='submitted != executed; unreturned requests retain conservative reserved upper bounds, not measured ticks';c._save()
    S.save(R.HOST/'EVALUATION_ORIGINAL_INCOMPLETE.json',dict(seal=S.finish(c,'incomplete')))
    c=T.make(OUT,len(missing))
    for row in saved:
        p=c.path/'runs'/row['id'];shutil.copytree(OLD/'runs'/row['id'],p);S.save(p/'summary.json',row)
    with c.locked():c.state['runs']=saved;c.state['import_note']='previous durable outcomes retained; quotas count only missing-case executions';c._save()
    return c,missing


def collect(c,requests):
    cases={r['id']:r for k in ['roots','iid','transfer'] for r in S.read(R.PARENT/'panels'/f'{k}.json')['cases']}
    pending={};cursor=0;rows=[];errors=[];io=S.SaveTiming();started=time.monotonic()
    with ProcessPoolExecutor(max_workers=R.WORKERS,mp_context=mp.get_context('spawn')) as pool:
        while pending or cursor<len(requests):
            while not errors and len(pending)<R.WORKERS and cursor<len(requests):
                req=dict(requests[cursor],status='running');case=cases[req['case_id']];assert case['fixture_sha256']==req['fixture_sha256']
                job=dict(req,fixture=case['fixture'],artifact_path=str(c.path/'runs'/req['id']))
                with c.locked():
                    index=len(c.state['runs']);c._charge(attempts=1,resets=1,execution_ticks=1800);c.state['reservations'][req['id']]=dict(ticks=1800);c.state['runs'].append(req);c.state.update(phase='collecting',active_condition=req['condition']);c._save()
                io.call(S.save,c.path/'runs'/req['id']/'request.json',req)
                pending[pool.submit(durable_summary_worker,job)]=(req,index);cursor+=1
            if not pending:break
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                req,index=pending.pop(future)
                try:
                    result=future.result();assert result['fixture_sha256']==req['fixture_sha256']
                    expected=S.read(R.PARENT/'protocol.json')['source_hashes'][req['sid']]
                    assert R.sha(R.PARENT/'inputs'/f"{req['sid']}.py")==expected
                    assert result['source_sha256']==S.digest((R.PARENT/'inputs'/f"{req['sid']}.py").read_text())
                    row=dict(req,**result['summary'],outcome_class='cap_invalid_action' if policy_failure(result['summary']) else 'execution')
                    io.call(S.save,c.path/'runs'/req['id']/'summary.json',row)
                    with c.locked():
                        c.state['used']['execution_ticks']-=1800-row['ticks'];del c.state['reservations'][req['id']];c.state['runs'][index]=row;c._save()
                    rows.append(row)
                    if (row['error'] and not policy_failure(row)) or row['stop']=='wall_budget':errors.append(req['id'])
                except BaseException:
                    S.save(c.path/'runs'/req['id']/'collector_error.json',dict(traceback=traceback.format_exc()));errors.append(req['id'])
                if len(rows)%16==0:io.log('transfer resume',completed=len(rows),total=len(requests),success=sum(r['success'] for r in rows),elapsed_seconds=round(time.monotonic()-started,2))
            del done
    if errors:raise RuntimeError('evaluation infrastructure failures: '+str(errors))
    assert len(rows)==len(requests)


def run():
    c,requests=recover();collect(c,requests);rows=S.read(c.path/'campaign.json')['runs'];assert len(rows)==560
    result=R.summarize(rows);result.update(rows=rows,episodes=len(rows),ticks=sum(r['ticks'] for r in rows),restored_previous_results=sum('imported_from' in r for r in rows),evaluation_recovery_protocol=str(R.HOST/'EVALUATION_RECOVERY_PROTOCOL.json'))
    S.save(c.path/'comparison.json',result);S.finish(c,'complete')
    result['seals']={str(p):S.Campaign(p).verify_frozen() for p in [R.AUTHOR,R.PARENT,ORIGINAL,RECOVERY,R.DATA,R.FIT,OLD,OUT]}
    S.save(R.HOST/'RESULT.json',result);S.save(R.HOST/'FINISHED.json',dict(status='complete',episodes=len(rows),evaluation=str(OUT)));print('FINISHED',result['groups'],flush=True)


if __name__=='__main__':
    try:run()
    except BaseException as exc:S.save(R.HOST/'eval_resume_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
