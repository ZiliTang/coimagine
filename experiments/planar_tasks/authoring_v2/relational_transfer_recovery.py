"""Audited collector recovery; unchanged frozen CAP, panels and fitting protocol.

Known CAP invalid recommendations remain failures. Only missing artifacts rerun;
their first attempts retain full reserved cost. No evaluation outcome was read.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,multiprocessing as mp,shutil,subprocess,sys,time,traceback
from . import relational_transfer_study as R
S=R.S;T=R.T
ORIGINAL=R.COLLECT
RECOVERY=S.BASE/'rel_transfer_collection_20261117_02'


def policy_failure(row):
    e=row.get('error') or {}
    return e.get('type')=='ValueError' and e.get('message')=='CAP first recommendation missing/illegal'


def durable_worker(job):
    result=R.worker(job)
    # Preserve result even if the main collector fails while consuming futures.
    S.pack(Path(job['artifact_path'])/'result.json.gz',result,compresslevel=1)
    return result


def collect(c,jobs):
    """Process complete bounded batch; never discard finished futures on failure."""
    existing=S.read(c.path/'campaign.json')['runs'];offset=len(existing);rows=[];failures=[]
    io=S.SaveTiming();pending={}
    with ProcessPoolExecutor(max_workers=R.WORKERS,mp_context=mp.get_context('spawn')) as pool:
        for j,job in enumerate(jobs):
            rid=f'a{offset+j:04d}';req=dict(id=rid,**{k:v for k,v in job.items() if k!='fixture'},status='running')
            with c.locked():
                c._charge(attempts=1,resets=1,execution_ticks=1800);c.state['reservations'][rid]=dict(ticks=1800);c.state['runs'].append(req);c.state.update(phase='collecting',active_condition=job['condition']);c._save()
            io.call(S.save,c.path/'runs'/rid/'request.json',req)
            pending[pool.submit(durable_worker,dict(job,artifact_path=str(c.path/'runs'/rid)))]=(req,offset+j)
        for future in as_completed(pending):
            req,index=pending[future]
            try:
                result=future.result();assert result['fixture_sha256']==req['fixture_sha256']
                # Do not rewrite the worker's saved raw result/error. Dataset split
                # is supplied explicitly to demonstration_traces downstream.
                row=dict(req,**result['summary'])
                row['outcome_class']='cap_invalid_action' if policy_failure(row) else 'execution'
                io.call(S.save,c.path/'runs'/req['id']/'summary.json',row)
                with c.locked():
                    c.state['used']['execution_ticks']-=1800-row['ticks'];del c.state['reservations'][req['id']];c.state['runs'][index]=row;c._save()
                rows.append(row)
                if (row['error'] and not policy_failure(row)) or row['stop']=='wall_budget':failures.append(req['id'])
            except BaseException:
                S.save(c.path/'runs'/req['id']/'collector_error.json',dict(traceback=traceback.format_exc()));failures.append(req['id'])
            if len(rows)%16==0:io.log('transfer',completed=len(rows),total=len(jobs),success=sum(r['success'] for r in rows))
    if failures:raise RuntimeError('infrastructure failures; all available artifacts retained: '+str(failures))
    return sorted(rows,key=lambda r:r['id'])


def recover_collection():
    R.TransferCampaign(R.AUTHOR).verify_frozen();S.Campaign(R.PARENT).verify_frozen()
    original=S.read(ORIGINAL/'campaign.json');jobs=R.teacher_jobs();saved={}
    for r in original['runs']:
        if (ORIGINAL/'runs'/r['id']/'result.json.gz').exists():saved[r['case_id']]=r
    attempted={r['case_id'] for r in original['runs']};missing=[j for j in jobs if j['case_id'] not in saved]
    repeat=[j['case_id'] for j in missing if j['case_id'] in attempted]
    assert len(jobs)==52 and len(saved)==28 and len(missing)==24 and len(repeat)==15
    S.save(R.HOST/'RECOVERY_PROTOCOL.json',dict(cause='collector misclassified a CAP illegal recommendation as infrastructure failure and discarded pending results on exit',original_state=original,reuse=len(saved),missing_attempts_repeated=repeat,not_yet_attempted=len(missing)-len(repeat),additional_attempts=24,total_collection_attempts=67,original_unreturned_ticks_upper_bound=15*1800,unchanged=['source','panels','teacher assignments','fit budgets','success criterion'],known_failures_not_retried=True,no_test_results_seen=True))
    old=S.Campaign(ORIGINAL)
    with old.locked():
        for row in old.state['runs']:
            if row['id'] in old.state['reservations']:row.update(status='interrupted_result_unavailable',result_available=False,ticks_upper_bound=1800)
        old.state['reservations']={};old.state['accounting_note']='15 unreturned attempts charged full 1800-tick upper bounds; not refunded';old._save()
    S.save(R.HOST/'ORIGINAL_COLLECTION_INCOMPLETE.json',dict(seal=S.finish(old,'incomplete')))
    c=T.make(RECOVERY,len(missing));imported=[]
    for case_id,row in sorted(saved.items()):
        rid=f'a{len(imported):04d}';dest=c.path/'runs'/rid;shutil.copytree(ORIGINAL/'runs'/row['id'],dest)
        row=dict(row,id=rid,imported_from=str(ORIGINAL/'runs'/row['id']),outcome_class='cap_invalid_action' if policy_failure(row) else 'execution')
        S.save(dest/'summary.json',row);imported.append(row)
    with c.locked():c.state['runs']=imported;c.state['import_note']='28 original artifacts reused; new budget counts only 24 new executions';c._save()
    rows=imported+collect(c,missing);assert len(rows)==52
    R.COLLECT=RECOVERY;S.save(RECOVERY/'coverage.json',R.summarize(rows));R.admission_checks(rows);S.finish(c,'complete')


def run():
    started=time.monotonic();recover_collection()
    subprocess.run([sys.executable,'-m',__spec__.name,'extract'],check=True)
    subprocess.run([sys.executable,'-m','experiments.planar_tasks.authoring_v2.relational_transfer_trainability','--data',str(R.DATA),'--output',str(R.HOST/'trainability')],check=True)
    if not S.read(R.HOST/'trainability/RESULT.json')['accepted']:raise ValueError('training-only canary failed; no silent tuning')
    T.make(R.FIT,0,fits=4)
    for arm in R.I.ARMS:subprocess.run([sys.executable,'-m','experiments.planar_tasks.authoring_v2.relational_transfer_study','fit','--arm',arm],check=True)
    S.save(R.HOST/'FIT_FINISHED.json',dict(seal=S.finish(S.Campaign(R.FIT),'complete')));S.Campaign(R.FIT).verify_frozen()
    final=S.read(R.PARENT/'inputs/final.json');cases=sum([S.read(R.PARENT/'panels'/f'{k}.json')['cases'] for k in ['roots','iid','transfer']],[])
    conditions=[(arm,final['feature_sid']) for arm in R.I.ARMS]+[('CAP_'+sid,sid) for sid in dict.fromkeys(final['cap_baseline_sids']+[final['feature_sid']])]
    jobs=[dict(case_id=r['id'],base_id=r['base_id'],layout=r['layout'],relation=r['relation'],split=r['split'],fixture=r['fixture'],fixture_sha256=r['fixture_sha256'],condition=name,sid=sid) for name,sid in conditions for r in cases]
    assert len(jobs)==560
    c=T.make(R.EVAL,len(jobs));rows=collect(c,jobs);result=R.summarize(rows);result.update(rows=rows,episodes=len(rows),ticks=sum(r['ticks'] for r in rows),elapsed_seconds=time.monotonic()-started,recovery_protocol=str(R.HOST/'RECOVERY_PROTOCOL.json'))
    S.save(R.EVAL/'comparison.json',result);S.finish(c,'complete');result['seals']={str(p):S.Campaign(p).verify_frozen() for p in [R.AUTHOR,R.PARENT,ORIGINAL,RECOVERY,R.DATA,R.FIT,R.EVAL]}
    S.save(R.HOST/'RESULT.json',result);S.save(R.HOST/'FINISHED.json',dict(status='complete',episodes=len(rows)));print('FINISHED',result['groups'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','extract']);a=p.parse_args()
    try:
        if a.command=='run':run()
        else:R.COLLECT=RECOVERY;R.build_data()
    except BaseException as exc:S.save(R.HOST/('recovery_'+a.command+'_FAILURE.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise
