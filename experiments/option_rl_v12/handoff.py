"""After run11 exits cleanly, archive its result and launch one bounded run12 pilot.

The recipe and source fingerprints are frozen before waiting. This program does
not edit source, tune rewards, or generate/repair algorithms after launch.
"""
from pathlib import Path
import argparse,json,hashlib,os,subprocess,sys,time,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[2]

def read(p,default=None):
 try:return json.loads(Path(p).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def atomic(p,data):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)

def eligible(result,supervision):
 if not result or not supervision:return False
 return supervision.get('exit_code')==0 and (result.get('status') in ('wall_budget_reached','complete') or result.get('status')=='requested_stop' and result.get('stop_reason')=='agent_bounded_plateau_review_6000_no_gain_in_two_windows')

def readiness():
 smoke=ROOT/'outputs/option_rl_v12/smoke_006';bench=ROOT/'outputs/option_testbench_v1/continuous_modes_v1/round_008';pre=read(ROOT/'outputs/option_rl_v12/preflight_003/preflight.json')
 assert pre['status']=='complete' and pre['new_invalid']==0 and pre['physical_anomalies']==0
 assert read(smoke/'result.json')['status']=='complete'
 h=read(smoke/'history.json');assert len(h)==2 and sum(r['valid_episodes'] for r in h)==16
 assert all(r['train_anomaly_rate']==0 and r['eval_anomaly_rate']==0 and r['initial_logp_max_error']<2e-5 for r in h)
 metrics=read(bench/'metrics.json');assert len(metrics)==20 and all(not m['safety_triggers'] and m['max_penetration']<=.002 and m['max_reference_step']<=.025+1e-8 for m in metrics)
 assert all(r['same_physical_work_and_forces'] for r in read(bench/'cadence_equivalence.json'))
 suite=ET.parse(ROOT/'outputs/option_rl_v12/readiness_tests.xml').getroot().find('testsuite');assert int(suite.get('tests'))>=8 and suite.get('failures')=='0' and suite.get('errors')=='0'
 view=read(smoke/'viewer_validation.json');assert view and not view['script_errors'] and not view['mobile_overflow']
 sources=read(smoke/'manifest.json')['source_sha256'];sources['experiments/option_rl_v12/supervise.py']=hashlib.sha256((ROOT/'experiments/option_rl_v12/supervise.py').read_bytes()).hexdigest()
 for f,d in sources.items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==d,f
 return sources

def archive_old(old_job,folder):
 run=old_job/'run11_001';h=read(run/'history.json',[]);es=[r for r in h if r.get('success_rate') is not None];tail=es[-50:];best=max(es,key=lambda r:(r['success_rate'],r['mean_progress'])) if es else None
 summary=dict(result=read(run/'result.json'),iteration_count=len(h),last50_eval_mean_outside_success=sum(r['eval_outside_goal_success'] for r in tail)/len(tail) if tail else None,last50_eval_mean_progress=sum(r['mean_progress'] for r in tail)/len(tail) if tail else None,best=best,source_run=str(run),notes='fixed validation time-window means, not independent generalization samples; best includes initially-inside roots, see outside metric')
 atomic(folder/'run11_final_summary.json',summary)
 (folder/'run11_final_summary.md').write_text('# Run11 completed: automatic evidence summary\n\n'+json.dumps(summary,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-job',type=Path,required=True);p.add_argument('--folder',type=Path,required=True);p.add_argument('--new-job',type=Path,required=True);args=p.parse_args()
 args.folder.mkdir(parents=True,exist_ok=False);sources=readiness()
 command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'experiments/option_rl_v12/supervise.py'),'--job',str(args.new_job.resolve()),'--hours','2','--workers','8','--episodes','8','--eval-roots','64','--eval-every','10','--render-every','20']
 atomic(args.folder/'plan.json',dict(old_job=str(args.old_job),new_job=str(args.new_job),command=command,source_sha256=sources,pid=os.getpid(),budget_hours=2,reason='user authorized autonomous follow-up after current run; independent fixed-period continuous-mode pilot',handoff_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),readiness='9 focused tests; 1000 legal initializations; 48 first-tick checks; 20 replay branches; 16 smoke training episodes; browser checked'))
 while True:
  result=read(args.old_job/'run11_001/result.json');sup=read(args.old_job/'supervisor_result.json');live=read(args.old_job/'run11_001/live_status.json',{})
  status=dict(updated_unix=time.time(),old_phase=live.get('phase'),old_iteration=live.get('iteration'),decision='waiting_for_run11_clean_exit')
  if sup is not None:
   archive_old(args.old_job,args.folder)
   if not eligible(result,sup):status['decision']='held_for_review_old_run_not_clean'
   elif any(not (ROOT/f).exists() or hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=d for f,d in sources.items()):status['decision']='held_for_review_new_source_changed'
   elif args.new_job.exists():status['decision']='held_for_review_target_already_exists'
   else:
    env=os.environ.copy()
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[name]='1'
    with (args.folder/'new_supervisor_console.log').open('w') as log:child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    status.update(decision='launched_run12_two_hour_pilot',supervisor_pid=child.pid,new_job=str(args.new_job))
    atomic(ROOT/'outputs/option_rl_v12/latest_launch.json',dict(job=str(args.new_job.resolve()),supervisor_pid=child.pid,command=command,console=str(args.folder/'new_supervisor_console.log'),started_unix=time.time()))
   atomic(args.folder/'result.json',status);atomic(args.folder/'status.json',status);return
  atomic(args.folder/'status.json',status);time.sleep(30)
if __name__=='__main__':main()
