"""Detached local training supervisor: periodic evidence and explicit fault stops.

No model/reward/controller tuning. The agent can review reports and restart into a
NEW directory after a concrete repair. This program never promises future LLM turns.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,signal,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from experiments.option_rl_v13.live import atomic_json,read_json,stop_reason
from experiments.option_rl_v13.monitor import write_dashboard


def fault(history,live,*,now=None):
    now=time.time() if now is None else now
    for row in history:
        if any(isinstance(v,(int,float)) and not math.isfinite(v) for v in row.values()):return 'nonfinite_history'
    if live and now-live.get('updated_unix',now)>300:return 'heartbeat_stale_over_300s'
    if len(history)>=3 and all(r.get('train_anomaly_rate',0)>=.25 for r in history[-3:]):return 'train_anomaly_rate_ge_25pct_three_iterations'
    evaluation=[r for r in history if r.get('success_rate') is not None]
    if len(evaluation)>=2 and all(r.get('eval_anomaly_rate',0)>=.10 for r in evaluation[-2:]):return 'validation_anomaly_rate_ge_10pct_twice'
    if len(history)>=3 and all(r.get('approx_kl',0)>.5 for r in history[-3:]):return 'ppo_kl_over_0_5_three_iterations'
    if (len(history)>=10 and len(evaluation)>=2 and all(r.get('entropy',1)<.1 for r in history[-10:])
        and all(r.get('eval_outside_goal_success',1)==0 and r.get('mean_progress',1)<=0 for r in evaluation[-2:])):
        return 'entropy_collapse_without_nontrivial_validation_success'
    return None


def snapshot(history,live,decision):
    evaluations=[r for r in history if r.get('success_rate') is not None]
    latest=history[-1] if history else {}
    return dict(timestamp=time.time(),decision=decision,iteration=latest.get('iteration',0),
        phase=live.get('phase','starting'),latest=latest,validation=evaluations[-1] if evaluations else None,
        best_validation=max(evaluations,key=lambda r:(r['success_rate'],r.get('mean_progress',-1))) if evaluations else None,
        evaluation_count=len(evaluations),rolling_train_anomaly_rate=sum(r.get('anomaly_episodes',0) for r in history[-10:])/max(1,sum(r.get('attempted_episodes',0) for r in history[-10:])))


def write_summary(out,snap,name='monitor_summary.md'):
    last=snap['latest'];val=snap['validation'];best=snap['best_validation']
    lines=['# RL 本地监测摘要','',f"决策：{snap['decision']}；最近完成训练轮次：{snap['iteration']}；阶段：{snap['phase']}。",
      '','本文件由本地监测进程每分钟更新；不是未来人工或模型复核的承诺。',
      '训练曲线每迭代更新；固定验证每 10 轮重测，异常验证根不从分母删除。',
      '',f"近 10 轮训练异常比例：{snap['rolling_train_anomaly_rate']:.3%}。"]
    if last:
        lines += [f"训练成功率 {last.get('explore_success')}；初始在目标外成功率 {last.get('train_outside_goal_success')}；progress {last.get('explore_progress')}。",
                  f"policy loss {last.get('policy_loss')}；value loss {last.get('value_loss')}；entropy {last.get('entropy')}。",
                  f"actor/critic trunk gradient {last.get('actor_trunk_grad_norm')} / {last.get('critic_trunk_grad_norm')}；KL {last.get('approx_kl')}。"]
    if val:
        lines += ['',f"最近完整验证：iteration {val['iteration']}，success {val['success_rate']}，progress {val['mean_progress']}，异常比例 {val.get('eval_anomaly_rate')}。",
                  f"初始在目标外的验证成功率：{val.get('eval_outside_goal_success')}。",
                  f"历史较好完整验证：iteration {best['iteration']}，success {best['success_rate']}，progress {best['mean_progress']}。"]
    lines += ['','未因早期验证成功为零而停训。低成功率本身不证明程序错误。',
              '停止规则：非有限数值、心跳超过 300 秒、连续三轮训练异常 ≥25%、连续两次验证异常 ≥10%、连续三轮 KL >0.5，或持续低熵且无非平凡验证成功。',
              '规则触发时写 STOP.json，训练器在回合边界停止；不自动删除结果或更改奖励。',
              'checkpoint.pt 每迭代保存；best.pt 保留较好验证权重。保留所有 rollouts、逐根验证与训练日志。']
    p=out/name;temp=p.with_suffix('.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(p)


def run(job,hours,episodes=8,seed=0,eval_roots=64,eval_every=10,entropy=.02,render_every=20,poll_seconds=15,summary_seconds=60,workers=8,resume_from=None):
    job=Path(job).resolve()
    if job.exists() and any(job.iterdir()):raise ValueError('preserve existing jobs; choose a fresh directory')
    job.mkdir(parents=True,exist_ok=True);out=job/'run13_001'
    command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'experiments/option_rl_v13/train.py'),
        '--out',str(out),'--iterations','1000000','--hours',str(hours),'--episodes',str(episodes),'--seed',str(seed),
        '--workers',str(workers),'--eval-roots',str(eval_roots),'--eval-every',str(eval_every),'--entropy',str(entropy),'--render-every',str(render_every)]
    if resume_from is not None:command+=['--resume-from',str(Path(resume_from).resolve())]
    plan=dict(command=command,hours=hours,episodes=episodes,workers=workers,seed=seed,eval_roots=eval_roots,eval_every=eval_every,
        poll_seconds=poll_seconds,summary_seconds=summary_seconds,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stop_rule='see supervise.fault; no stop solely for early zero success',
        recovery='preserve originals; no automatic code/reward edits; agent may diagnose and restart separately')
    atomic_json(job/'supervision_plan.json',plan);started=time.monotonic();last_report=0.;last_ckpt_check=0.;request_time=None;next_hour=1;sources_saved=False
    with (job/'training.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(job/'processes.json',dict(supervisor_pid=os.getpid(),trainer_pid=child.pid,run=str(out),started_unix=time.time()))
        try:
            while True:
                code=child.poll();history=read_json(out/'history.json',[]) or [];live=read_json(out/'live_status.json',{}) or {}
                reason=fault(history,live)
                manifest=read_json(out/'manifest.json',{}) or {}
                if manifest and not sources_saved:
                    for name in manifest.get('source_sha256',{}):
                        target=job/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(ROOT/name,target)
                    shutil.copy2(Path(__file__),job/'source_snapshot/supervise.py');sources_saved=True
                if manifest and time.monotonic()-last_ckpt_check>=60:
                    for p,digest in manifest.get('source_sha256',{}).items():
                        if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest:
                            reason='runtime_source_changed:'+p;break
                    path=out/'checkpoint.pt'
                    if path.exists() and not reason:
                        import torch
                        data=torch.load(path,map_location='cpu',weights_only=False)
                        if any(not torch.isfinite(t).all() for t in data['state_dict'].values() if t.is_floating_point()):reason='nonfinite_checkpoint'
                    last_ckpt_check=time.monotonic()
                if time.monotonic()-started>hours*3600+300:reason='supervisor_wall_budget_overrun'
                if reason and code is None and manifest and not stop_reason(out):
                    atomic_json(out/'STOP.json',dict(reason=reason,source='local_supervisor',time=time.time()));request_time=time.monotonic()
                if request_time and code is None and time.monotonic()-request_time>120:
                    os.killpg(child.pid,signal.SIGTERM)
                decision=('stopping: '+str(stop_reason(out))) if stop_reason(out) else 'continue: no clear fault detected'
                if code is not None:
                    result=read_json(out/'result.json',{}) or {}
                    decision=f"finished: {result.get('status','process_exit')} (code {code})"
                if manifest:
                    snap=snapshot(history,live,decision);atomic_json(out/'supervisor_status.json',snap)
                    write_dashboard(history,out/'dashboard.html',subtitle=f'{out} | {hours:g}h budget | dynamic legal continuous46-5hz')
                    if time.monotonic()-last_report>=summary_seconds or code is not None:
                        write_summary(out,snap)
                        with (job/'observations.jsonl').open('a') as f:f.write(json.dumps(snap,allow_nan=False)+'\n')
                        last_report=time.monotonic()
                    if time.monotonic()-started>=next_hour*3600:
                        atomic_json(job/f'hourly_{next_hour:02d}.json',snap)
                        write_summary(out,snap,name=f'hourly_{next_hour:02d}.md');next_hour+=1
                if code is not None:
                    atomic_json(job/'supervisor_result.json',dict(exit_code=code,decision=decision,seconds=time.monotonic()-started,run=str(out)));return code
                time.sleep(poll_seconds)
        except BaseException as error:
            atomic_json(job/'supervisor_error.json',dict(error=repr(error),trainer_pid=child.pid,time=time.time()))
            if child.poll() is None and out.exists():atomic_json(out/'STOP.json',dict(reason='supervisor_error',detail=repr(error)))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--hours',type=float,default=12.);p.add_argument('--episodes',type=int,default=8)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--eval-roots',type=int,default=64)
    p.add_argument('--eval-every',type=int,default=10);p.add_argument('--entropy',type=float,default=.02)
    p.add_argument('--render-every',type=int,default=20);p.add_argument('--poll-seconds',type=float,default=15)
    p.add_argument('--summary-seconds',type=float,default=60);p.add_argument('--workers',type=int,default=8);p.add_argument('--resume-from',type=Path);args=vars(p.parse_args());run(**args)
if __name__=='__main__':main()
