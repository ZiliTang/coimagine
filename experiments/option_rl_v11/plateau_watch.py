"""One bounded plateau review; reads fixed-panel results and can request clean STOP.

Independent of the trainer: does not modify models, rewards, controllers or source.
Thresholds are an operational stopping rule, not a statistical proof of convergence.
"""
from __future__ import annotations
import argparse,json,math,os,time
from pathlib import Path

VERSION='run11-two-window-plateau-gate-v1'

def read(path,default=None):
 try:return json.loads(Path(path).read_text())
 except (FileNotFoundError,json.JSONDecodeError):return default

def atomic(path,data):
 path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def panel_window(run,lo,hi):
 samples=[]
 for iteration in range(lo,hi+1):
  if iteration%10:continue
  d=read(Path(run)/'evaluations'/f'iteration_{iteration:06d}.json')
  if not d or not d.get('complete'):return None
  outside=[r for r in d['roots'] if r['initial_distance']>.2]
  if len(outside)!=61:raise ValueError('frozen outside-goal denominator changed')
  samples.append((d,outside))
 if not samples:raise ValueError('empty evaluation window')
 names=[r['scene_hash'] for r in samples[0][1]];counts={k:0 for k in names}
 for d,roots in samples:
  if {r['scene_hash'] for r in roots}!=set(names):raise ValueError('fixed validation panel changed')
  for r in roots:counts[r['scene_hash']]+=r['success']
 return dict(lo=lo,hi=hi,eval_count=len(samples),outside_success=sum(sum(r['success'] for r in rs)/61 for _,rs in samples)/len(samples),progress=sum(d['mean_progress'] for d,_ in samples)/len(samples),stable50=sum(v>=.5*len(samples) for v in counts.values()),stable80=sum(v>=.8*len(samples) for v in counts.values()))

def plateau(base,windows,success_margin=.01,progress_margin=.02):
 """Every subsequent window must lack meaningful gain in all four indicators."""
 def improved(w):
  return (w['outside_success']>base['outside_success']+success_margin or
          w['progress']>base['progress']+progress_margin or
          w['stable50']>base['stable50'] or w['stable80']>base['stable80'])
 if len(windows)!=2:raise ValueError('two complete windows required')
 if any(not math.isfinite(w[k]) for w in [base]+windows for k in ['outside_success','progress']):raise ValueError('nonfinite metrics')
 return not any(improved(w) for w in windows)

def run(run_dir,*,baseline_end=5000,window=500,success_margin=.01,progress_margin=.02,poll_seconds=30):
 run_dir=Path(run_dir).resolve();folder=run_dir.parent/'plateau_gate_6000';folder.mkdir(exist_ok=False)
 plan=dict(version=VERSION,baseline=[baseline_end-window+1,baseline_end],windows=[[baseline_end+1,baseline_end+window],[baseline_end+window+1,baseline_end+2*window]],success_margin=success_margin,progress_margin=progress_margin,run=str(run_dir),pid=os.getpid(),started_unix=time.time(),meaning='all two windows lack gains: outside success +1pp, progress +0.02, stable50/80 root counts; operational stop, no statistical convergence claim')
 atomic(folder/'plan.json',plan)
 baseline=panel_window(run_dir,*plan['baseline'])
 if baseline is None:raise ValueError('baseline must be complete before arming')
 while True:
  result=read(run_dir/'result.json');stop=read(run_dir/'STOP.json');history=read(run_dir/'history.json',[]) or []
  current=history[-1]['iteration'] if history else 0
  status=dict(updated_unix=time.time(),iteration=current,baseline=baseline,at_update=baseline_end+2*window)
  if result is not None or stop is not None:
   status.update(decision='run_already_finishing',result=result,stop=stop);atomic(folder/'result.json',status);return status
  if current>=baseline_end+2*window:
   windows=[panel_window(run_dir,*bounds) for bounds in plan['windows']]
   if all(w is not None for w in windows):
    should_stop=plateau(baseline,windows,success_margin,progress_margin)
    status.update(windows=windows,decision='request_stop_plateau' if should_stop else 'continue_measurable_gain')
    if should_stop:
     payload=dict(reason='agent_bounded_plateau_review_6000_no_gain_in_two_windows',source=VERSION,time=time.time(),baseline=baseline,windows=windows)
     temp=folder/'STOP_payload.json';atomic(temp,payload)
     try:os.link(temp,run_dir/'STOP.json')
     except FileExistsError:status['stop_already_present']=True
    atomic(folder/'result.json',status);atomic(folder/'status.json',status);return status
  status['decision']='waiting_for_two_complete_windows';atomic(folder/'status.json',status)
  time.sleep(poll_seconds)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);args=p.parse_args();print(json.dumps(run(args.run)),flush=True)
if __name__=='__main__':main()
