"""Bounded, atomic replay of complete legal policy-observation sequences at 5 Hz."""
from pathlib import Path
import json,os
import numpy as np

def save_episode(folder,index,observations,actions,durations,rewards,*,scene_name,policy_iteration,decision_flags,context):
 # 31 is coprime with the eight-task panel: sparse replay does not permanently
 # select the same developer scene on every pass through the panel.
 if folder is None or index%31:return None
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/f'episode_{(index//31)%256:03d}.npz';tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 obs=np.asarray(observations,dtype=np.float32);a=np.asarray(actions,dtype=np.int16);dt=np.asarray(durations,dtype=np.float32);r=np.asarray(rewards,dtype=np.float32)
 if not (len(obs)==len(a)+1==len(dt)+1==len(r)+1):raise ValueError('one observation more than genuine decisions required')
 metadata=json.dumps(dict(version='continuous-held-replay-v14',context=context,actor_decision_flags='true only on genuine actor draws',scene_name=scene_name,sample_ordinal=index,policy_iteration=policy_iteration,observation_hz=5,semantics='legal observations only; bounded cache, every 31st episode occurrence, last 256 selected episodes',teacher_forcing=False))
 with tmp.open('wb') as f:np.savez_compressed(f,observations=obs,actions=a,dt=dt,rewards=r,actor_decision=np.asarray(decision_flags,dtype=bool),metadata=np.asarray(metadata))
 tmp.replace(path);return path
