"""Local entropy derivative on legal replay states, not the full PPO gradient."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.option_rl_v12.train import load_checkpoint
from experiments.option_rl_v12.interface import observe
from experiments.option_rl_v12.mode_env import ContinuousModeEnv
from experiments.option_rl_v13.render_episodes import instance_from_dict
from experiments.option_rl_v13.diagnose_recovery import SOURCE

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise ValueError('preserve existing diagnostic')
    torch.set_num_threads(1);policy,stats,data=load_checkpoint(SOURCE/'policy.pt')
    ev=json.loads((SOURCE/'evaluations/iteration_000890.json').read_text());xs=[];m0=[];m1=[]
    for idx in (0,1,18,23):
        w=ContinuousModeEnv(instance_from_dict(data['manifest']['validation_instances'][idx]))
        for cmd in ev['roots'][idx]['actions'][:16]:
            obs,left,right=observe(w);encoded=stats.normalise(obs);encoded[760:]=obs[760:]
            xs.append(encoded);m0.append(left);m1.append(right);w.joint_step(cmd['a0'],cmd['a1'])
    x=torch.tensor(np.stack(xs));masks=[np.stack(m0),np.stack(m1)];results=[]
    original=float(policy.persistence_bias.detach())
    for bias in (2.5,original,0.):
        with torch.no_grad():policy.persistence_bias.fill_(bias)
        ds,_=policy.distributions(x,masks)
        entropy=sum(d.entropy().mean() for d in ds)
        derivative=float(torch.autograd.grad(entropy,policy.persistence_bias)[0])
        results.append(dict(bias=bias,entropy=float(entropy.detach()),d_entropy_d_bias=derivative,
                            d_entropy_loss_d_bias=-.02*derivative))
    result=dict(states=len(xs),weights='run12 final895',states_source='first16 recorded890 commands on each of validation roots0,1,18,23',
        interpretation='positive entropy-loss derivative contributes a decrease of persistence bias under gradient descent; actor gradient not measured here, so this is not causal attribution of total training drift',results=results)
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
