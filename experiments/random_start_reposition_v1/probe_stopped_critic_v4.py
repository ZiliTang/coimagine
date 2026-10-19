"""One real PPO update for the proposed B arm; old v3 environment held fixed."""
import concurrent.futures
import multiprocessing as mp
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/random_start_rl_diagnosis_v4"
sys.path.insert(0, str(OUT / "code/experiments/random_start_reposition_v1"))
import diagnose_v4 as d
d.SOURCE = ROOT / "outputs/random_start_time_budget_v3_coverage256"


def main():
    torch.set_num_threads(1)
    model = d.load_policy(d.SOURCE / "checkpoints/initial.pt")
    episodes = torch.load(OUT / "stochastic24.pt", weights_only=False)[:8]
    paths = torch.load(OUT / "replay_initial.pt", weights_only=False)
    before = d.path_scores(model, paths)
    original_weights = {k: v.clone() for k, v in model.state_dict().items()}
    hook = model.value_head.register_forward_pre_hook(lambda module, args: (args[0].detach(),))
    optimizer = torch.optim.AdamW(model.parameters(), lr=d.rl.LEARNING_RATE, weight_decay=0)
    metrics = d.rl.update(model, optimizer, episodes)
    hook.remove()
    actor_only = torch.load(OUT / "one_update_actor_only.pt", weights_only=True)
    actor_keys = [k for k in original_weights if not k.startswith("value_head.")]
    report = dict(world_version=d.rl.ENV_VERSION, training_batches=1, ppo_epochs=d.rl.UPDATE_EPOCHS,
                  note="value-head input detached; normal weighted value loss and global clipping retained",
                  update=metrics,
                  actor_max_abs_difference_from_actor_only=max(float((model.state_dict()[k]-actor_only[k]).abs().max())
                                                               for k in actor_keys),
                  value_head_max_change=max(float((model.state_dict()[k]-v).abs().max())
                                              for k,v in original_weights.items() if k.startswith("value_head.")),
                  path_changes=d.path_change(before, d.path_scores(model, paths)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,
            mp_context=mp.get_context("spawn"), initializer=d.runner.worker_init) as executor:
        report["closed_loop"] = d.summary(d.collect(executor, model, d.tracking()))
    torch.save(model.state_dict(), OUT / "one_update_stopped_value_grad.pt")
    d.save(OUT / "stopped_value_grad_probe.json", report)
    print(report["closed_loop"]["success"], report["actor_max_abs_difference_from_actor_only"])


if __name__ == "__main__":
    main()
