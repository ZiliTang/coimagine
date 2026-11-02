"""Offline greedy-decoding diagnostic for v15 checkpoints; no training.

Reads the frozen checkpoint and the declared developer panel, reconstructs the
actor's joint type distribution at each scene's initial observation, and reports
the marginal / conditional / joint-MAP pairs.  It never feeds true state to the
policy: the only inputs are the same legal observation and masks used by the
trainer.  Scene names and action labels are used for reporting only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.option_rl_v15.interface import ACTIONS, make_env, observe
from experiments.option_rl_v15.learnability import load_panel
from experiments.option_rl_v15.train import load_checkpoint
from experiments.option_rl_v15.live import atomic_json

PUSH_PREFIXES = ('PUSH_N', 'PUSH_E', 'PUSH_S', 'PUSH_W', 'PUSH_NE', 'PUSH_NW', 'PUSH_SE', 'PUSH_SW')


def label(action):
    return ACTIONS[action].label


def push_direction(action):
    text = label(action)
    for prefix in PUSH_PREFIXES:
        if text.startswith(prefix + '['):
            return prefix
    return None


def scene_report(policy, normalizer, instance):
    env = make_env(instance, context='extended')
    obs, m0, m1 = observe(env)
    encoded = normalizer.normalise(obs)
    encoded[760:] = obs[760:]
    x = torch.tensor(encoded)[None, :]
    with torch.no_grad():
        (d0, d1), value = policy.distributions(x, (m0, m1))
    p0 = d0.probs[0]
    a0 = int(p0.argmax())
    pairs = []
    for first in range(len(ACTIONS)):
        if float(p0[first]) < 1e-7:
            continue
        with torch.no_grad():
            (_, conditional), _ = policy.distributions(x, (m0, m1), a0=torch.tensor([first]))
        row = conditional.probs[0]
        for second in range(len(ACTIONS)):
            probability = float(p0[first] * row[second])
            if probability > 1e-6:
                pairs.append(dict(p=probability, a0=first, a1=second,
                                  label0=label(first), label1=label(second)))
    pairs.sort(key=lambda row: row['p'], reverse=True)
    with torch.no_grad():
        (_, conditional), _ = policy.distributions(x, (m0, m1), a0=torch.tensor([a0]))
    conditional_top = [
        dict(p=float(probability), action=int(index), label=label(int(index)))
        for probability, index in zip(torch.topk(conditional.probs[0], 6).values.tolist(),
                                      torch.topk(conditional.probs[0], 6).indices.tolist())
    ]
    both_push = sum(row['p'] for row in pairs if push_direction(row['a0']) and push_direction(row['a1']))
    same_direction = sum(row['p'] for row in pairs
                         if push_direction(row['a0']) and push_direction(row['a0']) == push_direction(row['a1']))
    reposition_mass = sum(
        float(p0[index]) for index, action in enumerate(ACTIONS) if action.label.startswith('REPOSITION')
    )
    return dict(
        name=instance.name, value=float(value), greedy_a0=dict(action=a0, label=label(a0)),
        marginal_top=[dict(p=float(probability), action=int(index), label=label(int(index)))
                      for probability, index in zip(torch.topk(p0, 6).values.tolist(),
                                                    torch.topk(p0, 6).indices.tolist())],
        conditional_given_greedy=conditional_top,
        joint_map=pairs[:8],
        probability_both_push=both_push,
        probability_same_direction_push=same_direction,
        probability_a0_reposition=reposition_mass,
    )


def run(checkpoints, panel, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    train, validation = load_panel(panel)
    report = dict(scope='offline greedy decoding of frozen v15 checkpoints; no training; '
                        'same legal observation/masks as the trainer; true state not used as policy input',
                  panel=str(Path(panel).resolve()),
                  panel_sha256=hashlib.sha256(Path(panel).read_bytes()).hexdigest(),
                  validation_instances=[item.name for item in validation],
                  checkpoints={})
    for name, path in checkpoints.items():
        policy, normalizer, data = load_checkpoint(path)
        report['checkpoints'][name] = dict(
            path=str(Path(path).resolve()),
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            iteration=data['history'][-1]['iteration'],
            scenes=[scene_report(policy, normalizer, instance) for instance in train],
        )
    atomic_json(out / 'summary.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='append', required=True,
                        help='name=path; repeat for several checkpoints')
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    checkpoints = {}
    for item in args.checkpoint:
        name, _, path = item.partition('=')
        if not name or not path:
            raise SystemExit('checkpoint must be name=path: ' + item)
        checkpoints[name] = Path(path)
    run(checkpoints, args.panel, args.out)
