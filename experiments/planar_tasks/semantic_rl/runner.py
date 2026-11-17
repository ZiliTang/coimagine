"""Bounded collection/training and inspectable artifacts; no implicit API calls."""
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import marshal
import time
import torch
from experiments.planar_tasks.action_semantics.program import Program
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.relational_search import pipeline_execution as X
from .environment import Environment, EventConfig
from .features import Features
from .actions import ActionTree
from .learning import demonstration_traces, warm_start, ppo_update
from . import VERSION
from .scale import protocol, CONTROL_DT


def rollout(fixture, source, **kwargs):
    from .typed_compiler import typed_source
    owned=None
    if kwargs.get('features') is None and typed_source(source):
        cfg=kwargs.get('config',EventConfig())
        owned=Features(source,fast_embedding=True,runtime='typed_batch',
                       action_protocol=protocol(cfg) if cfg.mode=='duration' else None)
        kwargs['features']=owned
    try:return _rollout(fixture,source,**kwargs)
    finally:
        if owned:owned.close()


def _rollout(fixture, source, *, model=None, features=None, config=EventConfig(),
            max_ticks=1800, max_decisions=600, wall_seconds=300, seed=17, deterministic=False, on_progress=None,
            temperature=1., fast_feedback=False, record_details=True, training_fastpath=False,
            prefix=()):
    if min(max_ticks, max_decisions, wall_seconds) <= 0:
        raise ValueError('positive rollout budgets required')
    prefix = X.validate_commands(list(prefix), allow_empty=True)
    expected = protocol(config) if config.mode == 'duration' else None
    if model is not None and model.action_protocol != expected:
        raise ValueError('action-scale protocol mismatch: model/executor; recollect demonstrations')
    if features is not None and features.action_protocol != expected:
        raise ValueError('action-scale protocol mismatch: features/executor')
    from .memory_flow import MemoryHistory,current_event,compact_step
    history_factory=({'history_factory':features.new_history} if features is not None else {})
    if training_fastpath and features is not None and features.runtime=='typed_stream':
        history_factory=dict(history_factory=MemoryHistory)
    environment = Environment(fixture, config, fast_feedback=fast_feedback,record_details=record_details,
                              **history_factory); env = environment.env
    h = environment.history
    typed_cap = model is None and features is not None and features.runtime in ('typed_batch','typed_stream')
    program = Program(source) if model is None and not typed_cap else None
    generator = torch.Generator().manual_seed(seed)
    # Already projected plain-data packets; C-level copying preserves independent
    # replay records without recursively walking thousands of menu entries in Python.
    rows = [dict(action=None, observation=marshal.loads(marshal.dumps(current_event(h)['observation'])), history_hash=h.sha256)] if record_details else []
    steps = []; decisions = []; error = None; stop = 'decision_budget'; started = time.monotonic()
    choice_counts=dict(model_decisions=0,recommended=0,duration_only_deviation=0,motion_deviation=0)
    failure_cost = 2*env.config.time_budget
    prefix_executed = prefix_ticks = 0
    for index in range(max_decisions):
        if time.monotonic()-started >= wall_seconds:
            stop = 'wall_budget'; break
        remaining = max_ticks-env.total_periods
        if remaining <= 0:
            stop = 'execution_budget'; break
        try:
            replaying_prefix = index < len(prefix)
            if replaying_prefix:
                # Update the receiving policy's memory from actual history,
                # including actions it did not recommend. No state teleport.
                if features is not None:
                    with features.decision(h):
                        pass
                action = prefix[index]; trace = None
            elif model is None:
                packet = features.recommendation(h) if typed_cap else program.evaluate(h, generic=False)
                if not packet['first_legal']:
                    raise ValueError('CAP first recommendation missing/illegal')
                action = packet['raw_first']; trace = None
            else:
                with torch.no_grad():
                    trace = model.decide(h, features, generator=generator, deterministic=deterministic,
                                         temperature=temperature)
                action = trace['action']
            if config.mode == 'time' and action['ticks'] > remaining:
                stop = 'execution_budget'; break
            previous_hash=h.sha256 if record_details else None
            event = environment.step(action, remaining_ticks=remaining)
            if replaying_prefix:
                prefix_executed += 1
                prefix_ticks += event['periods']
            terminal = env.termination is not None
            reward = -.05*event['periods']
            if terminal and not env.success:
                # Telescopes to -failure_cost regardless of early failure time.
                reward -= failure_cost-env.total_periods*.05
            d = dict(index=index, action=deepcopy(action), event=event, reward=reward,
                     history_hash=previous_hash)
            if trace is not None:
                choice_counts['model_decisions']+=1
                if trace['recommended']:choice_counts['recommended']+=1
                elif any(all(action[k]==q[k] for k in ('target_id','a0','a1')) for q in trace['panels'][0]['proposals']):
                    choice_counts['duration_only_deviation']+=1
                else:choice_counts['motion_deviation']+=1
                trace.update(periods=event['periods'], reward=reward, terminal=terminal)
                steps.append(compact_step(trace) if training_fastpath and not record_details else trace)
                d.update(recommended=trace['recommended'], recommendations=trace['panels'][0]['proposals'],
                         legal_action_count=trace['legal_action_count'], scored_prefixes=trace['scored_prefixes'],
                         logp=trace['logp'], value=trace['value'], temperature=trace['temperature'],
                         deviation_status='not_a_preference_label' if not trace['recommended'] else 'recommended')
            if record_details:
                decisions.append(d)
                rows.append(dict(action=deepcopy(action), observation=marshal.loads(marshal.dumps(current_event(h)['observation'])), history_hash=h.sha256))
            if on_progress:on_progress(dict(index=index,tick=env.total_periods,action=action,reason=event['reason']))
            if terminal:
                stop = 'terminal'; break
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc)); stop = 'tool_error'; break
    bootstrap = 0.
    if model is not None and steps and not steps[-1]['terminal'] and not error:
        with torch.no_grad():
            tree=ActionTree(current_event(h)['observation'],_projected=isinstance(h,MemoryHistory),
                            _action_index=h.action_index if isinstance(h,MemoryHistory) else None)
            bootstrap = float(model.value(features.panel(h, tree, {})))
    status = 'tool_error' if error else 'success' if env.success else 'task_failure' if env.termination else 'incomplete'
    # Physical anomalies are reported, but excluded from updates like prior RL.
    from experiments.option_rl_v13 import mode_env as mode
    anomalies = getattr(mode.E, 'ANOMALY_TERMINATIONS', ())
    eligible = not error and env.termination not in anomalies
    result = dict(version=VERSION, execution_mode=config.mode, event_config=asdict(config),
                  fixture_sha256=digest(fixture), source_sha256=digest(source),
                  summary=dict(status=status, success=bool(env.success), ticks=env.total_periods,
                               task_cost_seconds=env.total_periods*.05 if env.success else failure_cost if env.termination else None,
                               termination=env.termination, stop=stop, error=error, eligible_for_ppo=eligible,
                               bootstrap=bootstrap, decisions=environment.decision_count, wall_seconds=time.monotonic()-started,
                               choice_counts=choice_counts),
                  decisions=decisions, public_rows=rows, events=environment.events, replay=environment.replay() if record_details else None)
    from coimagine.random_start.lidar_native import runtime_info
    result['summary']['analytic_lidar_backend'] = runtime_info()
    result['recording']=dict(details=record_details,history_identity='episode_cursor_not_evidence_hash' if isinstance(h,MemoryHistory) else 'canonical_hash')
    if features is not None and features.semantic_protocol is not None:
        result["semantic_protocol"]=features.semantic_protocol
        from .stream_history import VERSION as STREAM_VERSION
        result["history_protocol"]=STREAM_VERSION
    if expected is not None:result["action_protocol"]=expected
    if prefix:
        result['prefix_replay'] = dict(requested=len(prefix), executed=prefix_executed,
            ticks=prefix_ticks, complete=prefix_executed==len(prefix),
            suffix_decisions=environment.decision_count-prefix_executed,
            accounting='prefix included in total ticks and task horizon; memory follows actual executed actions')
    return result, steps


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def provenance(out, source, config):
    (out/'source.py').write_text(source)
    files = list(Path(__file__).parent.glob('*.py')) + [
        Path(X.__file__), Path('experiments/planar_tasks/action_semantics/program.py'),
        Path('experiments/planar_tasks/relational_search/contract.py'),
        Path('experiments/planar_tasks/relational_search/environment.py'),
        Path('experiments/planar_tasks/relational_search/ray_query.py'),
        *Path('src/coimagine/random_start').glob('lidar*.py'),
        Path('src/coimagine/random_start/lidar_kernel.c')]
    save(out/'manifest.json', dict(version=VERSION, semantic_sha256=digest(source), config=config,
         source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         purpose='bounded engineering/training; not held-out evaluation'))


def train_run(out, fixture, source, demonstration, *, mode='time', episodes=1, max_ticks=120,
              max_decisions=12, supervised_epochs=80, max_demo_samples=None, seed=17, condition='M'):
    if episodes < 1:
        raise ValueError('positive episode count required')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    config = dict(mode=mode, episodes=episodes, max_ticks=max_ticks, max_decisions=max_decisions,
                  supervised_epochs=supervised_epochs, max_demo_samples=max_demo_samples, seed=seed, condition=condition,
                  gamma=1., trace_seconds=10., residual_coef=.05, nonrecommended_penalty=0.)
    provenance(out, source, config); save(out/'fixture.json', fixture)
    features = Features(source,fast_embedding=True,runtime='cached_fork',
                        action_protocol=protocol(EventConfig(mode=mode)) if mode=='duration' else None)
    traces = demonstration_traces(demonstration, features, split='train', mode=mode, max_samples=max_demo_samples)
    save(out/'demonstration_manifest.json', dict(sha256=digest(demonstration), mode=mode, samples=len(traces),
          source_sha256=demonstration.get('source_sha256'), provenance=demonstration.get('provenance'),
          selection='all' if max_demo_samples is None else 'first N; engineering subset, not full imitation'))
    model, curve = warm_start(traces, features, epochs=supervised_epochs, seed=seed, condition=condition)
    metadata = dict(config=config, demonstration_sha256=digest(demonstration), event_config=asdict(EventConfig(mode=mode)))
    model.save(out/'initial.pt', metadata); save(out/'supervised.json', curve)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    reports = []
    for episode in range(episodes):
        result, steps = rollout(fixture, source, model=model, features=features, config=EventConfig(mode=mode),
                                max_ticks=max_ticks, max_decisions=max_decisions, seed=seed+episode, fast_feedback=True)
        save(out/f'episode_{episode:03d}.json', result)
        # Save immutable feature batches/log probabilities for independent replay.
        save(out/f'batch_{episode:03d}.json', steps)
        update = ppo_update(model, optimizer, steps, bootstrap=result['summary']['bootstrap']) if steps and result['summary']['eligible_for_ppo'] else dict(updates=0, skipped=True)
        reports.append(dict(episode=episode, summary=result['summary'], update=update))
        model.save(out/f'checkpoint_{episode:03d}.pt', dict(metadata, episode=episode))
        # A separate optimizer/RNG file permits a future explicit resume driver.
        torch.save(dict(optimizer=optimizer.state_dict(), rng=torch.get_rng_state()), out/f'optimizer_{episode:03d}.pt')
        save(out/'progress.json', dict(episodes=reports, feature_batches=features.calls,
                                     executed_ticks=sum(r['summary']['ticks'] for r in reports)))
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('collect', 'train'))
    p.add_argument('--source', type=Path, required=True); p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--mode', choices=('time','progress','duration'), default='progress')
    p.add_argument('--demonstration', type=Path); p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--max-ticks', type=int, default=1800); p.add_argument('--max-decisions', type=int, default=600)
    p.add_argument('--supervised-epochs', type=int, default=80); p.add_argument('--max-demo-samples', type=int)
    p.add_argument('--seed', type=int, default=17); p.add_argument('--condition', choices=('S','E','M'), default='M')
    args = p.parse_args(); fixture = json.loads(args.fixture.read_text()); source = args.source.read_text()
    if args.command == 'collect':
        args.out.mkdir(parents=True, exist_ok=False)
        provenance(args.out, source, dict(mode=args.mode, max_ticks=args.max_ticks, max_decisions=args.max_decisions))
        result, _ = rollout(fixture, source, config=EventConfig(mode=args.mode), max_ticks=args.max_ticks, max_decisions=args.max_decisions)
        result['split'] = 'train'; save(args.out/'demonstration.json', result)
        print(json.dumps(result['summary']))
    else:
        if args.demonstration is None:
            p.error('--demonstration required for supervised initialization')
        demonstration = json.loads(args.demonstration.read_text())
        if demonstration.get('split') != 'train':
            p.error('demonstration must explicitly declare split=train')
        reports = train_run(args.out, fixture, source, demonstration, mode=args.mode, episodes=args.episodes,
                max_ticks=args.max_ticks, max_decisions=args.max_decisions, supervised_epochs=args.supervised_epochs,
                max_demo_samples=args.max_demo_samples, seed=args.seed, condition=args.condition)
        print(json.dumps([r['summary'] for r in reports]))


if __name__ == '__main__':
    main()
