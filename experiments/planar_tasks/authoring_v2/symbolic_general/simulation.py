"""Measured physical execution; privileged audit is separate from policy inputs."""
from copy import deepcopy
import time
import math

from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig
from experiments.planar_tasks.semantic_rl.memory_flow import current_event
from experiments.planar_tasks.relational_search.contract import public_observation, digest
from . import frontend
from .scaffold import verify_source


def observation(world):
    return current_event(world.history)['observation']


def truth(world):
    backend = world.env.env.backend
    body = backend.bodies[world.env.primary_id]
    contacts = backend.contacts()
    return dict(xy=[float(body.position.x), float(body.position.y)], angle=float(body.angle),
                object_static=any(c['pair'] == 'object_wall' and c.get('entity_id') == world.env.primary_id for c in contacts),
                robot_static=any(c['pair'] == 'robot_wall' for c in contacts),
                robot_peer=any(c['pair'] == 'robot_robot' for c in contacts))


def measured_step(world, command, limit=None):
    """Audit every actual physics tick, including internal handoffs and braking."""
    before = truth(world); states = []
    original = world.env.joint_step
    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        states.append(truth(world))
        return result
    world.env.joint_step = step
    started = time.monotonic()
    try:
        event = world.step(command, remaining_ticks=limit)
    finally:
        world.env.joint_step = original
    end = truth(world); goal = observation(world)['goal']['xy']
    return dict(event=event, before=before, after=end,
                object_static=before['object_static'] or any(s['object_static'] for s in states),
                robot_static=before['robot_static'] or any(s['robot_static'] for s in states),
                robot_peer=before['robot_peer'] or any(s['robot_peer'] for s in states),
                object_static_ticks=sum(s['object_static'] for s in states),
                goal_gain=math.dist(before['xy'], goal)-math.dist(end['xy'], goal),
                displacement=math.dist(before['xy'], end['xy']), wall_seconds=time.monotonic()-started)


def episode(fixture, source, *, controller=None, max_ticks=1800, max_decisions=450,
            wall_seconds=180, keep_observations=False, replay=False, prefix=None):
    policy = verify_source(source)
    world = Environment(fixture, EventConfig(mode='duration'), fast_feedback=True, record_details=replay)
    memory = frontend.initial_memory(); records = []; public = []; error = None
    started = time.monotonic(); initial = truth(world); prefix = prefix or []
    executed_prefix = []
    for index in range(max_decisions):
        if world.env.termination or world.env.total_periods >= max_ticks: break
        if time.monotonic()-started > wall_seconds:
            error = 'episode_wall_budget'; break
        o = observation(world)
        memory = frontend.update_memory(memory, dict(observation=o))
        z, commands = frontend.semantic_packet(memory, o)
        evidence = None
        if index < len(prefix):
            cmd = prefix[index]; choice = dict(option='prefix', rule='prefix')
        else:
            if controller is not None:
                # Explicitly only public inputs cross this call.
                evidence = controller.decide(deepcopy(memory), public_observation(o), deepcopy(z), deepcopy(commands), policy)
                memory['evidence'] = evidence['permissions']
                z, commands = frontend.semantic_packet(memory, o)
            choice = policy.select(z)
            if choice['option'] is None:
                error = 'no_selected_action'; break
            if not choice['executable']:
                error = 'selected_nonexecutable_action'; break
            cmd = commands[choice['option']]
        if keep_observations:
            public.append(dict(observation=o, memory=deepcopy(memory), action=deepcopy(cmd)))
        try:
            measured = measured_step(world, cmd, max_ticks-world.env.total_periods)
        except (ValueError, KeyError, RuntimeError) as exc:
            error = type(exc).__name__+': '+str(exc); break
        next_memory = frontend.update_memory(deepcopy(memory), dict(observation=observation(world)))
        after_z, _ = frontend.semantic_packet(next_memory, observation(world))
        records.append(dict(index=index, tick=o['tick'], choice=choice, command=cmd, semantic=z,
                            next_semantic=after_z, audit=measured, evidence=evidence,
                            public_sha256=digest(o), memory_sha256=digest(memory)))
        executed_prefix.append(cmd)
    if len(prefix) > len(executed_prefix) and not world.env.success:
        error = error or 'prefix_not_completed'
    summary = dict(success=bool(world.env.success), termination=world.env.termination,
                   ticks=world.env.total_periods, decisions=len(records), error=error,
                   object_static_commands=sum(r['audit']['object_static'] for r in records),
                   object_static_ticks=sum(r['audit']['object_static_ticks'] for r in records),
                   robot_static_commands=sum(r['audit']['robot_static'] for r in records),
                   peer_contact_commands=sum(r['audit']['robot_peer'] for r in records),
                   brake_commands=sum(r['choice']['option'] == 'brake' for r in records),
                   goal_gain=math.dist(initial['xy'], observation(world)['goal']['xy'])-math.dist(truth(world)['xy'], observation(world)['goal']['xy']),
                   task_cost_seconds=world.env.total_periods*.05 if world.env.success else 2*world.env.config.time_budget,
                   wall_seconds=time.monotonic()-started)
    if controller is not None: summary['online'] = controller.ledger
    return dict(summary=summary, records=records, public=public,
                replay=world.replay() if replay else None)


def replay_boundary(fixture, prefix):
    world = Environment(fixture, EventConfig(mode='duration'), fast_feedback=True, record_details=False)
    memory = frontend.initial_memory(); ticks = 0
    for command in prefix:
        memory = frontend.update_memory(memory, dict(observation=observation(world)))
        if world.env.termination: raise ValueError('terminal development prefix')
        event = world.step(command); ticks += event['periods']
    memory = frontend.update_memory(memory, dict(observation=observation(world)))
    return world, memory, ticks


def branch(fixture, prefix, action, source, *, follow_ticks=120, controller=None, retain_root_public=False):
    """Same physical prefix; actual policy selects the successor, which is executed.

    Successor PUSH completion is >=2 cm goal progress in its window; an orbit
    completes on a publicly offered goal-aligned PUSH; MOVE completes >=5 cm
    commanded robot travel. These are evaluation predicates, never extra rewards.
    """
    world, memory, prefix_ticks = replay_boundary(fixture, prefix)
    policy = verify_source(source); o = observation(world)
    root_public_sha256=digest(public_observation(o))
    root_public=public_observation(o)if retain_root_public else None
    z, commands = frontend.semantic_packet(memory, o)
    measured = measured_step(world, commands[action])
    memory = frontend.update_memory(memory, dict(observation=observation(world)))
    after, next_commands = frontend.semantic_packet(memory, observation(world))
    successor_values=after;successor_evidence=None
    if controller is not None and not world.env.termination:
        successor_evidence=controller.decide(deepcopy(memory),public_observation(observation(world)),
                                             deepcopy(after),deepcopy(next_commands),policy)
        memory['evidence']=successor_evidence['permissions']
        successor_values,next_commands=frontend.semantic_packet(memory,observation(world))
    successor = policy.select(successor_values)
    next_name = successor['option']; completion = False; follow = []; follow_error = None
    start = truth(world); initial_robots = [[r['own']['x'], r['own']['y']] for r in observation(world)['robots']]
    begin = world.env.total_periods
    if world.env.success:
        completion = True
    elif next_name and successor['executable']:
        opt = frontend.option_table()[next_name]
        if opt['family'] != 'brake':
            while not world.env.termination and world.env.total_periods-begin < follow_ticks:
                try:
                    row = measured_step(world, next_commands[next_name], follow_ticks-(world.env.total_periods-begin))
                except ValueError as exc:
                    follow_error = str(exc); break
                follow.append(row)
                memory = frontend.update_memory(memory, dict(observation=observation(world)))
                current, next_commands = frontend.semantic_packet(memory, observation(world))
                if opt['family'] == 'push':
                    goal = observation(world)['goal']['xy']
                    completion = math.dist(start['xy'], goal)-math.dist(truth(world)['xy'], goal) >= .02
                elif opt['family'] == 'orbit':
                    completion = any(current['legal:'+name] and current['align_%d' % item['heading']] >= .70
                                     for name, item in frontend.option_table().items() if item['family'] == 'push')
                elif opt['family'] == 'move':
                    d = opt['heading']*math.pi/4
                    completion = all((r['own']['x']-old[0])*math.cos(d)+(r['own']['y']-old[1])*math.sin(d) >= .05
                                     for r, old in zip(observation(world)['robots'], initial_robots))
                if completion or world.env.success:
                    completion = True; break
                if row['event']['periods'] == 0: break
    unsafe = measured['object_static'] or any(r['object_static'] for r in follow)
    return dict(action=action, semantic=z, next_semantic=after, audit=measured,
                root_public_sha256=root_public_sha256,
                root_public=root_public,
                successor=successor,successor_semantic=successor_values,successor_evidence=successor_evidence,
                successor_completed=completion,
                safe_effective=bool(completion and not unsafe), full_object_static=unsafe,
                follow_error=follow_error, follow=follow, prefix_ticks=prefix_ticks,
                follow_ticks=world.env.total_periods-begin, root_ticks=measured['event']['periods'])
