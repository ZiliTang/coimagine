"""Execute one duration-capped joint request; normal skill phases stay internal."""
from copy import deepcopy
from dataclasses import asdict
from experiments.planar_tasks.relational_search.contract import validate_action
from .scale import protocol, CONTROL_DT
from .memory_flow import current_event


def step_duration(world, requested, remaining_ticks):
    env=world.env
    if env.termination is not None:raise ValueError('terminal environment')
    if remaining_ticks is not None and remaining_ticks<1:raise ValueError('no execution budget')
    before=current_event(world.history)['observation'];validate_action(requested,before)
    start=env.total_periods;limit=min(requested['ticks'],remaining_ticks or requested['ticks'])
    env.command_index=world.decision_count
    handoffs=[];segments=[];handoff=None;reason='duration'
    for _ in range(limit):
        if handoff is None or handoff['phase']=='switch_brake':
            if env.total_periods!=start:
                before=env.public()
                try:validate_action(requested,before)
                except (ValueError,TypeError,KeyError):
                    reason='target_unavailable';break
            handoff=env.binding.select(requested,before)
            handoffs.append(dict(tick=env.total_periods,**deepcopy(handoff)))
            if handoff['commit']:
                env._commit_target(handoff['target'])
                before=env.public()  # legal sensors of the newly bound target
            world.detector.begin(handoff['command'],before)
        executed=handoff['command'];pair=[executed['a0'],executed['a1']]
        if not segments or segments[-1]['pair']!=pair:
            segments.append(dict(start_tick=env.total_periods,end_tick=env.total_periods,pair=pair))
        env.joint_step(*pair);segments[-1]['end_tick']=env.total_periods
        feedback=world.feedback() if world.fast_feedback else env.public()
        detected=world.detector.update(feedback)
        if env.termination:
            reason='terminal';break
        if handoff['phase']=='switch_brake':
            continue  # internal braking/ownership handshake, no actor call
        if detected in ('contact_lost','blocked','controller_end'):
            reason=detected;break
        # Stable contact and ordinary travel are normal internal milestones.
    periods=env.total_periods-start
    if periods==limit and limit<requested['ticks'] and reason=='duration':reason='collector_cut'
    public=env.public()  # maintain historical boundary numbering
    env.command_prefix.append(deepcopy(requested))
    world._append_public(public,requested)
    event=dict(start_tick=start,end_tick=env.total_periods,requested=deepcopy(requested),
               executed=deepcopy(handoff['command']),handoff=deepcopy(handoff),handoffs=handoffs,
               segments=segments,termination=env.termination,reason=reason,periods=periods,
               success=bool(env.success),config=asdict(world.config),action_protocol=protocol(world.config),
               scale=dict(kind='duration_cap',requested_ticks=requested['ticks'],executed_ticks=periods,
                          requested_seconds=requested['ticks']*CONTROL_DT,executed_seconds=periods*CONTROL_DT))
    world.decision_count+=1
    if world.record_details:world.events.append(event)
    return event
