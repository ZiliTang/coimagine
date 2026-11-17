"""Explicit action-scale identity shared by new demos, actor and PPO.

The wire action remains (target_id,a0,a1,ticks). ticks is always a duration
cap in control ticks, never silently reinterpreted as metres or degrees.
"""
from dataclasses import asdict
from experiments.planar_tasks.relational_search.action_scale import CONTROL_DT, MIN_TICKS, MAX_TICKS, duration_seconds


def protocol(config):
    modes = {'time': 'legacy-command-time-v1', 'progress': 'legacy-time-cap-events-v1',
             'duration': 'duration-cap-fault-interrupt-v1'}
    return dict(version=modes[config.mode], scale='duration_cap', unit='control_ticks',
                control_dt=CONTROL_DT, min_ticks=MIN_TICKS, max_ticks=MAX_TICKS,
                shared_joint_scale=True, includes_approach=True, includes_target_handoff=True,
                event_config=asdict(config))


def check_trace(model, trace):
    expected = getattr(model, 'action_protocol', None)
    if expected != trace.get('action_protocol'):
        raise ValueError('action-scale protocol mismatch: model/trace')
    if expected is not None:
        actual=trace.get('periods')
        requested=trace['action']['ticks']
        duration_seconds(requested)
        if actual is not None and (type(actual) is not int or not 1 <= actual <= requested):
            raise ValueError('executed periods outside requested scale')
