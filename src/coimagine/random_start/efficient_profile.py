"""Explicit candidate profile for cheaper next-round option/RL experiments.

Historical menus/configurations remain selectable for reproduction. This module
never resumes a checkpoint, changes task costs, or starts a training process.
"""
from dataclasses import asdict,replace
from functools import partial
from .env import RandomStartEnv,EnvConfig
from .effect_push import (EffectPushConfig,make_effect_skill,joint_action_mask,
                          ACTIONS_EFFICIENT,EFFICIENT_OPTION_VERSION)

ACTIONS=ACTIONS_EFFICIENT
VERSION=EFFICIENT_OPTION_VERSION
CONFIG=EffectPushConfig(endpoint_tracking=True,fast_approach=True,same_face_allocation=True,
    reposition_boundary='rounded_scan',reposition_peer_detour='outer_arc',
    translation_completion='normal_travel',post_push='contact_follow',contact_recovery='joint_approach',
    follow_completion='peer_push',standalone_follow_gap=.02,standalone_follow_peer_guard='scan_projection_v1',
    mixed_contact_allocation=True,approach_normal_speed=.15,approach_tangential_speed=.30,
    approach_tangential_gain=3.3,preparation_policy='blended',solo_approach_gap=.03,progress_guard=True,
    contact_tangent_damping=.5)
ROTATION_CONFIG=replace(CONFIG,preparation_policy='legacy',solo_approach_gap=.03,
    approach_normal_speed=.10,approach_tangential_speed=.20,approach_tangential_gain=3.,contact_tangent_damping=0.)


def skill_factory(action,force_limit):
    return make_effect_skill(action,force_limit,config=ROTATION_CONFIG
                             if getattr(action,'effect_kind',None)=='rotate' else CONFIG)


def make_env(instance):
    return RandomStartEnv(instance,EnvConfig(peer_clearance_recovery='boundary_v1',lidar_backend='numpy_batch'),
        skill_factory=skill_factory,action_table=ACTIONS,action_version=VERSION)


def mask(world):
    return joint_action_mask(world.boundary_observations(),CONFIG,actions=ACTIONS)


def metadata():
    return dict(version=VERSION,actions=[asdict(a) for a in ACTIONS],controller=asdict(CONFIG),
        rotation_controller=asdict(ROTATION_CONFIG),
        env=EnvConfig(peer_clearance_recovery='boundary_v1',lidar_backend='numpy_batch').as_dict(),
        mask='FOLLOW requires PUSH peer; single WAIT[2s]; PUSH+RP and dual PUSH retained',
        information='centralized sharing of local observations; no finite-bit deployment claim')
