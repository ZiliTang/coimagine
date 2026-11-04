"""Self-contained, auto-refreshing dashboard for the event-progress PPO runner.

The page is rewritten in place (atomic replace) by the trainer after every
iteration and by the supervisor every ten seconds, so a browser refresh shows
the newest curves and the live phase without any server or CDN.
"""
from __future__ import annotations
from pathlib import Path
import html
import json
import os

from experiments.option_rl_v45.monitor import TEMPLATE
from experiments.option_rl_v15.live import read_json

CHARTS = [
    ('Return', [('explore_return', 'train', '#4aa3ff'), ('eval_return', 'greedy validation', '#ffb347')]),
    ('Success', [('explore_success', 'train', '#4ad991'), ('success_rate', 'greedy validation', '#ffb347')]),
    ('Object progress (fraction of initial distance)',
     [('explore_progress', 'train', '#4ad991'), ('mean_progress', 'validation', '#ffb347')]),
    ('Final object-to-goal distance (m)',
     [('explore_final_distance', 'train', '#4aa3ff'), ('eval_final_distance', 'validation', '#ffb347')]),
    ('Decisions per episode', [('explore_decisions', 'train', '#4aa3ff'), ('eval_decisions', 'validation', '#ffb347')]),
    ('Decision interval (physical seconds)',
     [('explore_decision_seconds', 'train', '#4aa3ff'), ('eval_decision_seconds', 'validation', '#ffb347')]),
    ('Contact acquisition (episodes with a contact event)',
     [('explore_contact_rate', 'train', '#4ad991'), ('eval_contact_rate', 'validation', '#ffb347')]),
    ('Event reasons per decision', [('event_progress_rate', 'progress', '#4ad991'),
                                    ('event_contact_rate', 'contact', '#4aa3ff'),
                                    ('event_wait_rate', 'wait', '#8b93a7'),
                                    ('event_timeout_rate', 'timeout', '#ffb347'),
                                    ('event_failure_rate', 'failure', '#ff6b6b'),
                                    ('event_contact_lost_rate', 'contact lost', '#c792ea'),
                                    ('event_direction_unavailable_rate', 'direction unavailable', '#ff87ce'),
                                    ('event_yield_blocked_rate', 'yield blocked', '#9f88ff')]),
    ('Reward components (mean per episode)', [('reward_progress', 'progress', '#4ad991'),
                                              ('reward_distance_cost', 'distance cost', '#ff6b6b'),
                                              ('reward_time_cost', 'time cost', '#ffb347'),
                                              ('reward_success_bonus', 'success bonus', '#4aa3ff'),
                                              ('reward_failure_penalty', 'failure penalty', '#c792ea')]),
    ('Joint action mix (robot decisions)', [('action_push', 'PUSH', '#4ad991'),
                                            ('action_reposition', 'REPOSITION', '#4aa3ff'),
                                            ('action_wait', 'BRAKE', '#8b93a7'),
                                            ('action_yield', 'YIELD', '#00d7df')]),
    ('Motor effort (diagnostic only, no penalty)',
     [('diagnostic_mean_motor_force_N', 'mean force N', '#4aa3ff'),
      ('diagnostic_effort_integral_s', 'squared normalized force integral', '#ffb347')]),
    ('Opposing contact per episode (physical seconds)',
     [('diagnostic_opposing_contact_s', 'opposing contact', '#ffd166'),
      ('diagnostic_stalled_opposing_contact_s', 'opposing with little object motion', '#ff6b6b'),
      ('eval_diagnostic_stalled_opposing_contact_s', 'greedy stalled opposition', '#c792ea')]),
    ('YIELD execution per episode (robot seconds)',
     [('diagnostic_yield_robot_s', 'executing', '#4aa3ff'),
      ('diagnostic_yield_active_robot_s', 'active', '#00d7df'),
      ('diagnostic_yield_blocked_robot_s', 'blocked', '#ff6b6b')]),
    ('Non-pusher object contact (robot seconds)',
     [('diagnostic_wait_object_contact_robot_s', 'BRAKE', '#ffb347'),
      ('diagnostic_yield_object_contact_robot_s', 'YIELD', '#00d7df')]),
    ('Policy loss', [('policy_loss', 'policy', '#c792ea')]),
    ('Value loss', [('value_loss', 'value', '#4aa3ff')]),
    ('Entropy (joint robot actions)', [('entropy', 'entropy', '#ffd166')]),
    ('Advantage / target scale', [('advantage_std', 'advantage std', '#c792ea'),
                                  ('return_target_std', 'value target std', '#4aa3ff')]),
    ('Value explained variance', [('explained_variance', 'EV', '#4aa3ff')]),
    ('PPO KL / clip fraction', [('approx_kl', 'approx KL', '#ffd166'), ('clip_fraction', 'clip fraction', '#c792ea')]),
    ('Sample / PPO log-prob error', [('initial_logp_max_error', 'max absolute error', '#ff6b6b')]),
    ('Actor parameter gradient (pre-clip)', [('actor_parameter_grad_norm', 'actor', '#4ad991')]),
    ('Critic parameter gradient (pre-clip)', [('critic_parameter_grad_norm', 'critic', '#4aa3ff')]),
    ('Anomaly rate', [('train_anomaly_rate', 'train quarantined', '#ff6b6b'),
                      ('eval_anomaly_rate', 'validation counted as failures', '#ffb347')]),
    ('Samples / iteration', [('steps', 'PPO transitions', '#4aa3ff'), ('valid_episodes', 'valid episodes', '#4ad991')]),
    ('Wall seconds / iteration', [('iteration_seconds', 'iteration', '#8b93a7'),
                                  ('eval_seconds', 'validation', '#ffb347')]),
    ('Sampling throughput (control periods / wall second)',
     [('sample_periods_per_wall_second', 'rollout', '#4ad991')]),
]


def write_dashboard(history, path, *, subtitle='', live=None, supervision=None):
    path = Path(path)
    live = live if live is not None else (read_json(path.parent/'live_status.json', {}) or {})
    supervision = supervision if supervision is not None else (read_json(path.parent/'supervisor_status.json', {}) or {})
    content = TEMPLATE.format(title='V2 · event-progress PPO', subtitle=html.escape(subtitle),
                              data=json.dumps(history, allow_nan=False).replace('<', '\\u003c'),
                              charts=json.dumps(CHARTS))
    content = content.replace('minmax(430px,1fr)', 'minmax(min(100%,360px),1fr)')
    status = ('<div id="live-status" style="padding:10px;border:1px solid #344256;border-radius:7px">'
              + html.escape(
                  f"状态 {live.get('phase', 'starting')} · 轮次 {live.get('iteration', 0)} · "
                  f"训练回合 {live.get('episode', 0)}/{live.get('episodes', 0)} · "
                  f"采样 workers {live.get('workers', '—')} · "
                  f"验证 {live.get('eval_done', 0)}/{live.get('eval_total', 0)} · "
                  f"更新时间 {live.get('updated_local', '—')}") + '</div>')
    status += ('<div style="padding:8px 0;color:#a9bdd6">'
               + html.escape('训练曲线每轮写入；固定验证按计划重测，空点不伪造评估值。页面每 10 秒刷新。')
               + '</div>')
    status += ('<div id="supervision">'
               + html.escape('监测：' + str(supervision.get('decision', 'waiting')))
               + ' · <a style="color:#8bbdff" href="monitor_summary.md">阶段总结</a>'
               + ' · <a style="color:#8bbdff" href="training.log">训练日志</a></div>')
    content = content.replace('</header>', status + '</header>')
    content = content.replace('</style>',
                              'header span {overflow-wrap:anywhere;word-break:break-word;white-space:normal;max-width:100%} </style>')
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(content)
    temp.replace(path)
    return path
