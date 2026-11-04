"""Detached supervisor for the event-progress long run.

Launches ``long_run.py`` in its own session, rewrites the dashboard every ten
seconds (so the phase and evaluation progress stay live), keeps a one-minute
monitor summary, and requests a clean stop on explicit fault rules. It never
restarts a failed run, never changes reward/controller/physics, and keeps all
partial evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.option_rl_v15.live import atomic_json, read_json
from experiments.option_rl_v21.monitor import write_dashboard
from experiments.option_rl_v21.long_run import source_files

PYTHON = str(ROOT / '.venv/bin/python')


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def finite_row(row):
    for value in row.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            return False
    return True


def fault(history, live, plan, now, *, elapsed=None, heartbeat_age=None):
    for row in history:
        if not finite_row(row):
            return 'nonfinite_history'
    heartbeat = live.get('updated_unix') if live else None
    age = heartbeat_age if heartbeat_age is not None else (now-heartbeat if heartbeat is not None else 0.)
    if age > 300:
        return 'heartbeat_stale_over_300s'
    if len(history) >= 3 and all(r.get('train_anomaly_rate', 0) >= .25 for r in history[-3:]):
        return 'train_anomaly_rate_ge_25pct_three_iterations'
    evaluation = [r for r in history if r.get('success_rate') is not None]
    if len(evaluation) >= 2 and all(r.get('eval_anomaly_rate', 0) >= .10 for r in evaluation[-2:]):
        return 'validation_anomaly_rate_ge_10pct_twice'
    if len(history) >= 3 and all(r.get('approx_kl', 0) > .5 for r in history[-3:]):
        return 'ppo_kl_over_0_5_three_iterations'
    if len(history) >= 10 and len(evaluation) >= 2 and all(r.get('entropy', 1) < .1 for r in history[-10:]):
        if all(r.get('eval_outside_goal_success', 1) == 0 and r.get('mean_progress', 1) <= 0
               for r in evaluation[-2:]):
            return 'entropy_collapse_without_nontrivial_validation_success'
    manifest = read_json(Path(plan['run']) / 'manifest.json', {}) or {}
    for name, digest in (manifest.get('source_sha256') or plan.get('source_sha256', {})).items():
        path = ROOT / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return 'runtime_source_changed:' + name
    if (elapsed is not None and elapsed > plan['hours']*3600+600) or (elapsed is None and now > plan['deadline_unix']+600):
        return 'supervisor_hard_deadline'
    return None


def request_stop(out, reason):
    """The trainer reads run/STOP.json; mirror it at the dashboard root too."""
    data = dict(reason=reason, source='supervisor', requested_local=stamp())
    for path in (Path(out)/'run/STOP.json', Path(out)/'STOP.json'):
        if not path.exists():
            atomic_json(path, data)


def summary_lines(out, plan, status, history, live):
    last = history[-1] if history else {}
    evaluation = [r for r in history if r.get('success_rate') is not None]
    latest_eval = evaluation[-1] if evaluation else {}
    best = max(evaluation, key=lambda r: (r.get('success_rate', 0), r.get('mean_progress', -1))) if evaluation else {}
    lines = ['# Event-progress RL 本地监测摘要', '',
             f'更新时间：{stamp()}；决策：{status.get("decision", "running")}；阶段：{live.get("phase", "—")}；'
             f'最近完成训练轮次：{last.get("iteration", 0)}。',
             '', f'运行目录：`{out}`；训练器目录：`{Path(out) / "run"}`；训练日志：`training.log`；dashboard：`dashboard.html`。',
             f'并行采样：{plan.get("workers", 1)} workers；每轮 {plan.get("episodes_per_iteration")} 个有效 episode。',
             '', '本文件由本地监督进程每分钟更新；不是未来人工或模型复核的承诺。',
             '训练曲线每轮写入；固定验证按计划重测；异常回合不从分母删除。', '']
    if last:
        lines += [f'- 训练：return {last.get("explore_return"):.4f}；success {last.get("explore_success"):.3f}；'
                  f'progress {last.get("explore_progress"):.4f}；decisions {last.get("explore_decisions"):.1f}；'
                  f'接触率 {last.get("explore_contact_rate"):.2f}。',
                  f'- PPO：policy loss {last.get("policy_loss"):.4g}；value loss {last.get("value_loss"):.4g}；'
                  f'entropy {last.get("entropy"):.3f}；KL {last.get("approx_kl"):.4g}；'
                  f'logp 误差 {last.get("initial_logp_max_error"):.2g}。',
                  f'- 事件构成：progress {last.get("event_progress_rate", 0):.2f}；contact {last.get("event_contact_rate", 0):.2f}；'
                  f'wait {last.get("event_wait_rate", 0):.2f}；timeout {last.get("event_timeout_rate", 0):.2f}；'
                  f'failure {last.get("event_failure_rate", 0):.2f}。',
                  f'- V2：YIELD动作比例 {last.get("action_yield", 0):.3f}；'
                  f'平均施力 {last.get("diagnostic_mean_motor_force_N", 0):.3f}N；'
                  f'低运动对抗 {last.get("diagnostic_stalled_opposing_contact_s", 0):.2f}s/episode；'
                  f'YIELD blocked {last.get("diagnostic_yield_blocked_robot_s", 0):.2f} robot-s/episode。',
                  f'- 吞吐：{last.get("steps", 0)} transitions/轮；{last.get("sample_periods_per_wall_second", 0):.0f} '
                  f'control periods/墙钟秒；轮次 {last.get("iteration_seconds", 0):.1f}s。']
    if latest_eval:
        lines += ['', f'- 最近完整验证（第 {latest_eval.get("iteration")} 轮）：success {latest_eval.get("success_rate"):.3f}；'
                  f'progress {latest_eval.get("mean_progress"):.4f}；return {latest_eval.get("eval_return"):.4f}；'
                  f'outside-goal success {latest_eval.get("eval_outside_goal_success")}；'
                  f'anomaly rate {latest_eval.get("eval_anomaly_rate", 0):.3f}。',
                  f'- 历史较好验证：第 {best.get("iteration")} 轮，success {best.get("success_rate"):.3f}，'
                  f'progress {best.get("mean_progress"):.4f}。']
    lines += ['', '停止规则：非有限数值；心跳超过 300 秒；连续三轮训练异常 ≥25%；连续两次验证异常 ≥10%；'
              '连续三轮 PPO KL >0.5；持续低熵且无非平凡验证成功；运行中关键源码改变；监督硬截止。',
              '规则触发时写 STOP.json，训练器在当前回合边界停止并保存结果；不自动重启或改奖励。',
              'checkpoint.pt 每轮保存；best.pt及其逐根评估保留；周期快照/逐轮原始评估按plan中的retention上限保留。',
              '施力/对抗仅作诊断，没有施力惩罚；test面板不在训练中执行。']
    return '\n'.join(lines) + '\n'


def write_summary(out, plan, status, history, live):
    path = Path(out) / 'monitor_summary.md'
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(summary_lines(out, plan, status, history, live))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--hours', type=float, default=12.)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--episodes', type=int, default=8,
                        help='valid episodes per PPO iteration, summed over workers')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--entropy', type=float, default=.02)
    parser.add_argument('--trace-seconds', type=float, default=10.)
    parser.add_argument('--target-kl', type=float, default=.03)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--eval-roots', type=int, default=24)
    parser.add_argument('--test-roots', type=int, default=32)
    parser.add_argument('--test-seed', type=int, default=None)
    parser.add_argument('--snapshot-every', type=int, default=50)
    parser.add_argument('--keep-snapshots', type=int, default=8)
    parser.add_argument('--keep-evaluations', type=int, default=8)
    parser.add_argument('--panel', choices=['dynamic', 'micro'], default='dynamic')
    parser.add_argument('--panel-path', type=Path)
    parser.add_argument('--interval', type=float, default=10.)
    parser.add_argument('--summary-every', type=float, default=60.)
    args = parser.parse_args()

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('supervisor requires a new empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    run = out / 'run'
    command = [PYTHON, '-u', '-m', 'experiments.option_rl_v21.long_run',
               '--out', str(run), '--hours', str(args.hours), '--iterations', str(args.iterations),
               '--episodes', str(args.episodes), '--seed', str(args.seed), '--hidden', str(args.hidden),
               '--epochs', str(args.epochs), '--lr', str(args.lr), '--entropy', str(args.entropy),
               '--trace-seconds', str(args.trace_seconds), '--target-kl', str(args.target_kl),
               '--eval-every', str(args.eval_every), '--eval-roots', str(args.eval_roots),
               '--test-roots', str(args.test_roots), '--snapshot-every', str(args.snapshot_every),
               '--keep-snapshots', str(args.keep_snapshots), '--keep-evaluations', str(args.keep_evaluations),
               '--panel', args.panel, '--workers', str(args.workers)]
    if args.panel_path:
        command += ['--panel-path', str(Path(args.panel_path).resolve())]
    if args.test_seed is not None:
        command += ['--test-seed', str(args.test_seed)]
    started = time.time()
    started_monotonic = time.monotonic()
    plan = dict(out=str(out), run=str(run), command=command, workers=args.workers,
                episodes_per_iteration=args.episodes, started_unix=started, started_local=stamp(),
                deadline_unix=started + args.hours * 3600, hours=args.hours,
                budget_clock='monotonic; displayed wall deadline is informational',
                test_seed=args.seed if args.test_seed is None else args.test_seed,
                retention=dict(snapshot_every=args.snapshot_every, keep_snapshots=args.keep_snapshots,
                               keep_evaluations=args.keep_evaluations),
                interval=args.interval, summary_every=args.summary_every,
                source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in source_files()},
                note='one bounded run; supervisor never restarts a failed run or edits its config')
    atomic_json(out / 'plan.json', plan)

    with (out / 'training.log').open('w') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    atomic_json(out / 'processes.json', dict(supervisor_pid=os.getpid(), trainer_pid=child.pid,
                                             out=str(out), started_local=stamp()))
    print(f'[{stamp()}] supervisor {os.getpid()} started trainer {child.pid}: {out}', flush=True)

    last_summary = 0.
    last_heartbeat = None
    last_heartbeat_monotonic = started_monotonic
    stop_requested_at = None
    term_sent_at = None
    status = dict(decision='running', reason=None, checked_local=stamp())
    while child.poll() is None:
        history = read_json(run / 'history.json', []) or []
        live = read_json(run / 'live_status.json', {}) or {}
        now_monotonic = time.monotonic()
        if live.get('updated_unix') != last_heartbeat:
            last_heartbeat = live.get('updated_unix')
            last_heartbeat_monotonic = now_monotonic
        elapsed = now_monotonic-started_monotonic
        manual = read_json(out/'STOP.json', {}) or read_json(run/'STOP.json', {}) or {}
        reason = manual.get('reason') or fault(history, live, plan, time.time(), elapsed=elapsed,
                                              heartbeat_age=now_monotonic-last_heartbeat_monotonic)
        if reason is None and elapsed >= args.hours*3600:
            reason = 'wall_budget_reached'
        if reason:
            request_stop(out, reason)
            if stop_requested_at is None:
                stop_requested_at = now_monotonic
                print(f'[{stamp()}] stop: {reason}; run/STOP.json written', flush=True)
        status = dict(decision='stop_requested' if reason else 'running', reason=reason,
                      checked_local=stamp(), trainer_alive=True,
                      iteration=history[-1].get('iteration') if history else None)
        atomic_json(out / 'supervisor_status.json', status)
        try:
            write_dashboard(history, out / 'dashboard.html',
                            subtitle=f'{out} | event-progress | dynamic roots | gamma=1', live=live,
                            supervision=status)
        except Exception as error:  # the trainer may be mid-write; retry next tick
            print(f'[{stamp()}] dashboard rewrite skipped: {error}', flush=True)
        if time.monotonic() - last_summary >= args.summary_every:
            write_summary(out, plan, status, history, live)
            last_summary = time.monotonic()
        if term_sent_at is None and ((stop_requested_at is not None and now_monotonic-stop_requested_at > 120)
                                      or elapsed > args.hours*3600+600):
            print(f'[{stamp()}] hard deadline; terminating trainer process group', flush=True)
            os.killpg(child.pid, signal.SIGTERM)
            term_sent_at = now_monotonic
        elif term_sent_at is not None and now_monotonic-term_sent_at > 30:
            os.killpg(child.pid, signal.SIGKILL)
        time.sleep(args.interval)

    history = read_json(run / 'history.json', []) or []
    live = read_json(run / 'live_status.json', {}) or {}
    result = read_json(run / 'result.json', {}) or {}
    final = dict(exit_code=child.returncode, result=result, completed_local=stamp(),
                 iterations=len(history), trainer_pid=child.pid)
    atomic_json(out / 'supervisor_result.json', final)
    decision = (result.get('status') or 'complete') if child.returncode == 0 else 'trainer_exit_failed'
    status = dict(decision=decision, reason=status.get('reason'), checked_local=stamp(),
                  trainer_alive=False)
    atomic_json(out / 'supervisor_status.json', status)
    try:
        write_dashboard(history, out / 'dashboard.html',
                        subtitle=f'{out} | event-progress | dynamic roots | gamma=1', live=live,
                        supervision=status)
        write_summary(out, plan, status, history, live)
    except Exception as error:
        print(f'[{stamp()}] final dashboard/summary failed: {error}', flush=True)
    print(f'[{stamp()}] trainer exited {child.returncode}: {json.dumps(final, ensure_ascii=False)}', flush=True)
    raise SystemExit(0 if child.returncode == 0 else 1)


if __name__ == '__main__':
    main()
