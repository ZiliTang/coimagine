"""Export official host receipts, public author notes and aggregate usage of the two matched CAP authors."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo

from experiments.planar_tasks.relational_search.pipeline import read, save
from tools.cap_checked_pair_author import OUT, ROOT

DOC = ROOT / 'docs/research/key_findings/2026-11-24_checker_author_pair'
ROLES = {'control': '/root/cap_pair_formal_reference', 'checker': '/root/cap_pair_formal_checked'}
LABELS = {'control': '普通反馈', 'checker': '检查反馈'}
FIELDS = ('total_tokens','input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')
LOCAL = ZoneInfo('Asia/Shanghai')


def session_usage(role, start):
    db = Path.home()/'.codex/state_5.sqlite'
    with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as conn:
        rows = conn.execute('SELECT id,rollout_path FROM threads WHERE cwd=? AND agent_path=? AND created_at>=?',
                            (str(ROOT),ROLES[role],int(start)-60)).fetchall()
    if len(rows) != 1:
        return dict(agent=ROLES[role],status='missing' if not rows else 'ambiguous',records=[])
    thread_id, path = rows[0]
    points = []
    with Path(path).open('rb') as stream:
        # Read only the usage event payloads. Messages and model reasoning are not exported.
        for line in stream:
            if b'"type":"token_usage_record"' not in line[:128]:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record['type'] != 'token_usage_record' or record['payload']['thread_id'] != thread_id:
                raise ValueError('inconsistent token usage stream')
            stats = record['payload']['thread_token_usage']
            counts = {k: stats[k] for k in FIELDS}
            if points and any(counts[k] < points[-1]['counts'][k] for k in FIELDS):
                raise ValueError('usage counter decreased')
            at = datetime.fromisoformat(record['timestamp'].replace('Z','+00:00')).timestamp()
            points.append(dict(at=at,counts=counts))
    return dict(agent=ROLES[role],thread_id=thread_id,status='observed' if points else 'pending',records=points)


def public_notes(path, previous):
    rows = []
    if path.exists():
        for raw in path.read_bytes().splitlines():
            try:
                value = json.loads(raw)
            except ValueError:
                continue
            key = hashlib.sha256(raw).hexdigest()
            rows.append(dict(event=value, row_sha256=key, first_observed=previous.get(key,time.time())))
    return rows


def last_record(points, cutoff):
    return next((p for p in reversed(points) if p['at'] <= cutoff), None)


def paired(rows, parent):
    rescued = []
    lost = []
    success = 0
    for key,r in rows.items():
        s = r['success']
        b = parent[key]['success']
        success += s
        if s > b:
            rescued.append(key)
        elif s < b:
            lost.append(key)
    return dict(n=len(rows),success=success,rescued=rescued,lost=lost)


def arm_data(role, parent, previous_notes):
    folder = OUT/'authors'/role
    ledger = OUT/'ledger'/role
    clock = read(ledger/'clock.json')
    frozen_path, closed_path = OUT/'frozen'/(role+'.json'), ledger/'closed.json'
    final = read(frozen_path) if frozen_path.exists() else None
    closed = read(closed_path) if closed_path.exists() else None
    cutoff = final['received'] if final else min(clock['deadline'],time.time())
    meter = session_usage(role,clock['start'])
    points = meter.pop('records')
    meter['num_records_through_cutoff'] = sum(p['at'] <= cutoff for p in points)
    meter['at_cutoff'] = last_record(points,cutoff)
    prior_counts = {k:0 for k in FIELDS}
    candidates = []
    for claim_path in sorted(ledger.glob('trial*.claimed.json')):
        name = claim_path.name.split('.')[0]
        registered = read(claim_path)['registered']
        data_dir = ledger/name
        req = read(data_dir/'request.json') if (data_dir/'request.json').exists() else {}
        result = read(data_dir/'result.json') if (data_dir/'result.json').exists() else None
        err = ledger/(name+'.error.json')
        error = read(err) if err.exists() else None
        request_file = folder/'requests'/(name+'.json')
        public_request = read(request_file) if request_file.exists() else {}
        if not isinstance(public_request,dict):
            public_request = {}
        evidence = result['results'] if result else {
            p.stem: read(p)['result'] for p in sorted((OUT/'dev_runs'/role/name).glob('*.json'))}
        counts = last_record(points,registered)
        delta = {k:counts['counts'][k]-prior_counts[k] for k in FIELDS} if counts and prior_counts else None
        status = 'error' if error else 'complete' if result else 'incomplete' if closed else 'running'
        candidates.append(dict(name=name,registered=registered,source_sha256=req.get('source_sha256'),
                           submitted_sha256=result.get('source_sha256') if result else None,
                           requested=req.get('rooms'),intent=req.get('request',public_request).get('intent'),
                           status=status,paired=paired(evidence,parent),error=error,usage_asof_receipt=counts,
                           usage_since_previous_receipt=delta))
        prior_counts = counts['counts'] if counts else None
    status = 'frozen' if final else 'incomplete' if closed else 'running'
    return dict(role=role,status=status,clock=clock,cutoff=cutoff,final=final,incomplete=closed,telemetry=meter,
                candidates=candidates,actual_development_runs=sum(r['paired']['n'] for r in candidates),
                reserved_development_runs=sum(len(r['requested'] or []) for r in candidates),
                notes=public_notes(folder/'worklog.jsonl',previous_notes))


def fmt(at):
    return datetime.fromtimestamp(at,LOCAL).strftime('%H:%M:%S')


def number(value):
    return '待取得' if value is None else f'{value:,}'


def markdown(data):
    lines = ['# 两位作者的开发记录','',
        '两位作者各有 20 分钟和 3 次候选提交机会，每次最多试跑同一组 16 个开发房间。下表的救回和丢失均与共同母策略在实际完成的同一批房间逐项比较。时间为北京时间。','',
        '| 组别 | 宿主签收 | 候选 | 成功 / 已完成 | 救回 / 丢失 | 本段 tokens | 累计 tokens |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for role, row in data['arms'].items():
        for trial in row['candidates']:
            t = trial['paired']
            delta, observed = trial['usage_since_previous_receipt'], trial['usage_asof_receipt']
            status = {'complete':'','running':'（试跑中）','error':'（异常）','incomplete':'（未完成）'}[trial['status']]
            tested = f"{t['success']} / {t['n']}" if t['n'] else '未执行'
            paired_result = f"{len(t['rescued'])} / {len(t['lost'])}" if t['n'] else '—'
            lines.append(f"| {LABELS[role]} | {fmt(trial['registered'])} | {trial['name']}{status} | {tested} | {paired_result} | {number(delta['total_tokens'] if delta else None)} | {number(observed['counts']['total_tokens'] if observed else None)} |")
    for role, row in data['arms'].items():
        title = LABELS[role]
        last = row['telemetry']['at_cutoff']
        counts = last['counts'] if last else {}
        frozen = row['final']
        selected = '开发中' if row['status']=='running' else '未完成' if not frozen else f"定稿 {frozen['name']}（{fmt(frozen['received'])}）"
        lines.extend(['',f'## {title}','','- '+f"宿主于 {fmt(row['clock']['start'])} 开始计时，{selected}。累计实际试跑 {row['actual_development_runs']} 个房间次，预订 {row['reserved_development_runs']} 个房间次。",
            '- '+f"截至{ '签收定稿' if frozen else '当前统计'}，累计 {number(counts.get('total_tokens'))} tokens。输入 {number(counts.get('input_tokens'))}，其中缓存输入 {number(counts.get('cached_input_tokens'))}，输出 {number(counts.get('output_tokens'))}。"])
        for trial in row['candidates']:
            intent = trial['intent']
            if intent and not frozen:
                lines.append(f"- {trial['name']} 提交说明（作者原文） {intent}")
            if trial['error']:
                message = str(trial['error'].get('message','')).splitlines()[0]
                if message.startswith('native compilation failed'):
                    message='生成的 Cython 源码未通过本地编译'
                elif message.startswith('source at most'):
                    message='提交的源程序超出 48 KiB'
                elif message=='expected ordinary bounded candidate source':
                    message='宿主签收时未找到符合要求的候选源文件'
                lines.append(f"- {trial['name']} 未进入试跑，{message}。")
        if row['notes']:
            first = row['notes'][0]
            t = fmt(first['first_observed'])
            noted = first['event'].get('time') if isinstance(first['event'],dict) else None
            supplied = '作者标注 '+fmt(noted)+'，' if isinstance(noted,(float,int)) else ''
            lines.append(f"- 收录 {len(row['notes'])} 条作者公开笔记。首条笔记{ supplied}宿主首次读取于 {t}。")
    lines.extend(['','token 采用签收时该作者线程最新的累计计数，包含缓存输入。相邻签收的差额计入该版；计数按模型响应更新，可能与文件写入时刻略有错位。定稿后的作者说明不计入预算。原始公开笔记、签收时间及逐房间得失见[开发账本](development_usage.json)。',''])
    return '\n'.join(lines)


def export():
    doc = DOC/'development_usage.json'
    old = read(doc) if doc.exists() else {'arms':{}}
    parent = read(OUT/'conformance/result.json')['rooms']
    arms = {}
    for role in ROLES:
        previous = {r['row_sha256']:r['first_observed'] for r in old['arms'].get(role,{}).get('notes',[])}
        arms[role] = arm_data(role,parent,previous)
    data = dict(schema=1,root=str(OUT.relative_to(ROOT)),updated=time.time(),arms=arms)
    save(doc,data)
    content = markdown(data)
    md = DOC/'development.md'
    extra = '<!-- signed-public-notes -->'
    if md.exists() and extra in md.read_text():
        content += '\n' + extra + md.read_text().split(extra, 1)[1]
    if not md.exists() or md.read_text() != content:
        temp = md.with_suffix('.pending')
        temp.write_text(content)
        temp.replace(md)
    return all(v['status']!='running' for v in arms.values())


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--watch',action='store_true')
    args=cli.parse_args()
    while True:
        finished=export()
        if finished or not args.watch:
            break
        time.sleep(10)
