"""Static, self-refreshing progress page for the inverse-sampling dose study."""
from datetime import datetime
from html import escape
import json
from pathlib import Path
import time

OUT=Path(__file__).resolve().parents[1]/'outputs/semantic_rl/cap_inverse_scale_20261119_01'
DOSES=(250,500,750,1001)


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def render():
    s=read(OUT/'collection_state.json')
    e=read(OUT/'eval_state.json')
    error=s.get('error') or e.get('error') or read(OUT/'pipeline_error.json').get('error')
    done=(OUT/'FINISHED.json').exists()
    cards=[('不同成功序列',f"{s.get('distinct',0):,} / 1,001"),('新采集执行',f"{s.get('runs',0):,}"),
           ('成功轨迹',f"{s.get('success',0):,}"),('覆盖基础场景',f"{s.get('families',0)} / 200")]
    cells=''.join(f'<article><small>{k}</small><strong>{v}</strong></article>' for k,v in cards)
    status='完成' if done else '执行异常' if error else '冻结模型评估' if e else '离线拟合' if s.get('complete') else '开发采集进行中'
    rows=[]
    for d in (0,*DOSES):
        prefix=read(OUT/'doses'/f'{d}.json')
        m=OUT/f'fit_{d}'/'models/GA_s17'
        p=read(m/'progress.json')
        steps=p.get('updates',0)+(16000 if p.get('stage')=='Q_init' else 0)
        trained=d==0 or (m/'offline_complete.json').exists()
        train='原始冻结模型' if d==0 else '24,000 / 24,000' if trained else f'{steps:,} / 24,000' if p else '准备训练数据' if prefix else '等候采集'
        wins=prefix.get('distinct','—') if d else 0
        total=4256+prefix['runs'] if prefix else '—' if d else 4256
        rows.append(f'<tr><td>{d:,}</td><td>{wins}</td><td>{total}</td><td>{train}</td></tr>')
    result='<p><a href="curve.png">查看闭环规模曲线</a> · <a href="results.csv">下载各档结果</a></p><img src="curve.png" alt="闭环成功率随新增数据量的变化">' if done else ''
    message='<pre>'+escape(error or '')+'</pre>' if error else ''
    title='反比采样 · 离线 Q 数据规模曲线'
    doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="8"><title>{title}</title><style>
    *{{box-sizing:border-box}}body{{background:#f1f4ee;color:#18392f;font:16px/1.6 "Instrument Sans","Noto Sans SC",sans-serif;margin:0}}main{{max-width:950px;margin:5vh auto;padding:24px}}h1{{font-size:clamp(28px,4vw,46px);line-height:1.15}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin:18px 0}}article,section{{padding:18px;border:1px solid #cdd9d1;border-radius:12px;background:white}}strong{{display:block;font-size:26px;color:#126853}}progress{{width:100%;accent-color:#22725a}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;text-align:left;border-bottom:1px solid #dde4df}}section{{overflow:auto}}small{{color:#48675d}}pre{{white-space:pre-wrap;color:#982e20}}img{{width:100%}}</style>
    <main><h1>{title}</h1><small>{status} · {datetime.now().strftime('%H:%M:%S')} · 每 8 秒刷新</small><div class="grid">{cells}</div>
    <progress value="{s.get('distinct',0)}" max="1001"></progress><section><table><tr><th>目标剂量</th><th>实际不同成功序列</th><th>总训练轨迹</th><th>拟合更新</th></tr>{''.join(rows)}</table>
    <p>新面板闭环执行 {e.get('completed',0):,} / 7,200 · 同一批 600 个新初态</p>{message}{result}</section></main></html>'''
    dest=OUT/'live/index.html';tmp=dest.with_suffix('.tmp')
    tmp.write_text(doc);tmp.replace(dest)
    return done


if __name__=='__main__':
    while not render():
        time.sleep(5)
