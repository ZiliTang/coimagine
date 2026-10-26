"""Self-contained HTML training dashboard (no external JS/CDN).

Usage:
    python experiments/option_rl_v45/monitor.py outputs/option_rl_v45/run2
The trainer calls ``write_dashboard`` after every evaluation, so the file is rewritten in place
and a browser refresh shows the newest curves.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# (chart title, [(key, label, colour), ...])
CHARTS = [
    ("Return", [("explore_return", "explore", "#4aa3ff")]),
    ("Success", [("explore_success", "explore", "#4ad991"), ("success_rate", "validation", "#ffb347")]),
    ("Progress (goal fraction)", [("explore_progress", "explore", "#4ad991"),
                                  ("mean_progress", "validation", "#ffb347")]),
    ("Losses", [("policy_loss", "policy", "#c792ea"), ("value_loss", "value", "#4aa3ff")]),
    ("Entropy", [("entropy", "entropy", "#ffd166")]),
    ("Reward components", [("reward_progress", "progress", "#4ad991"),
                           ("reward_distance_cost", "distance cost", "#ff6b6b"),
                           ("reward_time_cost", "time cost", "#ffb347"),
                           ("reward_success_bonus", "success bonus", "#4aa3ff"),
                           ("reward_failure_penalty", "failure penalty", "#c792ea")]),
    ("Iteration cost", [("seconds", "seconds", "#8b93a7"), ("steps", "steps", "#4aa3ff")]),
]

TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body{{margin:0;background:#0b0f14;color:#dfe6f3;font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{padding:18px 22px 6px}}h1{{font-size:18px;margin:0 0 4px}}h1 span{{color:#8b93a7;font-weight:400}}
.wrap{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px;padding:14px 22px 24px}}
.card{{background:#11161e;border:1px solid #1f2734;border-radius:10px;padding:10px 12px}}
.card h2{{font-size:13px;margin:0 0 6px;color:#cdd6e5;font-weight:600}}
.legend{{font-size:11px;color:#8b93a7;margin:2px 0 6px}}
.legend i{{display:inline-block;width:14px;height:2px;vertical-align:middle;margin:0 4px 0 8px}}
canvas{{width:100%;height:190px;display:block}}
</style></head><body>
<header><h1>option RL training <span>{subtitle}</span></h1>
<div class="legend">x = evaluation iteration · y = value · curves are redrawn after every evaluation</div></header>
<div class="wrap" id="wrap"></div>
<script>
const H = {data};
const CHARTS = {charts};
function seriesValues(key){{ return H.map(e => (key in e && e[key] !== null) ? Number(e[key]) : null); }}
function draw(canvas, chart) {{
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * 2, Hh = canvas.height = canvas.clientHeight * 2;
  ctx.clearRect(0,0,W,Hh);
  const pad = 34;
  const all = [];
  chart[1].forEach(s => seriesValues(s[0]).forEach(v => {{ if (v !== null && isFinite(v)) all.push(v); }}));
  if (!all.length) {{ ctx.fillStyle='#46536b'; ctx.font='13px sans-serif';
    ctx.fillText('no data yet', pad, Hh/2); return; }}
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 1e-9) {{ hi += 1; lo -= 1; }}
  const xmax = Math.max(1, H.length - 1);
  ctx.strokeStyle = '#1f2734'; ctx.lineWidth = 2;
  for (let g = 0; g <= 4; g++) {{
    const y = pad + (Hh - 2*pad) * g / 4;
    ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - 8, y); ctx.stroke();
    ctx.fillStyle = '#46536b'; ctx.font = '11px sans-serif';
    ctx.fillText((hi - (hi-lo)*g/4).toFixed(2), 2, y + 4);
  }}
  chart[1].forEach(([key, label, colour]) => {{
    const vals = seriesValues(key);
    ctx.strokeStyle = colour; ctx.lineWidth = 2.5; ctx.beginPath();
    let started = false;
    vals.forEach((v, i) => {{
      if (v === null || !isFinite(v)) return;
      const x = pad + (W - pad - 8) * (xmax ? i / xmax : 0);
      const y = pad + (Hh - 2*pad) * (1 - (v - lo) / (hi - lo));
      if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }});
}}
const wrap = document.getElementById('wrap');
CHARTS.forEach(chart => {{
  const card = document.createElement('div'); card.className = 'card';
  const h = document.createElement('h2'); h.textContent = chart[0]; card.appendChild(h);
  const legend = document.createElement('div'); legend.className = 'legend';
  chart[1].forEach(([key, label, colour]) => {{
    const icon = document.createElement('i'); icon.style.background = colour;
    legend.appendChild(icon); legend.appendChild(document.createTextNode(label));
  }});
  card.appendChild(legend);
  const canvas = document.createElement('canvas'); card.appendChild(canvas);
  wrap.appendChild(card);
  requestAnimationFrame(() => draw(canvas, chart));
}});
</script></body></html>
"""


def write_dashboard(history, path, title: str = "option RL", subtitle: str = "") -> Path:
    data = json.dumps(history)
    charts = json.dumps(CHARTS)
    path = Path(path)
    path.write_text(TEMPLATE.format(title=title, subtitle=subtitle, data=data, charts=charts))
    return path


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/option_rl_v45/run2")
    history_path = out / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    path = write_dashboard(history, out / "dashboard.html", subtitle=str(out))
    print(path)


if __name__ == "__main__":
    main()
