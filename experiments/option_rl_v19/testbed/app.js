const S = {
  options: [], params: [],
  scene: null, baseScene: null, currentPreset: null, loadedInitial: null, dirty: false,
  object: null, robots: [],
  sequence: [], overrides: {}, presets: [], presetSort: 'created', controllerVersion: 'option-v2',
  run: null, frame: 0, playing: false, timer: null, runId: null,
  running: false,
  view: {lidar:false, target:false, force:false, contact:false, estimate:true, yield:true},
  curves: {force:true, speed:true},
};
const $ = (id) => document.getElementById(id);
const STAGE_COLORS = {approach:'#3d6fd6', ready:'#f2c94c', push:'#27ae60', recover:'#eb5757',
  brake:'#8c98a8', move:'#9b51e0', align:'#00b8a9', follow:'#2d9cdb', blocked:'#56616f', initial:'#6b7280',
  done:'#ef7d3b', yield_idle:'#38b2ac', yield_escape:'#00e5ff', yield_blocked:'#ff477e', find:'#ffb74d'};
const PHASE_COLORS = {IDLE:'#56616f', APPROACH:'#1f4ea8', READY:'#f2c94c', ENGAGE:'#ef7d3b',
  PUSH:'#27ae60', RECOVER:'#eb5757', EXIT:'#2b3440'};
const CONTACT_COLOR = '#3fa45b', ACTIVE_COLOR = '#e0b84a';
const physicalContact = (st) => st?.physical_contact || st?.contact || [false, false];

async function api(path, method = 'GET', body = null) {
  const opts = {method, headers: {}};
  if (body !== null) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  const r = await fetch(path, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function setStatus(text, bad = false) {
  const el = $('status'); el.textContent = text; el.style.color = bad ? '#d88989' : '';
}

function optionLabel(i) {
  const o = S.options.find(o => o.action_index === Number(i));
  return o ? o.label : `#${i}`;
}

function clonePose(p) { return {x: p.x, y: p.y, angle: p.angle || 0}; }

function setPresetLabel() {
  const el = $('presetLabel');
  if (!S.currentPreset) el.textContent = S.dirty ? 'Unsaved changes' : 'Unsaved';
  else el.textContent = `Preset: ${S.currentPreset.name}${S.dirty ? ' · modified' : ''}`;
}
function markDirty() { S.dirty = true; setPresetLabel(); }
function defaultPresetName() {
  return `${S.baseScene || 'scene'}_${stampNow()}`;
}
function copySegment(s) {
  // Historical preset/run IDs are aliases, not additional current speed choices.
  const alias = {38:40, 39:40, 41:43, 42:43};
  const canonical = i => alias[Number(i)] ?? Number(i);
  const out = {a0:canonical(s.a0), a1:canonical(s.a1), decisions:Number(s.decisions || 1)};
  if (s.budget_m != null) out.budget_m = Number(s.budget_m);
  if (s.time_budget_s != null) out.time_budget_s = Number(s.time_budget_s);
  if (s.progress) out.progress = String(s.progress);
  return out;
}
function applyPreset(p) {
  S.currentPreset = p;
  S.baseScene = p.base_scene || p.scene || 'contact_2';
  S.scene = p.scene_meta || S.scene;
  S.object = clonePose(p.object || S.scene.initial.object);
  S.robots = (p.robots || S.scene.initial.robots).map(clonePose);
  S.loadedInitial = {object: clonePose(S.object), robots: S.robots.map(clonePose)};
  S.sequence = (p.sequence || []).map(copySegment);
  S.controllerVersion = 'option-v2';
  if ($('controllerVersion')) $('controllerVersion').value = S.controllerVersion;
  S.overrides = p.overrides || {};
  S.run = null; S.frame = 0; S.runId = null;
  S.dirty = false;
  $('presetName').value = p.name || defaultPresetName();
  renderSequence(); renderParams(); drawScene(); drawTimeline(); renderInfo(); updatePlayback();
  $('runMeta').textContent = '';
  setPresetLabel();
}
function resetPoses() {
  if (!S.loadedInitial) return;
  S.object = clonePose(S.loadedInitial.object);
  S.robots = S.loadedInitial.robots.map(clonePose);
  S.run = null; S.frame = 0; S.runId = null;
  drawScene(); drawTimeline(); renderInfo(); updatePlayback();
  markDirty();
}
function initOptions() {
  for (const id of ['option0', 'option1']) {
    const sel = $(id); sel.innerHTML = '';
    for (const o of S.options) {
      const opt = document.createElement('option');
      opt.value = o.action_index;
      opt.textContent = `${String(o.action_index).padStart(2,'0')}  ${o.label}`;
      sel.appendChild(opt);
    }
    sel.value = '2';
  }
}

function renderParams() {
  const box = $('paramList'); box.innerHTML = '';
  const fams = {};
  for (const p of S.params) (fams[p.family] ||= []).push(p);
  for (const fam of Object.keys(fams)) {
    const h = document.createElement('div'); h.className = 'param-family'; h.textContent = fam;
    box.appendChild(h);
    for (const p of fams[fam]) {
      const row = document.createElement('div'); row.className = 'param-row';
      const name = document.createElement('span'); name.textContent = p.name; name.title = p.name;
      const inp = document.createElement('input'); inp.type = 'number';
      inp.min = p.low; inp.max = p.high; inp.step = 'any';
      const cur = (S.overrides[p.family] || {})[p.name];
      inp.value = cur !== undefined ? cur : p.default;
      inp.dataset.family = p.family; inp.dataset.name = p.name;
      inp.addEventListener('change', () => {
        const val = parseFloat(inp.value);
        if (!Number.isFinite(val)) return;
        (S.overrides[p.family] ||= {})[p.name] = val; markDirty();
      });
      const unit = document.createElement('em'); unit.textContent = p.unit || '';
      row.append(name, inp, unit); box.appendChild(row);
    }
  }
}

function collectOverrides() {
  const out = {};
  for (const inp of $('paramList').querySelectorAll('input[data-name]')) {
    const v = parseFloat(inp.value);
    if (!Number.isFinite(v)) continue;
    (out[inp.dataset.family] ||= {})[inp.dataset.name] = v;
  }
  S.overrides = out;
  return out;
}

function renderSequence() {
  const box = $('sequenceList'); box.innerHTML = '';
  let t = 0;
  S.sequence.forEach((seg, i) => {
    const row = document.createElement('div'); row.className = 'segment';
    const idx = document.createElement('span'); idx.className = 'idx'; idx.textContent = `${i+1}`;
    const a = document.createElement('span'); a.className = 'opt'; a.textContent = optionLabel(seg.a0); a.title = a.textContent;
    const b = document.createElement('span'); b.className = 'opt'; b.textContent = optionLabel(seg.a1); b.title = b.textContent;
    const d = document.createElement('span'); d.className = 'mono';
    d.textContent = seg.budget_m != null
      ? `${Number(seg.budget_m).toFixed(2)}m${seg.time_budget_s != null ? ` / ${Number(seg.time_budget_s).toFixed(1)}s` : ''}`
      : `${(seg.decisions*0.2).toFixed(1)}s`;
    const rm = document.createElement('button'); rm.textContent = '×'; rm.title = 'remove'; rm.onclick = () => { S.sequence.splice(i,1); renderSequence(); };
    row.append(idx, a, b, d, rm); box.appendChild(row);
    t += (seg.budget_m == null ? seg.decisions * 0.2 : (seg.time_budget_s || 0));
  });
  $('seqMeta').textContent = S.sequence.length ? `${S.sequence.length} segments · ${(S.sequence.reduce((x,s)=>x+s.decisions,0)*0.2).toFixed(1)} s` : 'empty';
}

function addSegment() {
  const a0 = parseInt($('option0').value, 10);
  const a1 = parseInt($('option1').value, 10);
  const sec = Math.max(0.2, Math.min(10, parseFloat($('duration').value) || 1));
  const decisions = Math.max(1, Math.round(sec / 0.2));
  S.sequence.push({a0, a1, decisions});
  renderSequence(); markDirty();
}

async function refreshControllerStatus() {
  try {
    const st = await api('/api/status');
    const runtime = st.runtime_source_sha256 || st.controller_source_sha256;
    $('serverVersion').textContent = `q ${st.controller_source_sha256.slice(0, 8)} / sim ${runtime.slice(0, 8)}${st.support_memory_enabled === false ? ' · 无轮廓memory' : ''}${st.restart_required ? ' · 需要重启' : ' · 已加载'}`;
    $('serverVersion').title = `PID ${st.pid} · controller ${st.controller_source_sha256} · runtime ${runtime}`;
    return st;
  } catch (e) {
    $('serverVersion').textContent = '版本未确认';
    return null;
  }
}

async function runSequence() {
  if (S.running) return;
  if (!S.scene || !S.sequence.length) { setStatus('add at least one segment', true); return; }
  const server = await refreshControllerStatus();
  if (S.running) return;
  if (!server || server.restart_required) { setStatus('请先确认测台已重启并加载当前控制器', true); return; }
  collectOverrides();
  S.running = true;
  $('runBtn').disabled = true;
  setStatus('running…');
  const body = {
    scene: S.baseScene, name: S.currentPreset?.name || 'session',
    object: S.object, robots: S.robots,
    sequence: S.sequence, overrides: S.overrides, goal: S.scene?.goal, controller_version: 'option-v2',
  };
  const requestStarted = performance.now();
  api('/api/run', 'POST', body).then(run => {
    const apiSeconds = (performance.now()-requestStarted)/1000;
    const drawingStarted = performance.now();
    loadRunObject(run);
    const drawMs = performance.now()-drawingStarted;
    $('runMeta').textContent += ` · 接口 ${apiSeconds.toFixed(3)}s · 绘制 ${drawMs.toFixed(1)}ms`;
    setStatus(`run ${run.id} · ${run.success ? 'task success' : 'task incomplete'}`);
    refreshRuns();
  }).catch(e => setStatus(e.message, true)).finally(() => {
    S.running = false;
    $('runBtn').disabled = false;
  });
}

function loadRunObject(run) {
  S.run = run; S.runId = run.id; S.frame = 0;
  $('frameSlider').max = String(Math.max(0, run.frames.length - 1));
  $('frameSlider').value = '0';
  $('runMeta').textContent = `${run.scene} · ${run.controller_version || 'legacy'} · ${run.success ? 'TASK SUCCESS' : 'TASK INCOMPLETE'} · ${run.periods} ticks · seq ${run.sequence.length}`;
  $('runMeta').textContent += ` · ${run.action_version || '历史动作速度（重跑使用当前挡位）'}`;
  if (run.controller_source_sha256) $('runMeta').textContent += ` · ${run.controller_source_sha256.slice(0, 8)}`;
  if (run.support_memory_enabled === false) $('runMeta').textContent += ' · 仅当前可见面';
  else if (run.frames.some(f => f.memory && f.memory.length)) $('runMeta').textContent += ' · 历史记录：含轮廓memory';
  if (run.timing) $('runMeta').textContent += ` · 仿真/记录 ${run.timing.execution_and_recording_seconds.toFixed(3)}s · 回放整理 ${run.timing.frame_build_seconds.toFixed(3)}s`;
  drawScene(); drawTimeline(); renderInfo(); updatePlayback();
}

function currentFrame() {
  if (!S.run) return null;
  return S.run.frames[Math.max(0, Math.min(S.frame, S.run.frames.length - 1))];
}

function currentStrip() {
  if (!S.run) return null;
  return S.run.strip[Math.max(0, Math.min(S.frame, S.run.strip.length - 1))];
}

let hoverFrame = -1;

function updatePlayback() {
  const n = S.run ? S.run.frames.length : 0;
  const t = S.run ? (S.frame * 0.05) : 0;
  $('timeLabel').textContent = `${t.toFixed(2)} s`;
  $('frameSlider').value = String(S.run ? S.frame : 0);
  $('playBtn').textContent = S.playing ? 'Pause' : 'Play';
  $('playBtn').disabled = !S.run;
  $('prevBtn').disabled = !S.run;
  $('nextBtn').disabled = !S.run;
  $('frameSlider').disabled = !S.run;
}

function togglePlay() {
  if (!S.run) return;
  S.playing = !S.playing;
  if (S.playing) {
    if (S.timer) clearInterval(S.timer);
    S.timer = setInterval(() => {
      if (!S.run) { S.playing = false; clearInterval(S.timer); S.timer = null; return; }
      if (S.frame >= S.run.frames.length - 1) {
        S.playing = false; clearInterval(S.timer); S.timer = null; updatePlayback(); return;
      }
      S.frame++; drawScene(); drawTimeline(); renderInfo(); updatePlayback();
    }, 50);
  } else if (S.timer) { clearInterval(S.timer); S.timer = null; }
  updatePlayback();
}

function seek(frame) {
  if (!S.run) return;
  S.frame = Math.max(0, Math.min(S.run.frames.length - 1, frame));
  drawScene(); drawTimeline(); renderInfo(); updatePlayback();
}

// ---------------------------------------------------------------------------
// scene canvas
// ---------------------------------------------------------------------------
const sceneCanvas = $('sceneCanvas');
const sctx = sceneCanvas.getContext('2d');
let view = null;

function computeView() {
  const W = sceneCanvas.width, H = sceneCanvas.height;
  const arena = S.scene ? S.scene.arena : [5, 4];
  const margin = 36;
  const scale = Math.min((W - 2*margin) / arena[0], (H - 2*margin) / arena[1]);
  const ox = (W - arena[0]*scale) / 2;
  const oy = (H - arena[1]*scale) / 2;
  view = {scale, ox, oy, arena, W, H};
}
function w2c(x, y) { return [view.ox + x*view.scale, view.H - (view.oy + y*view.scale)]; }
function c2w(px, py) {
  const rect = sceneCanvas.getBoundingClientRect();
  const sx = sceneCanvas.width / rect.width, sy = sceneCanvas.height / rect.height;
  const x = (px - rect.left) * sx, y = (py - rect.top) * sy;
  return [(x - view.ox) / view.scale, (view.H - y - view.oy) / view.scale];
}

function drawScene() {
  if (!S.scene) return;
  computeView();
  const ctx = sctx, W = view.W, H = view.H;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0d12'; ctx.fillRect(0, 0, W, H);
  // arena
  const [ax, ay] = w2c(0, view.arena[1]);
  const aw = view.arena[0]*view.scale, ah = view.arena[1]*view.scale;
  ctx.fillStyle = '#10161d'; ctx.fillRect(ax, ay, aw, ah);
  ctx.strokeStyle = '#2b3440'; ctx.lineWidth = 1;
  for (let x = 0; x <= view.arena[0]; x += 1) {
    const [px] = w2c(x, 0); ctx.beginPath(); ctx.moveTo(px, ay); ctx.lineTo(px, ay+ah); ctx.stroke();
  }
  for (let y = 0; y <= view.arena[1]; y += 1) {
    const [, py] = w2c(0, y); ctx.beginPath(); ctx.moveTo(ax, py); ctx.lineTo(ax+aw, py); ctx.stroke();
  }
  ctx.strokeStyle = '#3a4655'; ctx.lineWidth = 2; ctx.strokeRect(ax, ay, aw, ah);
  // goal (amber dashed region; draggable)
  const g = S.scene.goal; const [gx, gy] = w2c(g.x, g.y);
  ctx.beginPath(); ctx.arc(gx, gy, g.radius*view.scale, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(242,201,76,.13)'; ctx.fill();
  ctx.setLineDash([6, 4]); ctx.strokeStyle = '#f2c94c'; ctx.lineWidth = 2; ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f2c94c'; ctx.font = '10px ui-monospace, monospace';
  ctx.fillText('GOAL', gx - 13, gy - g.radius*view.scale - 6);
  // bodies: run frame or editable state
  const f = currentFrame();
  const obj = f ? {x:f.object[0], y:f.object[1], angle:f.object[2]} : S.object;
  const robots = f ? f.robots.map(r => ({x:r[0], y:r[1], angle:r[2]})) : S.robots;
  // object
  const [ox, oy] = w2c(obj.x, obj.y);
  ctx.save(); ctx.translate(ox, oy); ctx.rotate(-obj.angle);
  const poly = S.scene && S.scene.object_polygon;
  if (Array.isArray(poly) && poly.length >= 3) {
    ctx.beginPath();
    poly.forEach((p,k)=>{ const x=p[0]*view.scale, y=-p[1]*view.scale; if(k)ctx.lineTo(x,y); else ctx.moveTo(x,y); });
    ctx.closePath();
    ctx.fillStyle = '#c98b4b'; ctx.strokeStyle = '#f0c38e'; ctx.lineWidth = 1.5;
    ctx.fill(); ctx.stroke();
  } else {
    const ow = 1.0*view.scale, oh = 0.25*view.scale;
    ctx.fillStyle = '#c98b4b'; ctx.strokeStyle = '#f0c38e'; ctx.lineWidth = 1.5;
    ctx.fillRect(-ow/2, -oh/2, ow, oh); ctx.strokeRect(-ow/2, -oh/2, ow, oh);
  }
  ctx.restore();
  // robots
  robots.forEach((r, i) => {
    const [rx, ry] = w2c(r.x, r.y);
    const rad = 0.18*view.scale;
    ctx.beginPath(); ctx.arc(rx, ry, rad, 0, Math.PI*2);
    ctx.fillStyle = i === 0 ? 'rgba(91,127,184,.85)' : 'rgba(63,164,91,.85)'; ctx.fill();
    ctx.strokeStyle = i === 0 ? '#9bb8e0' : '#86d3a0'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(rx, ry);
    ctx.lineTo(rx + Math.cos(r.angle)*rad*1.35, ry - Math.sin(r.angle)*rad*1.35);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#dbe5f1'; ctx.font = '11px ui-monospace, monospace';
    ctx.fillText(`R${i}`, rx - 7, ry - rad - 6);
  });
  drawOverlays(ctx, f, robots);
  // label
  ctx.fillStyle = '#8c98a8'; ctx.font = '11px ui-sans-serif, system-ui';
  ctx.fillText(`${S.currentPreset?.name || 'Unsaved'}  ·  ${f ? `frame ${S.frame}/${S.run.frames.length-1}` : 'editable initial state'}`, 12, 20);
  drawEstimate(f);
}

function drawEstimate(f) {
  const win = $('estimateWindow'); if (!win) return;
  win.style.display = S.view.estimate ? 'block' : 'none';
  if (!S.view.estimate) return;
  const cv = $('estimateCanvas'), ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#0a0d12'; ctx.fillRect(0,0,W,H);
  const obj = f ? {x:f.object[0], y:f.object[1], angle:f.object[2]} : S.object;
  const robots = f ? f.robots.map(r=>({x:r[0],y:r[1],angle:r[2]})) : S.robots;
  const zoom = 135; // px per metre
  const toC=(x,y)=>[W/2+(x-obj.x)*zoom, H/2-(y-obj.y)*zoom];
  // background crosshair/grid around object
  ctx.strokeStyle='#202832'; ctx.lineWidth=1;
  for(let k=-1;k<=1;k++){ const gx=W/2+k*zoom*.5; ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,H);ctx.stroke(); const gy=H/2+k*zoom*.5; ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke(); }
  ctx.beginPath(); ctx.arc(W/2,H/2,3,0,Math.PI*2); ctx.fillStyle='#414c5b'; ctx.fill();
  // actual object faint reference (known in the testbed only)
  ctx.save(); ctx.translate(...toC(obj.x,obj.y)); ctx.rotate(-obj.angle);
  ctx.fillStyle='rgba(201,139,75,.20)'; ctx.strokeStyle='rgba(240,195,142,.55)'; ctx.lineWidth=1.2;
  const spoly = S.scene && S.scene.object_polygon;
  if (Array.isArray(spoly) && spoly.length >= 3) {
    ctx.beginPath(); spoly.forEach((p,k)=>{ const x=p[0]*zoom,y=-p[1]*zoom; if(k)ctx.lineTo(x,y); else ctx.moveTo(x,y); }); ctx.closePath(); ctx.fill(); ctx.stroke();
  } else { ctx.fillRect(-.5*zoom,-.125*zoom,1.0*zoom,.25*zoom); ctx.strokeRect(-.5*zoom,-.125*zoom,1.0*zoom,.25*zoom); }
  ctx.restore();
  // Compatibility display for archived runs only. New runs have no memory field.
  if (f && f.memory) f.memory.forEach((seg, k) => {
    const n=seg.n,t=seg.t,lo=seg.lo,hi=seg.hi;
    const a=[n[0]*seg.rho+t[0]*lo, n[1]*seg.rho+t[1]*lo];
    const b=[n[0]*seg.rho+t[0]*hi, n[1]*seg.rho+t[1]*hi];
    const [ax,ay]=toC(a[0],a[1]), [bx,by]=toC(b[0],b[1]);
    ctx.strokeStyle='rgba(220,231,243,.82)'; ctx.lineWidth=4.5; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(ax,ay); ctx.lineTo(bx,by); ctx.stroke();
    ctx.fillStyle='rgba(220,231,243,.7)'; ctx.font='9px ui-monospace,monospace';
  });
  const mbs = f && f.mb ? f.mb : [null,null];
  mbs.forEach((m,side)=>{
    if(!m) return;
    const col = side===0 ? '#f0b86e' : '#7fd1b9';
    if(m.face_points && m.face_points.length>1){
      const pts=m.face_points.map(p=>toC(p[0],p[1]));
      ctx.strokeStyle=col; ctx.lineWidth=2.5; ctx.setLineDash([5,4]);
      ctx.beginPath(); pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1])); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle=col; ctx.font='10px ui-monospace,monospace'; 
    }
    if(m.q_prime){ const [px,py]=toC(m.q_prime[0],m.q_prime[1]); ctx.strokeStyle='#63d2ff'; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(px-5,py-5);ctx.lineTo(px+5,py+5);ctx.moveTo(px+5,py-5);ctx.lineTo(px-5,py+5);ctx.stroke(); }
    if(m.candidate && m.candidate.q){ const [qx,qy]=toC(m.candidate.q[0],m.candidate.q[1]); ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();ctx.arc(qx,qy,5,0,Math.PI*2);ctx.stroke();ctx.fillStyle=col;ctx.font='10px ui-monospace,monospace'; }
    if(m.v_est && m.v_est.length>=3){
      const p=(m.candidate&&m.candidate.q)?m.candidate.q:m.own;
      if(p){ const [ax,ay]=toC(p[0],p[1]); const vx=m.v_est[0],vy=m.v_est[1],om=m.v_est[2]; const ex=vx-om*p[1],ey=vy+om*p[0]; const sc=42; drawArrow(ctx,ax,ay,ax+ex*sc,ay-ey*sc,'#63d2ff',2); ctx.fillStyle='#63d2ff';ctx.font='10px ui-monospace,monospace'; }
    }
  });
  robots.forEach((r,side)=>{ const [rx,ry]=toC(r.x,r.y); const rad=.18*zoom; ctx.beginPath();ctx.arc(rx,ry,rad,0,Math.PI*2);ctx.fillStyle=side===0?'rgba(91,127,184,.30)':'rgba(63,164,91,.30)';ctx.fill();ctx.strokeStyle=side===0?'rgba(155,184,224,.55)':'rgba(134,211,160,.55)';ctx.lineWidth=1.2;ctx.stroke();ctx.beginPath();ctx.moveTo(rx,ry);ctx.lineTo(rx+Math.cos(r.angle)*rad*1.25,ry-Math.sin(r.angle)*rad*1.25);ctx.strokeStyle='#fff';ctx.lineWidth=1.8;ctx.stroke();ctx.fillStyle='rgba(219,229,241,.65)';ctx.font='10px ui-monospace,monospace'; });
  if (S.scene && S.scene.goal) { const g=S.scene.goal; const [gx,gy]=toC(g.x,g.y); ctx.beginPath();ctx.arc(gx,gy,g.radius*zoom,0,Math.PI*2);ctx.fillStyle='rgba(242,201,76,.10)';ctx.fill();ctx.setLineDash([5,4]);ctx.strokeStyle='#f2c94c';ctx.lineWidth=1.3;ctx.stroke();ctx.setLineDash([]);ctx.fillStyle='#f2c94c';ctx.font='9px ui-monospace,monospace'; }
}

function drawArrow(ctx, x0, y0, x1, y1, color, width = 1.5) {
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  const a = Math.atan2(y1 - y0, x1 - x0), s = 6;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - s*Math.cos(a - .4), y1 - s*Math.sin(a - .4));
  ctx.lineTo(x1 - s*Math.cos(a + .4), y1 - s*Math.sin(a + .4));
  ctx.closePath(); ctx.fill();
}

function drawOverlays(ctx, f, robots) {
  if (!S.run) return;
  const st = currentStrip();
  // force arrows
  if (S.view.force && st && st.forces) {
    robots.forEach((r, side) => {
      const fv = st.forces[side]; if (!fv) return;
      const [rx, ry] = w2c(r.x, r.y);
      const mag = Math.hypot(fv[0], fv[1]);
      if (mag < 0.01) return;
      const scale = 22;
      drawArrow(ctx, rx, ry, rx + fv[0]*scale, ry - fv[1]*scale,
        side === 0 ? '#8db4ef' : '#7fe0a0', 1.6);
    });
  }
  // contact rings
  if (S.view.contact && st) {
    robots.forEach((r, side) => {
      if (!physicalContact(st)[side]) return;
      const [rx, ry] = w2c(r.x, r.y);
      ctx.beginPath(); ctx.arc(rx, ry, .18*view.scale + 4, 0, Math.PI*2);
      ctx.strokeStyle = '#66e39a'; ctx.lineWidth = 1.5; ctx.stroke();
    });
  }
  // lidar rays
  if (S.view.lidar && f && f.scan) {
    const colors = {1:'#7fe0a0', 2:'#73818f', 3:'#8db4ef'};
    f.scan.forEach((scan, side) => {
      if (!scan || !scan.valid) return;
      const [ox, oy] = w2c(scan.origin[0], scan.origin[1]);
      for (let i = 0; i < scan.valid.length; i++) {
        if (!scan.valid[i]) continue;
        const a = scan.angles[i], d = scan.distances[i];
        const [ex, ey] = w2c(scan.origin[0] + d*Math.cos(a), scan.origin[1] + d*Math.sin(a));
        ctx.strokeStyle = colors[scan.category?.[i]] || '#73818f';
        ctx.globalAlpha = .28; ctx.lineWidth = .8;
        ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.globalAlpha = 1;
      }
    });
  }
  // reference target / object centre
  if (S.view.target && f && f.target) {
    f.target.forEach((t, side) => {
      if (!t) return;
      const r = robots[side]; const [tx, ty] = w2c(t[0], t[1]); const [rx, ry] = w2c(r.x, r.y);
      ctx.setLineDash([4, 4]); ctx.strokeStyle = side === 0 ? '#9bb8e0' : '#9be3b7';
      ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(tx, ty, 6, 0, Math.PI*2);
      ctx.strokeStyle = side === 0 ? '#9bb8e0' : '#9be3b7'; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx-9, ty); ctx.lineTo(tx+9, ty); ctx.moveTo(tx, ty-9); ctx.lineTo(tx, ty+9); ctx.stroke();
      ctx.fillStyle = '#dbe5f1'; ctx.font = '10px ui-monospace, monospace';
    });
    if (f.ref_centre) f.ref_centre.forEach((t, side) => {
      if (!t) return;
      const [tx, ty] = w2c(t[0], t[1]);
      ctx.fillStyle = side === 0 ? '#5b7fb8' : '#3fa45b';
      ctx.beginPath(); ctx.moveTo(tx, ty-4); ctx.lineTo(tx+4, ty); ctx.lineTo(tx, ty+4); ctx.lineTo(tx-4, ty); ctx.closePath(); ctx.fill();
    });
  }
  // m3b debug: persistent candidate q and actual contact point
  if (f && f.mb) f.mb.forEach((m, side) => {
    if (!m) return;
    if (S.view.yield && robots[side]) {
      const r = robots[side]; const [rx, ry] = w2c(r.x, r.y);
      if (m.requested_direction) {
        const d = m.requested_direction;
        const [ex, ey] = w2c(r.x+.40*d[0], r.y+.40*d[1]);
        drawArrow(ctx, rx, ry, ex, ey, m.direction_valid ? '#ffce54' : '#ff477e', 2);
      }
      const y = m.yield_debug;
      if (y) {
        for (const candidate of y.candidates || []) {
          const [ex, ey] = w2c(r.x+candidate.v[0]*.6, r.y+candidate.v[1]*.6);
          ctx.strokeStyle = candidate.feasible ? '#387f7d' : '#733848'; ctx.lineWidth = .7;
          ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(ex, ey); ctx.stroke();
        }
        if (y.chosen_velocity) {
          const [ex, ey] = w2c(r.x+y.chosen_velocity[0], r.y+y.chosen_velocity[1]);
          drawArrow(ctx, rx, ry, ex, ey, '#00e5ff', 3);
        }
        if (y.threat_point) {
          const [px, py] = w2c(...y.threat_point);
          ctx.strokeStyle = '#ff477e'; ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI*2); ctx.stroke();
        }
      }
    }
    const col = side===0 ? '#f0b86e' : '#7fd1b9';
    if (m.q_prime) {
      const [px,py]=w2c(m.q_prime[0],m.q_prime[1]);
      ctx.strokeStyle='#63d2ff'; ctx.lineWidth=2; ctx.beginPath();
      ctx.moveTo(px-5,py-5); ctx.lineTo(px+5,py+5); ctx.moveTo(px+5,py-5); ctx.lineTo(px-5,py+5); ctx.stroke();
      ctx.fillStyle='#63d2ff'; ctx.font='10px ui-monospace,monospace'; ctx.fillText("q'",px+6,py-6);
    }
    if (m.candidate && m.candidate.q) {
      const [qx,qy]=w2c(m.candidate.q[0],m.candidate.q[1]);
      ctx.strokeStyle=col; ctx.fillStyle=col; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.arc(qx,qy,5,0,Math.PI*2); ctx.stroke();
      ctx.fillStyle=col; ctx.font='10px ui-monospace,monospace';
      if (m.own) { const [rx,ry]=w2c(m.own[0],m.own[1]); ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(rx,ry);ctx.lineTo(qx,qy);ctx.stroke();ctx.setLineDash([]); }
    }
    if (m.contact_point_world) {
      const [cx,cy]=w2c(m.contact_point_world[0],m.contact_point_world[1]);
      ctx.strokeStyle='#ff9f43';ctx.lineWidth=2;ctx.beginPath();ctx.arc(cx,cy,4,0,Math.PI*2);ctx.stroke();
    }
  });
}

// drag
let drag = null;
function hitBody(wx, wy) {
  const obj = S.object;
  const dx = wx - obj.x, dy = wy - obj.y;
  const c = Math.cos(-obj.angle), s = Math.sin(-obj.angle);
  const lx = dx*c - dy*s, ly = dx*s + dy*c;
  if (Math.abs(lx) <= 0.55 && Math.abs(ly) <= 0.18) return {type:'object'};
  for (let i = 0; i < S.robots.length; i++) {
    const r = S.robots[i];
    if (Math.hypot(wx - r.x, wy - r.y) <= 0.26) return {type:'robot', index:i};
  }
  if (S.scene && S.scene.goal) {
    const g = S.scene.goal;
    if (Math.hypot(wx - g.x, wy - g.y) <= g.radius + .03) return {type:'goal'};
  }
  return null;
}
sceneCanvas.addEventListener('pointerdown', (e) => {
  const [wx, wy] = c2w(e.clientX, e.clientY);
  const hit = hitBody(wx, wy);
  if (!hit) return;
  const origin = hit.type === 'object' ? S.object
    : hit.type === 'robot' ? S.robots[hit.index]
    : (S.scene && S.scene.goal);
  drag = {...hit, wx, wy, ox: origin.x, oy: origin.y};
  if (S.run) { S.run = null; S.frame = 0; S.runId = null; $('runMeta').textContent = ''; updatePlayback(); }
  sceneCanvas.setPointerCapture(e.pointerId);
  e.preventDefault();
});
sceneCanvas.addEventListener('pointermove', (e) => {
  if (!drag) return;
  const [wx, wy] = c2w(e.clientX, e.clientY);
  const nx = drag.ox + (wx - drag.wx), ny = drag.oy + (wy - drag.wy);
  if (drag.type === 'object') { S.object.x = nx; S.object.y = ny; }
  else if (drag.type === 'goal') { S.scene.goal.x = nx; S.scene.goal.y = ny; }
  else { S.robots[drag.index].x = nx; S.robots[drag.index].y = ny; }
  markDirty(); drawScene();
});
sceneCanvas.addEventListener('pointerup', () => { drag = null; });
sceneCanvas.addEventListener('pointercancel', () => { drag = null; });

// ---------------------------------------------------------------------------
// timeline
// ---------------------------------------------------------------------------
const timelineCanvas = $('timelineCanvas');
const tctx = timelineCanvas.getContext('2d');
function drawTimeline() {
  const hasCurves = S.curves.force || S.curves.speed;
  const H = hasCurves ? 260 : 150;
  if (timelineCanvas.height !== H) timelineCanvas.height = H;
  const ctx = tctx, W = timelineCanvas.width;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0a0d12'; ctx.fillRect(0, 0, W, H);
  const left = 92, right = 18, top = 18, rowH = 24, rows = 4;
  const plotW = W - left - right;
  const n = S.run ? S.run.frames.length : 0;
  ctx.font = '10px ui-monospace, monospace';
  const labels = ['R0 stage', 'R1 stage', 'joint phase', 'contact / active'];
  labels.forEach((lab, i) => {
    ctx.fillStyle = '#8c98a8'; ctx.fillText(lab, 8, top + i*rowH + 18);
  });
  if (!n) {
    ctx.fillStyle = '#3c4654'; ctx.fillText('run a sequence to see the phase strip', left, top + 18);
    return;
  }
  const cw = plotW / n;
  const xOfFrame = i => left + (i + 0.5) * cw;
  for (let i = 0; i < n; i++) {
    const st = S.run.strip[i];
    const x = left + i*cw;
    for (let r = 0; r < 2; r++) {
      ctx.fillStyle = STAGE_COLORS[st.stage[r]] || '#5b6470';
      ctx.fillRect(x, top + r*rowH + 4, Math.max(1, cw), rowH - 8);
    }
    ctx.fillStyle = PHASE_COLORS[st.phase] || '#5b6470';
    ctx.fillRect(x, top + 2*rowH + 4, Math.max(1, cw), rowH - 8);
    const c = physicalContact(st);
    ctx.fillStyle = c[0] ? CONTACT_COLOR : '#2a333d';
    ctx.fillRect(x, top + 3*rowH + 4, Math.max(1, cw/2), rowH - 8);
    ctx.fillStyle = c[1] ? CONTACT_COLOR : '#2a333d';
    ctx.fillRect(x + cw/2, top + 3*rowH + 4, Math.max(1, cw/2), rowH - 8);
    if (st.joint_active) {
      ctx.fillStyle = ACTIVE_COLOR; ctx.fillRect(x, top + 3*rowH + rowH - 5, Math.max(1,cw), 2);
    }
  }
  // curves share the exact same frame->x mapping
  const curveRows = [];
  if (S.curves.force) curveRows.push('force');
  if (S.curves.speed) curveRows.push('speed');
  const curveTop = top + rows*rowH + 6;
  const curveBottom = H - 20;
  const curveH = curveRows.length ? (curveBottom - curveTop) / curveRows.length : 0;
  const series = {
    force: [
      {label:'|F0|', color:'#8db4ef', get:i => Math.hypot(...(S.run.strip[i].forces?.[0] || [0,0]))},
      {label:'|F1|', color:'#7fe0a0', get:i => Math.hypot(...(S.run.strip[i].forces?.[1] || [0,0]))},
    ],
    speed: [
      {label:'|v0|', color:'#8db4ef', get:i => Math.hypot(S.run.frames[i].robots[0][3], S.run.frames[i].robots[0][4])},
      {label:'|v1|', color:'#7fe0a0', get:i => Math.hypot(S.run.frames[i].robots[1][3], S.run.frames[i].robots[1][4])},
      {label:'|v obj|', color:'#c98b4b', get:i => Math.hypot(S.run.frames[i].object[3], S.run.frames[i].object[4])},
    ],
  };
  curveRows.forEach((kind, row) => {
    const y0 = curveTop + row*curveH, h = curveH - 16;
    const data = series[kind];
    let max = 0;
    for (let i = 0; i < n; i++) for (const s of data) max = Math.max(max, s.get(i));
    max = Math.max(max, kind === 'force' ? 0.5 : 0.05);
    ctx.strokeStyle = '#222b36'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, y0+h); ctx.lineTo(W-right, y0+h); ctx.stroke();
    ctx.fillStyle = '#6f7d8d'; ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(kind === 'force' ? 'force N' : 'speed m/s', 6, y0 + 12);
    ctx.fillText(max.toFixed(2), 6, y0 + 24);
    for (const s of data) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = xOfFrame(i), y = y0 + h - (s.get(i)/max)*h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    let lx = left + 4;
    for (const s of data) {
      ctx.fillStyle = s.color; ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(s.label, lx, y0 + 10); lx += 44;
    }
  });
  // one playhead through timeline and curves
  const px = xOfFrame(S.frame);
  ctx.strokeStyle = '#f0f4fa'; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(px, top - 4); ctx.lineTo(px, H - 18); ctx.stroke();
  if (hoverFrame >= 0 && hoverFrame < n) {
    const hx = xOfFrame(hoverFrame);
    ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(hx, top - 4); ctx.lineTo(hx, H - 18); ctx.stroke();
  }
  // one shared time axis
  ctx.fillStyle = '#5c6878'; ctx.font = '10px ui-monospace, monospace';
  for (let i = 0; i < n; i += 20) {
    ctx.fillText(`${(i*0.05).toFixed(1)}s`, xOfFrame(i), H - 5);
  }
}

function timelineFrameFromEvent(e) {
  if (!S.run) return -1;
  const rect = timelineCanvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (timelineCanvas.width / rect.width);
  const left = 92, plotW = timelineCanvas.width - left - 18;
  const n = S.run.frames.length;
  return Math.max(0, Math.min(n - 1, Math.round(((x - left) / plotW) * n - 0.5)));
}

function updateTimelineTip(e, f) {
  const st = S.run.strip[f];
  const fd = S.run.frames[f];
  const coord = (S.run.coordinator || [])[f] || {};
  const tip = $('timelineTip');
  const c = physicalContact(st);
  const fc = st.forces || [[0,0],[0,0]];
  const dot = (k) => `<i style="background:${STAGE_COLORS[k] || '#6b7280'}"></i>`;
  const speed = r => Math.hypot(r[3], r[4]);
  const f0 = Math.hypot(fc[0][0], fc[0][1]), f1 = Math.hypot(fc[1][0], fc[1][1]);
  const v0 = speed(fd.robots[0]), v1 = speed(fd.robots[1]), vo = speed(fd.object);
  tip.innerHTML =
    `<div class="tt-title">frame ${f} · ${(f*0.05).toFixed(2)} s · tick ${st.tick}</div>` +
    `<div class="tt-row"><span>joint phase</span><b>${st.phase || '—'}</b></div>` +
    `<div class="tt-row"><span>R0 stage</span><b>${dot(st.stage?.[0])}${st.stage?.[0] || '—'}</b></div>` +
    `<div class="tt-row"><span>R1 stage</span><b>${dot(st.stage?.[1])}${st.stage?.[1] || '—'}</b></div>` +
    `<div class="tt-row"><span>contact (physical, tick-start)</span><b>${c[0]?'R0 ':''}${c[1]?'R1':''}${!c[0]&&!c[1]?'none':''}</b></div>` +
    `<div class="tt-row"><span>coord both_contact (near)</span><b>${coord.both_contact ? 'yes' : 'no'}</b></div>` +
    `<div class="tt-row"><span>coord release / active</span><b>${coord.release ? 'rel' : '—'} / ${coord.joint_active ? 'active' : '—'}</b></div>` +
    `<div class="tt-row"><span>|F0| / |F1|</span><b>${f0.toFixed(2)} / ${f1.toFixed(2)} N</b></div>` +
    `<div class="tt-row"><span>|v0| / |v1| / |vobj|</span><b>${v0.toFixed(3)} / ${v1.toFixed(3)} / ${vo.toFixed(3)}</b></div>`;
  const wrap = timelineCanvas.parentElement.getBoundingClientRect();
  let left = e.clientX - wrap.left + 14;
  let top = e.clientY - wrap.top + 14;
  tip.style.display = 'block';
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  if (left + tw > wrap.width - 4) left = e.clientX - wrap.left - tw - 14;
  if (top + th > wrap.height - 4) top = e.clientY - wrap.top - th - 14;
  tip.style.left = `${Math.max(4,left)}px`; tip.style.top = `${Math.max(4,top)}px`;
}

let timelineDrag = false;
timelineCanvas.addEventListener('pointerdown', (e) => {
  if (!S.run) return;
  timelineDrag = true;
  timelineCanvas.setPointerCapture(e.pointerId);
  const f = timelineFrameFromEvent(e);
  seek(f); updateTimelineTip(e, f);
});
timelineCanvas.addEventListener('pointermove', (e) => {
  if (!S.run) return;
  const f = timelineFrameFromEvent(e);
  if (timelineDrag) seek(f);
  hoverFrame = f;
  updateTimelineTip(e, f);
  drawTimeline();
});
timelineCanvas.addEventListener('pointerup', () => { timelineDrag = false; });
timelineCanvas.addEventListener('pointercancel', () => { timelineDrag = false; });
timelineCanvas.addEventListener('pointerleave', () => {
  if (timelineDrag) return;
  hoverFrame = -1;
  $('timelineTip').style.display = 'none';
  drawTimeline();
});

function renderLegend() {
  const box = $('legend'); box.innerHTML = '';
  const add = (label, color) => {
    const s = document.createElement('span');
    s.innerHTML = `<i style="background:${color}"></i>${label}`;
    box.appendChild(s);
  };
  ['approach','ready','push','recover','brake','move','align','follow'].forEach(k => add(k, STAGE_COLORS[k]));
  add('joint PUSH', PHASE_COLORS.PUSH); add('joint ENGAGE', PHASE_COLORS.ENGAGE);
  add('contact', CONTACT_COLOR); add('joint_active', ACTIVE_COLOR);
}

// ---------------------------------------------------------------------------
// info / runs
// ---------------------------------------------------------------------------
function renderInfo() {
  const st = currentStrip();
  const f = currentFrame();
  const box = $('stateInfo');
  if (!S.run) {
    box.innerHTML = `<span>preset</span><b>${S.currentPreset?.name || 'Unsaved'}</b>
      <span>description</span><b>${S.currentPreset?.description || '—'}</b>
      <span>object</span><b>${S.object ? `${S.object.x.toFixed(2)}, ${S.object.y.toFixed(2)}` : '—'}</b>
      <span>robots</span><b>${S.robots.map(r => `${r.x.toFixed(2)},${r.y.toFixed(2)}`).join('  ·  ') || '—'}</b>
      <span>sequence</span><b>${S.sequence.length} segments</b>`;
    return;
  }
  const c = physicalContact(st);
  const forces = st?.forces || [[0,0],[0,0]];
  box.innerHTML = `
    <span>time</span><b>${(S.frame*0.05).toFixed(2)} s · frame ${S.frame}</b>
    <span>joint phase</span><b>${st?.phase || '—'}</b>
    <span>R0 stage</span><b>${st?.stage?.[0] || '—'}</b>
    <span>R1 stage</span><b>${st?.stage?.[1] || '—'}</b>
    <span>contact (physical)</span><b>${c[0] ? 'R0' : '—'} / ${c[1] ? 'R1' : '—'}</b>
    <span>joint active (near)</span><b>${st?.joint_active ? 'yes' : 'no'}</b>
    <span>forces R0</span><b>${forces[0] ? `${forces[0][0].toFixed(2)}, ${forces[0][1].toFixed(2)}` : '—'}</b>
    <span>forces R1</span><b>${forces[1] ? `${forces[1][0].toFixed(2)}, ${forces[1][1].toFixed(2)}` : '—'}</b>
    <span>run</span><b>${S.run.success ? 'TASK SUCCESS' : 'TASK INCOMPLETE'} · ${S.run.reward.toFixed(2)}</b>
    <span>object</span><b>${f ? `${f.object[0].toFixed(2)}, ${f.object[1].toFixed(2)}` : '—'}</b>`;
  renderM3bDebug();
}

function renderM3bDebug(){
  const box=$('m3bDebug'); const f=currentFrame();
  if(!f){box.textContent='m3b debug: run a sequence';return;}
  const mb=f.mb||[];
  const fmt=(x,n=2)=>x===null||x===undefined?'null':(+x).toFixed(n);
  const lines=[];
  for(const side of [0,1]){
    const m=mb[side];
    if(!m){lines.push(`R${side}: no m3b record`);continue;}
    const c=m.candidate;
    const q=c&&c.q?`q=(${fmt(c.q[0])},${fmt(c.q[1])}) id${c.id} ${c.reason||''}`:'q=null/not_applicable';
    lines.push(`R${side} ${m.action||''} stage=${m.stage||'—'} phase=${m.contact_phase||'—'} sensor=${physicalContact(currentStrip())[side]?'YES':'no'} ready=${m.contact?'Y':'n'}`);
    lines.push(`  ${q} e_t=${fmt(m.candidate_err_t,3)} cp=${m.contact_point_world?`(${fmt(m.contact_point_world[0])},${fmt(m.contact_point_world[1])})`:'null'}`);
    lines.push(`  F=${m.force?`(${fmt(m.force[0])},${fmt(m.force[1])})`:'null'}  ff_t=${fmt(m.ff_t,3)} u_adj=${fmt(m.u_adjust,3)} c_err_t=${fmt(m.contact_err_t,3)}`);
    lines.push(`  pre=${m.desired_pre?`(${fmt(m.desired_pre[0])},${fmt(m.desired_pre[1])})`:'null'}  desired=${m.desired?`(${fmt(m.desired[0])},${fmt(m.desired[1])})`:'null'}  avoid=${m.avoidance_active?'Y':'n'}`);
    lines.push(`  ref=${m.reference?`(${fmt(m.reference[0])},${fmt(m.reference[1])})`:'null'}  n_ang=${fmt(m.normal_angle,3)}`);
    if (m.requested_direction) lines.push(`  PUSH exec=${m.direction_valid?'matched':'waiting'} request_face=${m.direction_unavailable?'UNAVAILABLE':'available'} dot=${fmt(m.direction_alignment,3)}`);
    const y=m.yield_debug;
    if(y) lines.push(`  YIELD active=${y.active?'Y':'n'} blocked=${y.blocked?'Y':'n'} gap=${fmt(y.min_gap,3)} predicted=${fmt(y.predicted_min_gap,3)} side=${y.side}`);
    const qt = m.selector_debug && m.selector_debug.q_tracking;
    if (qt) lines.push(`  segment=${qt.segment_id || '—'} ${qt.source} selector=${qt.selector_evaluated ? '重新选择' : '保持'} matched=${qt.matched ? 'Y' : 'n'}`);
  }
  box.textContent=lines.join('\n');
}

async function refreshRuns() {
  const started = performance.now();
  try {
    const runs = await api('/api/runs');
    const fetched = performance.now();
    const box = $('runList'); box.innerHTML = '';
    if (!runs.length) { box.innerHTML = '<div class="hint" style="padding:6px">no saved runs yet</div>'; return; }
    for (const r of runs) {
      const row = document.createElement('div');
      row.className = 'run-item' + (r.id === S.runId ? ' active' : '');
      row.innerHTML = `<small>${r.name}<br>${new Date((r.created||0)*1000).toLocaleTimeString()} · ${r.scene}</small>
        <span class="${r.success?'ok':'warn'}">${r.success?'task ok':'task inc.'}</span><span class="mono">${r.periods}</span>`;
      row.onclick = () => loadRun(r.id);
      box.appendChild(row);
    }
    $('historyTiming').textContent = `获取 ${((fetched-started)/1000).toFixed(3)}s / 整理 ${(performance.now()-fetched).toFixed(1)}ms`;
  } catch (e) { setStatus(e.message, true); }
}

async function loadRun(id) {
  try {
    const run = await api(`/api/runs/${id}`);
    if (run.scene_meta) S.scene = run.scene_meta;
    if (run.scene) S.baseScene = run.scene;
    if (run.object) S.object = clonePose(run.object);
    if (run.robots) S.robots = run.robots.map(clonePose);
    S.loadedInitial = {object: clonePose(S.object), robots: S.robots.map(clonePose)};
    if (run.sequence) { S.sequence = run.sequence.map(copySegment); renderSequence(); }
    if (run.overrides) { S.overrides = run.overrides; renderParams(); }
    S.currentPreset = null; S.dirty = true;
    $('presetName').value = defaultPresetName();
    setPresetLabel();
    loadRunObject(run);
    setStatus(`loaded run ${run.id} — save as a new preset if useful`);
    refreshRuns();
  } catch (e) { setStatus(e.message, true); }
}

async function refreshOverrides() {
  try {
    const names = await api('/api/overrides');
    const box = $('overrideList'); box.innerHTML = '';
    if (!names.length) { box.innerHTML = '<div class="hint" style="padding:4px 6px">no saved profiles</div>'; return; }
    for (const name of names) {
      const row = document.createElement('div'); row.className = 'override-item'; row.textContent = name;
      row.onclick = () => loadOverride(name);
      box.appendChild(row);
    }
  } catch (e) { setStatus(e.message, true); }
}

async function loadOverride(name) {
  try {
    const res = await api(`/api/overrides/${encodeURIComponent(name)}`);
    S.overrides = res.overrides || {};
    renderParams();
    markDirty();
    $('overridePath').textContent = `loaded parameter profile: ${name}`;
    setStatus(`loaded override ${name}`);
  } catch (e) { setStatus(e.message, true); }
}

async function saveOverrides() {
  collectOverrides();
  try {
    const name = prompt('Override name', `${S.currentPreset?.name || S.baseScene || 'scene'}_override_${Date.now().toString(36)}`);
    if (!name) return;
    const res = await api('/api/overrides', 'POST', {name, overrides: S.overrides});
    $('overridePath').textContent = `saved: ${res.path}`;
    setStatus('override saved');
    await refreshOverrides();
  } catch (e) { setStatus(e.message, true); }
}

// ---------------------------------------------------------------------------
// presets
// ---------------------------------------------------------------------------
function stampNow() {
  const d = new Date();
  const p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}${p(d.getMonth()+1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}
function renderPresets() {
  const box = $('presetList'); box.innerHTML = '';
  const arr = [...S.presets];
  if (S.presetSort === 'name') arr.sort((a,b) => (a.name||'').localeCompare(b.name||''));
  else arr.sort((a,b) => (b.created||0) - (a.created||0));
  if (!arr.length) { box.innerHTML = '<div class="hint" style="padding:5px">no presets yet</div>'; return; }
  for (const p of arr) {
    const row = document.createElement('div'); row.className = 'preset-item';
    const info = document.createElement('div');
    info.innerHTML = `<div>${p.name || p.id}</div><small>${p.description || '—'} · ${p.sequence_len || 0} seg</small>`;
    const del = document.createElement('button'); del.textContent = '×'; del.title = 'delete preset';
    del.onclick = (e) => { e.stopPropagation(); deletePreset(p.id); };
    row.append(info, del);
    row.onclick = () => loadPreset(p.id);
    box.appendChild(row);
  }
}
async function refreshPresets() {
  try { S.presets = await api('/api/presets'); renderPresets(); }
  catch (e) { setStatus(e.message, true); }
}
async function savePreset() {
  collectOverrides();
  if (!S.baseScene || !S.sequence.length) { setStatus('need a base scene and at least one segment', true); return; }
  const name = $('presetName').value.trim() || defaultPresetName();
  try {
    const p = await api('/api/presets', 'POST', {
      name, scene: S.baseScene, base_scene: S.baseScene,
      object: S.object, robots: S.robots, sequence: S.sequence, overrides: S.overrides, goal: S.scene?.goal, controller_version: 'option-v2',
    });
    S.currentPreset = p; S.dirty = false;
    S.loadedInitial = {object: clonePose(S.object), robots: S.robots.map(clonePose)};
    $('presetName').value = p.name;
    setPresetLabel();
    setStatus(`preset saved: ${p.name}`);
    await refreshPresets();
  } catch (e) { setStatus(e.message, true); }
}
async function loadPreset(id) {
  try {
    const p = await api(`/api/presets/${id}`);
    applyPreset(p);
    setStatus(`loaded preset: ${p.name || p.id}`);
  } catch (e) { setStatus(e.message, true); }
}
async function deletePreset(id) {
  if (!confirm('Delete this preset?')) return;
  try { await api(`/api/presets/${id}`, 'DELETE'); await refreshPresets(); }
  catch (e) { setStatus(e.message, true); }
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
async function init() {
  try {
    await refreshControllerStatus();
    const [options, params, presets] = await Promise.all([
      api('/api/options'), api('/api/params'), api('/api/presets')]);
    S.options = options; S.params = params; S.presets = presets;
    initOptions(); renderParams(); renderLegend(); renderPresets(); refreshOverrides(); refreshRuns();
    const first = presets.find(p => p.name === 'seed_contact_2')
      || presets.find(p => (p.name || '').startsWith('seed_')) || presets[0];
    const query = new URLSearchParams(window.location.search);
    if (query.get('run')) {
      await loadRun(query.get('run'));
      const tick = Number(query.get('tick'));
      if (S.run && Number.isFinite(tick)) seek(Math.trunc(tick));
    } else if (first) await loadPreset(first.id);
    else setStatus('no presets available', true);
    setInterval(refreshControllerStatus, 15000);
  } catch (e) { setStatus(e.message, true); }
}
$('resetPoses').onclick = resetPoses;
if ($('controllerVersion')) $('controllerVersion').onchange = (e) => { S.controllerVersion = e.target.value; markDirty(); };
$('runBtn').onclick = runSequence;
$('addSegment').onclick = addSegment;
$('clearSequence').onclick = () => { S.sequence = []; renderSequence(); markDirty(); };
$('saveOverrides').onclick = saveOverrides;
$('savePreset').onclick = savePreset;
$('presetSort').onchange = (e) => { S.presetSort = e.target.value; renderPresets(); };
for (const [id, key] of [['viewLidar','lidar'],['viewTarget','target'],['viewForce','force'],['viewContact','contact'],['viewEstimate','estimate'],['viewYield','yield']]) {
  S.view[key] = $(id).checked;
  $(id).onchange = (e) => { S.view[key] = e.target.checked; drawScene(); };
}
for (const [id, key] of [['curveForce','force'],['curveSpeed','speed']]) {
  $(id).onchange = (e) => { S.curves[key] = e.target.checked; };
}
$('refreshRuns').onclick = refreshRuns;
$('playBtn').onclick = togglePlay;
$('prevBtn').onclick = () => seek(S.frame - 1);
$('nextBtn').onclick = () => seek(S.frame + 1);
$('frameSlider').addEventListener('input', e => seek(parseInt(e.target.value, 10)));
window.addEventListener('resize', () => { drawScene(); drawTimeline(); });
init();
