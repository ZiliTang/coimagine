'use strict';
const D=JSON.parse(document.getElementById('report-data').textContent);
const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function bars(id,labels,values,n){
 $(id).innerHTML=labels.map((label,i)=>`<div class="bar-row"><span>${esc(label)}</span><div class="track" title="${values[i]}/${n}，${(100*values[i]/n).toFixed(1)}%"><div class="bar" style="width:${100*values[i]/n}%"></div></div><span class="value">${values[i]}/${n}</span></div>`).join('');
}
bars('#coverage-chart',D.coverage.labels,D.coverage.counts,D.coverage.n);
function showSamples(n){bars('#attribution-chart',D.arms,D.attribution[n],216);$('#attribution-chart').insertAdjacentHTML('afterbegin',`<p class="caption">${n}条成功示范 · 216个固定诊断初态</p>`);document.querySelectorAll('#sample-switch button').forEach(b=>b.classList.toggle('active',b.dataset.n===n));}
function showSplit(s){bars('#transfer-chart',[...D.arms,'统一CAP'],D.transfer[s],s==='iid'?24:48);$('#transfer-chart').insertAdjacentHTML('afterbegin',`<p class="caption">${s==='iid'?'东／北向新留出 · 24例':'西／斜向跨布局迁移 · 48例'}</p>`);document.querySelectorAll('#split-switch button').forEach(b=>b.classList.toggle('active',b.dataset.split===s));}
document.querySelectorAll('#sample-switch button').forEach(b=>b.onclick=()=>showSamples(b.dataset.n));
document.querySelectorAll('#split-switch button').forEach(b=>b.onclick=()=>showSplit(b.dataset.split));showSamples('50');showSplit('iid');
$('#result-tables').innerHTML='<table><caption>输入归因：固定216例诊断面板</caption><thead><tr><th>输入</th><th>50示范</th><th>150示范</th></tr></thead><tbody>'+D.arms.map((a,i)=>`<tr><td>${a}</td><td>${D.attribution['50'][i]}/216</td><td>${D.attribution['150'][i]}/216</td></tr>`).join('')+'</tbody></table><table><caption>新关系迁移：独立留出；原8根另列诊断</caption><thead><tr><th>输入/策略</th><th>东／北留出</th><th>西／斜迁移</th></tr></thead><tbody>'+[...D.arms,'统一CAP'].map((a,i)=>`<tr><td>${a}</td><td>${D.transfer.iid[i]}/24</td><td>${D.transfer.transfer[i]}/48</td></tr>`).join('')+'</tbody></table>';

function drawScene(canvas,scene,frame,history=[]){
 canvas.style.aspectRatio=scene.arena[0]+'/'+scene.arena[1];
 const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
 const w=Math.max(100,rect.width),h=w*scene.arena[1]/scene.arena[0];
 if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
 const c=canvas.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,w,h);
 const pad=16,scale=Math.min((w-2*pad)/scene.arena[0],(h-2*pad)/scene.arena[1]);
 const point=p=>[pad+p[0]*scale,h-pad-p[1]*scale];
 c.fillStyle='#f3f6fa';c.fillRect(0,0,w,h);c.lineWidth=1;c.strokeStyle='#e1e8ef';
 for(let x=0;x<=scene.arena[0]+.01;x+=.5){const a=point([x,0]),b=point([x,scene.arena[1]]);c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke();}
 for(let y=0;y<=scene.arena[1]+.01;y+=.5){const a=point([0,y]),b=point([scene.arena[0],y]);c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke();}
 function polygon(poly,pose,fill,stroke){const co=Math.cos(pose[2]),si=Math.sin(pose[2]);c.beginPath();poly.forEach((v,i)=>{const p=point([pose[0]+v[0]*co-v[1]*si,pose[1]+v[0]*si+v[1]*co]);if(i)c.lineTo(...p);else c.moveTo(...p);});c.closePath();c.fillStyle=fill;c.fill();c.strokeStyle=stroke;c.lineWidth=1.5;c.stroke();}
 function box(b,pose,fill,stroke){polygon([[-b[0]/2,-b[1]/2],[b[0]/2,-b[1]/2],[b[0]/2,b[1]/2],[-b[0]/2,b[1]/2]],pose,fill,stroke);}
 for(const b of scene.obstacles)box([b.half_w*2,b.half_h*2],[b.cx,b.cy,b.angle||0],'#536274','#3f4e60');
 if(history.length>1){c.strokeStyle='#3578b280';c.lineWidth=2;c.setLineDash([3,4]);c.beginPath();history.forEach((f,i)=>{const p=point(f.entities[scene.primary_id]);if(i)c.lineTo(...p);else c.moveTo(...p);});c.stroke();c.setLineDash([]);}
 const goal=point([scene.goal.x,scene.goal.y]);c.beginPath();c.arc(...goal,scene.goal.radius*scale,0,Math.PI*2);c.fillStyle='#9863c514';c.fill();c.strokeStyle='#9364ce';c.lineWidth=2;c.setLineDash([5,3]);c.stroke();c.setLineDash([]);c.font='11px system-ui';c.textAlign='center';c.fillStyle='#7851a0';c.fillText('GOAL',goal[0],goal[1]-scene.goal.radius*scale-7);
 for(const b of scene.movable_objects){const pose=frame.entities[b.label];if(pose)box([b.width,b.height],pose,'#edbd6d','#b28032');}
 polygon(scene.object_polygon,frame.entities[scene.primary_id],'#72acda','#2b659a');
 const p=point(frame.entities[scene.primary_id]);c.fillStyle='#173f66';c.font='bold 11px system-ui';c.fillText('A',p[0],p[1]+4);
 frame.robots.forEach((pose,i)=>{const p=point(pose);c.beginPath();c.arc(...p,scene.robot_radius*scale,0,Math.PI*2);c.fillStyle=['#279a8d','#de8068'][i];c.fill();c.lineWidth=2;c.strokeStyle='white';c.stroke();c.beginPath();c.moveTo(...p);c.lineTo(p[0]+Math.cos(pose[2])*scene.robot_radius*scale,p[1]-Math.sin(pose[2])*scene.robot_radius*scale);c.strokeStyle='#ffffffaa';c.stroke();c.fillStyle='#fff';c.font='bold 10px system-ui';c.fillText('R'+i,p[0],p[1]+3);});
}
const layoutNames={east:'东向',north:'北向',west:'西向',diagonal:'斜向'};
$('#scene-grid').innerHTML=D.scenes.map((s,i)=>`<div class="scene-tile"><canvas id="scene-${i}" aria-label="${layoutNames[s.layout]}${s.relation==='blocked'?'挡路':'通道已空'}"></canvas><p>${layoutNames[s.layout]} · ${s.relation==='blocked'?'挡路':'通道已空'}${i<4?' · 开发布局':' · 迁移布局'}</p></div>`).join('');
$('#random-scene-grid').innerHTML=(D.random_scenes||[]).map((s,i)=>`<div class="scene-tile"><canvas id="random-scene-${i}" aria-label="${s.count}个可动物体"></canvas><p>${s.count}个可动物体 · ${esc(s.id)}</p></div>`).join('');
function renderScenes(){D.scenes.forEach((s,i)=>drawScene(document.getElementById('scene-'+i),s.scene,s.frame));(D.random_scenes||[]).forEach((s,i)=>drawScene(document.getElementById('random-scene-'+i),s.scene,s.frame));}

const examples={clear:[0],wall:[1],integration:[2,3],transfer:[4,5],...D.extra_examples};
if(examples.repair_learned) $('#example').insertAdjacentHTML('beforeend','<option value="repair_learned">同一失败前缀：教师修复 / 追加监督GDA接管</option>');
let current=[0],t=0,playing=false,last=null;
function label(id){return D.labels[id]||('动作 '+id);}
function atFrame(r,time){let lo=0,hi=r.frames.length-1;while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(r.frames[mid].t<=time)lo=mid;else hi=mid-1;}return lo;}
function renderPlayback(){
 current.forEach((id,i)=>{const r=D.replays[id],j=atFrame(r,t),f=r.frames[j];drawScene(document.getElementById('replay-'+i),r.scene,f,r.frames.slice(0,j+1).filter((_,k)=>k%4===0));
  let action=null;for(const a of r.actions){if(a.t<=t)action=a;else break;}
  const done=t>=r.frames.at(-1).t,target=action?.action.target_id;
  document.getElementById('action-'+i).innerHTML=action?`<b>${done?'执行结束':r.handoff_time!==undefined?(t<r.handoff_time?'共享失败前缀':'接管之后'):'当前决策'}</b> · ${target===null?'机器人移动 / 停驻':target===r.scene.primary_id?'目标物体 A':'可动障碍'}<br>R0: <code>${esc(label(action.action.a0))}</code><br>R1: <code>${esc(label(action.action.a1))}</code> · 请求 ${(action.action.ticks*.05).toFixed(2)} s · 实际 ${(action.end-action.t).toFixed(2)} s`:'初始状态';
 });
 $('#seek').value=t;$('#time').textContent=t.toFixed(2)+' s';
}
function setExample(key){playing=false;$('#play').textContent='播放';current=examples[key];t=0;const max=Math.max(...current.map(i=>D.replays[i].frames.at(-1).t));$('#seek').max=max;$('#endtime').textContent=max.toFixed(2)+' s';$('#players').classList.toggle('single',current.length===1);
 $('#players').innerHTML=current.map((id,i)=>{const r=D.replays[id];return `<article class="player"><div class="player-title"><b>${esc(r.title)}</b><span class="status ${r.summary.success?'':'fail'}">${r.summary.success?'最终成功':r.summary.error?'最终失败 / 非法建议':'最终未成功'}</span></div><canvas id="replay-${i}" aria-label="${esc(r.title)}"></canvas><div class="action-box" id="action-${i}"></div></article>`;}).join('');
 $('#selection-note').textContent=current.map(i=>D.replays[i].note).join('　');$('#handoff').hidden=!current.some(i=>D.replays[i].handoff_time!==undefined);renderPlayback();
}
$('#example').onchange=e=>setExample(e.target.value);
$('#play').onclick=()=>{playing=!playing;if(playing&&t>=Number($('#seek').max))t=0;last=null;$('#play').textContent=playing?'暂停':'播放';};
$('#seek').oninput=e=>{t=Number(e.target.value);renderPlayback();};
$('#reset').onclick=()=>{t=0;playing=false;$('#play').textContent='播放';renderPlayback();};
function jump(direction){const times=[...new Set(current.flatMap(i=>D.replays[i].actions.map(a=>a.t)))].sort((a,b)=>a-b);const next=direction>0?times.find(x=>x>t+.001):times.filter(x=>x<t-.001).at(-1);if(next!==undefined)t=next;playing=false;$('#play').textContent='播放';renderPlayback();}
$('#prev').onclick=()=>jump(-1);$('#next-boundary').onclick=()=>jump(1);
function animate(now){if(playing){if(last!==null)t=Math.min(Number($('#seek').max),t+Math.min((now-last)/1000,.1)*Number($('#speed').value));renderPlayback();if(t>=Number($('#seek').max)){playing=false;$('#play').textContent='播放';}}last=now;requestAnimationFrame(animate);}
setExample('clear');renderScenes();requestAnimationFrame(animate);
new ResizeObserver(()=>{renderScenes();renderPlayback();}).observe(document.querySelector('main'));
$('#print').onclick=()=>window.print();
$('#source-list').innerHTML=D.sources.map((s,i)=>`<div class="source-item" id="source-${i}"><a href="../../../../${esc(s.path)}">[${i+1}] ${esc(s.title)}</a><code>${esc(s.path)}</code></div>`).join('');
document.querySelectorAll('[data-source]').forEach(a=>a.href='#source-'+a.dataset.source);
window.addEventListener('beforeprint',()=>{renderScenes();renderPlayback();});

document.querySelectorAll('[data-example]').forEach(a=>a.onclick=()=>{$('#example').value=a.dataset.example;setExample(a.dataset.example);});

$('#handoff').onclick=()=>{const r=current.map(i=>D.replays[i]).find(r=>r.handoff_time!==undefined);if(r){t=r.handoff_time;playing=false;$('#play').textContent='播放';renderPlayback();}};
