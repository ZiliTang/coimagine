'use strict';
const $=id=>document.getElementById(id),node=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
let current=null,data=null,replay=null,playing=false,last=0,tick=0,busy=false;
async function api(path){const r=await fetch(path);if(!r.ok)throw Error(`记录暂不可读 (${r.status})`);return r.json();}
const base=()=>`/api/rl/${encodeURIComponent(current)}`;
function options(select,items,selected){select.replaceChildren(...items.map(([v,t])=>{const n=node('option',t);n.value=v;return n;}));if(items.some(([v])=>v===selected))select.value=selected;}
function chart(title,rows,key,xkey,xlabel){const div=node('div');div.className='chart';div.append(node('h3',title));const values=rows.filter(r=>Number.isFinite(r[key])&&Number.isFinite(r[xkey]));if(!values.length){div.append(node('p','此记录未保存该指标'));return div;}
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox','0 0 420 210');svg.setAttribute('role','img');svg.setAttribute('aria-label',title);div.append(svg);
 function el(tag,attrs,text){const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n;}
 const min=Math.min(...values.map(r=>r[key])),max=Math.max(...values.map(r=>r[key])),lo=min===max?min-Math.max(1,Math.abs(min)*.1):min,hi=min===max?max+Math.max(1,Math.abs(max)*.1):max;
 const xmin=values[0][xkey],xmax=values.at(-1)[xkey];const x=v=>56+348*(v-xmin)/Math.max(1,xmax-xmin),y=v=>170-150*(v-lo)/(hi-lo);
 for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;el('path',{d:`M56 ${y(v)} H404`,stroke:'#293749'});el('text',{x:2,y:y(v)+4,fill:'#9aabc0','font-size':10},Math.abs(v)<.001&&v!==0?v.toExponential(1):v.toFixed(2));}
 // Downsample only the display; exact metrics remain in the JSON artifact.
 const stride=Math.max(1,Math.ceil(values.length/1500)),points=values.filter((_,i)=>i%stride===0||i===values.length-1);
 el('polyline',{points:points.map(r=>`${x(r[xkey])},${y(r[key])}`).join(' '),fill:'none',stroke:'#75d4b5','stroke-width':2});
 el('circle',{cx:x(values.at(-1)[xkey]),cy:y(values.at(-1)[key]),r:3,fill:'#75d4b5'});
 el('text',{x:56,y:190,fill:'#9aabc0','font-size':11},String(xmin));el('text',{x:365,y:190,fill:'#9aabc0','font-size':11},String(xmax));el('text',{x:175,y:205,fill:'#9aabc0','font-size':11},xlabel);
 div.append(node('p',`最新 ${values.at(-1)[key].toPrecision(5)} · ${values.length} 个记录点`));return div;}
function render(){const s=data.state,m=data.metrics,rs=s.runs||[];const cfg=s.config||s.protocol||{},done=rs.filter(r=>r.status!=='running');
 const offset=s.continuation?.episode_offset||0;
 const cards=[['状态',`${s.phase||s.status} · ${s.completion||''}`],['采样',`${done.length}/${s.limits.attempts} 本轮 episode · 累计 ${offset+done.length} · ${cfg.workers||1} 路`],['PPO',`${m.rounds.length} 本轮批 · ${m.updates.length} 本轮 optimizer steps`],['实际物理 / 预留',`${s.used.execution_ticks} ticks · ${Object.keys(s.reservations||{}).length} 在途`]];
 $('cards').replaceChildren(...cards.map(([a,b])=>{const d=node('div');d.className='card';d.append(node('span',a),node('b',b));return d;}));
 $('workers').replaceChildren(...data.workers.map(w=>{const d=node('div');d.className='worker';d.append(node('b',`${w.id} · v${w.policy_version??'—'}`),node('div',w.case_id),node('div',w.progress?`PID ${w.progress.pid||'—'} · ${w.progress.tick} ticks · 边界 ${w.progress.index}`:'已排队 / 正在初始化'));return d;}));if(!data.workers.length)$('workers').append(node('p',s.phase==='updating'?'采样已收齐，正在更新策略。':'当前没有记录中的在途采样。'));
 const definitions=[['训练采样成功率','success_rate','rounds'],['任务比较成本（秒，截断计180）','mean_cost','rounds'],['决策 / 墙钟秒','decisions_per_second','rounds'],['物理 ticks / 墙钟秒','ticks_per_second','rounds'],['总损失','loss','updates'],['Actor 损失','actor','updates'],['Critic MSE','critic','updates'],['采样路径条件熵之和','entropy','updates'],['近似 KL','kl','updates'],['裁剪比例','clip_fraction','updates'],['残差惩罚','residual','updates'],['裁剪前梯度范数','grad_norm','updates'],['值函数解释方差','explained_variance','updates'],['概率重算最大误差','logp_recompute_error','rounds'],['墙钟截断 episode','wall_truncations','rounds'],['单批更新耗时（秒）','update_seconds','rounds'],['采样文件保存耗时（各 worker 之和，秒）','worker_save_seconds','rounds'],['宿主保存累计耗时（秒）','host_save_seconds','rounds']];
 $('charts').replaceChildren(...definitions.map(([title,key,kind])=>chart(title,m[kind],key,kind==='updates'?'optimizer_step':'round',kind==='updates'?'optimizer step':'采样批次（从0开始）')));
 if(s.continuation){
  for(const [title,key] of [['完整推荐动作比例','recommended_fraction'],['仅时长偏离比例','duration_only_deviation_fraction'],['目标或机器人动作偏离比例','motion_deviation_fraction']])$('charts').append(chart(title,m.rounds,key,'global_round','累计训练批次（从0开始）'));
  for(const family of ['open_transport','wall_release','clear_transport']){
   const rows=m.rounds.map(r=>({...r,rate:r.family_outcomes?.[family]?.episodes?r.family_outcomes[family].success/r.family_outcomes[family].episodes:null}));
   $('charts').append(chart(`${family} 随机训练成功率`,rows,'rate','global_round','累计训练批次（从0开始）'));
   const diag=(m.diagnostics||[]).map(r=>({...r,rate:r.families?.[family]?.success_rate??null}));
   $('charts').append(chart(`${family} 固定训练集确定性成功率`,diag,'rate','global_round','已完成累计训练批次'));
  }
 }
 const slots=data.schema?.slots||[];$('schema').textContent=slots.length?`${slots.filter(x=>x!==null).length} 个已用语义通道 / ${slots.length} 个固定槽位，${slots.filter(x=>x===null).length} 个预留。新通道启用需版本迁移并重新采样。`:'历史模型未预留命名槽位。';
 $('config').textContent=JSON.stringify({schema:data.schema,config:cfg,continuation:s.continuation,used:s.used,limits:s.limits,checkpoint:s.policy_checkpoint},null,2);
 const selected=$('episode').value;options($('episode'),done.filter(r=>r.status!=='infrastructure_error'&&r.record_details!==false).map(r=>[r.id,`${r.id} · ${r.case_id} · ${r.condition||'RL train'} · ${r.status}`]),selected);
 const t=node('table'),h=node('tr');['轨迹','初态','状态','物理 ticks','决策','墙钟秒','详细回放','保存秒'].forEach(x=>h.append(node('th',x)));t.append(h);
 for(const r of done.slice(-100).reverse()){const tr=node('tr');[r.id,r.case_id,r.status,r.ticks,r.decisions,r.wall_seconds?.toFixed(2),r.record_details===false?'未记录':'已记录 / 历史',r.save_seconds?.toFixed(3)].forEach(x=>tr.append(node('td',x??'—')));t.append(tr);}$('episodes').replaceChildren(t);
}
async function refresh(){if(busy)return;busy=true;try{$('error').textContent='';const list=await api('/api/rl');options($('experiment'),list.map(x=>[x.id,`${x.id} · ${x.phase||x.status}`]),current||new URLSearchParams(location.search).get('campaign')||list.at(-1)?.id);if(!list.length)return;const changed=current!==$('experiment').value;current=$('experiment').value;data=await api(base());render();if(changed){replay=null;tick=0;playing=false;await loadReplay();}history.replaceState(null,'',`/rl?campaign=${encodeURIComponent(current)}`);$('updated').textContent=`更新于 ${new Date().toLocaleTimeString()}`;}catch(e){$('error').textContent=e.message;}finally{busy=false;}}
async function loadReplay(){try{if(!$('episode').value){replay=null;playing=false;draw();$('time').textContent='没有已保存的详细回放（训练仍可正常进行）';return;}replay=await api(`${base()}/replay/${$('episode').value}`);if(replay.available===false){const reason=replay.reason;replay=null;draw();$('time').textContent=reason;return;}tick=0;$('seek').max=replay.frames.at(-1).tick;draw();}catch(e){$('error').textContent=e.message;}}
function draw(){const f=drawCanvas($('replay'),replay,tick);$('seek').value=tick;$('time').textContent=f?`${f.t.toFixed(2)}s · tick ${f.tick}`:'';}
$('refresh').onclick=refresh;$('experiment').onchange=()=>{current=null;const name=$('experiment').value;history.replaceState(null,'',`/rl?campaign=${encodeURIComponent(name)}`);refresh();};$('episode').onchange=loadReplay;$('seek').oninput=()=>{tick=Number($('seek').value);draw();};$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放';};
function animate(now){if(playing&&replay){tick=Math.min(tick+(now-last)/50*4,replay.frames.at(-1).tick);draw();if(tick>=replay.frames.at(-1).tick){playing=false;$('play').textContent='播放';}}last=now;requestAnimationFrame(animate);}requestAnimationFrame(animate);setInterval(()=>{if($('auto').checked)refresh();},5000);refresh();

function drawCanvas(canvas,replay,tick){const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);if(!replay)return null;const s=replay.scene,frames=replay.frames;let f=frames[Math.min(Math.round(tick),frames.length-1)]||frames[0];const w=canvas.width,h=canvas.height,pad=25,k=Math.min((w-2*pad)/s.arena[0],(h-2*pad)/s.arena[1]);
  ctx.save();ctx.translate(pad,h-pad);ctx.scale(k,-k);ctx.lineWidth=1/k;ctx.strokeStyle='#253445';for(let x=0;x<=s.arena[0];x+=.5){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,s.arena[1]);ctx.stroke();}for(let y=0;y<=s.arena[1];y+=.5){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(s.arena[0],y);ctx.stroke();}
  ctx.strokeStyle='#6e849e';ctx.strokeRect(0,0,...s.arena);ctx.fillStyle='#75d4b526';ctx.strokeStyle='#75d4b5';ctx.beginPath();ctx.arc(s.goal.x,s.goal.y,s.goal.radius,0,Math.PI*2);ctx.fill();ctx.stroke();
  function poly(pose,points,color){ctx.save();ctx.translate(pose[0],pose[1]);ctx.rotate(pose[2]);ctx.fillStyle=color;ctx.strokeStyle='#e1e9f3';ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fill();ctx.stroke();ctx.restore();}
  function box(x,y,a,hw,hh,color){poly([x,y,a],[[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]],color);}
  for(const b of s.obstacles)box(b.cx,b.cy,b.angle,b.half_w,b.half_h,'#465263');
  for(const [id,p] of Object.entries(f.entities)){const b=s.movable_objects.find(b=>b.label===id);if(id===s.primary_id)poly(p,s.object_polygon,'#bd944e');else if(b)box(...p.slice(0,3),b.width/2,b.height/2,'#608ab6');}
  f.robots.forEach((p,i)=>{ctx.fillStyle=i?'#75b286':'#799de1';ctx.beginPath();ctx.arc(p[0],p[1],s.robot_radius,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#ecf5ff';ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(p[0]+.24*Math.cos(p[2]),p[1]+.24*Math.sin(p[2]));ctx.stroke();});ctx.restore();
  function label(p,name,color){ctx.fillStyle=color;ctx.font='bold 13px system-ui';ctx.fillText(name,pad+p[0]*k+8,h-pad-p[1]*k-12);}
  Object.entries(f.entities).forEach(([id,p])=>label(p,id===s.primary_id?'A':id,'#edf4ff'));f.robots.forEach((p,i)=>label(p,'R'+i,'#edf4ff'));label([s.goal.x,s.goal.y],'GOAL','#75d4b5');return f;
}
