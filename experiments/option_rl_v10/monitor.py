"""Every-iteration, auto-refreshing local dashboard; no server/CDN required."""
from pathlib import Path
import html,json,os
from experiments.option_rl_v45.monitor import TEMPLATE
from experiments.option_rl_v10.live import read_json

CHARTS=[
 ('Return', [('explore_return','train','#4aa3ff'),('eval_return','validation','#ffb347')]),
 ('Success', [('explore_success','train all','#4ad991'),('success_rate','validation all','#ffb347'),
              ('train_outside_goal_success','train initially outside goal','#9ddcf7'),('eval_outside_goal_success','validation initially outside goal','#e58ac9')]),
 ('Progress (goal fraction)', [('explore_progress','train','#4ad991'),('mean_progress','validation','#ffb347')]),
 ('Policy loss', [('policy_loss','policy','#c792ea')]),
 ('Value loss', [('value_loss','value','#4aa3ff')]),
 ('Entropy (joint robot actions)', [('entropy','entropy','#ffd166')]),
 ('Reward components', [('reward_progress','progress','#4ad991'),('reward_distance_cost','distance','#ff6b6b'),
                       ('reward_time_cost','time','#ffb347'),('reward_success_bonus','success','#4aa3ff'),('reward_failure_penalty','failure','#c792ea')]),
 ('Anomaly rate', [('train_anomaly_rate','train quarantined','#ff6b6b'),('eval_anomaly_rate','validation counted as failures','#ffb347')]),
 ('Actor trunk gradient', [('actor_trunk_grad_norm','actor','#4ad991')]),
 ('Critic trunk gradient', [('critic_trunk_grad_norm','critic','#4aa3ff')]),
 ('PPO KL / clip fraction', [('approx_kl','approx KL','#ffd166'),('clip_fraction','clip fraction','#c792ea')]),
 ('Sample / PPO log-prob error', [('initial_logp_max_error','max absolute error','#ff6b6b')]),
 ('Action mix', [('action_translate','translation PUSH','#4ad991'),('action_rotate','rotation','#c792ea'),
                 ('action_reposition','REPOSITION','#4aa3ff'),('action_follow','FOLLOW','#ffb347'),('action_wait','WAIT','#8b93a7')]),
 ('Option exits', [('blocked_exit_rate','blocked / safety exits','#ff6b6b'),('short_option_rate','options <= 0.15 s','#c792ea')]),
 ('Validation success by distance', [('val_near_success','near <0.75 m','#4ad991'),('val_middle_success','middle 0.75-1.5 m','#ffb347'),('val_far_success','far >=1.5 m','#c792ea')]),
 ('Validation progress by distance', [('val_near_progress','near','#4ad991'),('val_middle_progress','middle','#ffb347'),('val_far_progress','far','#c792ea')]),
 ('Samples / iteration', [('steps','PPO transitions','#4aa3ff'),('valid_episodes','valid episodes','#4ad991'),('anomaly_episodes','quarantined episodes','#ff6b6b')]),
 ('Wall seconds / iteration', [('iteration_seconds','seconds','#8b93a7')]),
]


def write_dashboard(history,path,*,subtitle=''):
    path=Path(path);live=read_json(path.parent/'live_status.json',{}) or {}
    supervision=read_json(path.parent/'supervisor_status.json',{}) or {}
    content=TEMPLATE.format(title='effect68 RL live',subtitle=html.escape(subtitle),
        data=json.dumps(history,allow_nan=False).replace('<','\\u003c'),charts=json.dumps(CHARTS))
    content=content.replace('minmax(430px,1fr)','minmax(min(100%,360px),1fr)')
    manifest=read_json(path.parent/'manifest.json',{}) or {}
    every=manifest.get('config',{}).get('eval_every',10)
    note=(f'训练曲线每轮写入；固定验证每 {every} 轮重测（首轮也评估），空点不伪造评估值。'
          '页面每 10 秒刷新；评估期间显示已完成根数。')
    status_html='<div id="live-status" style="padding:10px;border:1px solid #344256;border-radius:7px">'+html.escape(
        f"状态 {live.get('phase','starting')} · 轮次 {live.get('iteration',0)} · "
        f"采样 workers {manifest.get('config',{}).get('workers',1)} · "
        f"回合 {live.get('episode',0)}/{live.get('episodes',0)} · "
        f"验证 {live.get('eval_done',0)}/{live.get('eval_total',0)} · 更新时间 {live.get('updated_local','—')}")+'</div>'
    status_html+='<div style="padding:8px 0;color:#a9bdd6">'+html.escape(note)+'</div>'
    status_html+='<div id="supervision">'+html.escape('监测：'+str(supervision.get('decision','waiting')))+' · <a style="color:#8bbdff" href="monitor_summary.md">阶段总结</a> · <a style="color:#8bbdff" href="episodes.html">行为回放</a></div>'
    content=content.replace('</header>',status_html+'</header>')
    # Draw real evaluation dots, including the first single measurement.
    marker="    ctx.stroke();\n  });\n}\nconst wrap"
    replacement="""    ctx.stroke();
    vals.forEach((v,i)=>{if(v===null||!isFinite(v))return;const x=pad+(W-pad-8)*i/xmax,y=pad+(Hh-2*pad)*(1-(v-lo)/(hi-lo));ctx.beginPath();ctx.arc(x,y,2.7,0,Math.PI*2);ctx.fillStyle=colour;ctx.fill();});
  });
}
const wrap"""
    content=content.replace(marker,replacement)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(content);temp.replace(path)
    return path
