"""Audit the predicted node at real handoffs and exploratory transitions."""
from collections import Counter
import numpy as np
from tools import cap_option_handoff as exp
from tools import cap_option_handoff_report as rep


def main():
    centres,_=exp.graph.frozen_model()
    rules=exp.read(exp.OUT/'frozen.json')['winner']
    x,y=rules['x'],rules['y']
    metrics={}
    for fold in ('dev','fresh'):
        data=rep.cases(fold)
        rows=rep.rows_for(data,x,y)
        nodes=Counter()
        for r,row in zip(data,rows):
            if row['switch']:
                bits=r['arms']['AA']['mid']['bits']
                label=int(exp.graph.nearest(np.array([bits],float),centres)[0])
                nodes[label]+=1
        tables={}
        for tag in rep.CONDITIONS:
            table=rep.summarize(rep.rows_for(data,'any',tag),True)
            tables[tag]=dict(formation=table['formation'],b_after_a=table['b_progress_by_origin']['A'],
                             b_after_b=table['b_progress_by_origin']['B'])
        outside=[r for r in rows if r['AA']==0 and r['single_b']==0]
        rescued=[dict(room=r['room'],routes=[k for k in ('AB','BA','BB') if r[k]])
                 for r in outside if any(r[k] for k in ('AB','BA','BB'))]
        metrics[fold]=dict(physical_at_active=dict(nodes),transitions=tables,
                           outside=len(outside),rescued_outside=rescued)
    exp.save(exp.DOC/'edge_audit.json',metrics)
    dev_edge=metrics['dev']['transitions']['node2']['formation']
    v=metrics['fresh']['transitions']['node2']['formation']
    correct=[metrics[k]['physical_at_active'].get(4,0) for k in ('dev','fresh')]
    totals=[sum(metrics[k]['physical_at_active'].values()) for k in ('dev','fresh')]
    example=exp.read(exp.DOC/'examples.json')
    example=next((r for r in example if r['final_A']['success'] and not r['final_AB']['success']),None)
    e=''
    if example:
        e=f"""在新房间 `{example['room']}`，接手前真实终距为 {100*example['mid']['gap']:.2f} cm。A 的一条提案推进 {100*example['continue_A']['gain']:.2f} cm（{example['continue_A']['ticks']} tick），B 推进 {100*example['take_B']['gain']:.2f} cm（{example['take_B']['ticks']} tick）。这条 B 分支随后交回 A，最终结果为 {example['final_AB']['termination']}；一直留 A 在第 {example['final_A']['ticks']} tick 成功。两条提案均为真实自然时长。该例显示同一阈值两侧的局部推进可以与最终成败不一致。"""
    text=f"""## 对节点解释的离线复核

在冻结规则真正触发的交接点，公开识别器均判为簇 4，离线由八项物理关系计算的真实簇也为 4 的有开发 {correct[0]}/{totals[0]}、新房间 {correct[1]}/{totals[1]}。这限制了把该公开路由规则直接解释成某个物理站位机制的程度。

逐边描述另有一个可能值得单独复验的方向。对原来不在公开簇 2 的相同前缀，执行 A、B 各一条提案而进入簇 2 的次数分别为开发 {dev_edge['first']}、{dev_edge['second']}（{dev_edge['n']} 个前缀），新房间 {v['first']}、{v['second']}（{v['n']} 个前缀）；新面板 B 相对 A 的百分点差区间 [{-100*v['ci'][1]:+.1f},{-100*v['ci'][0]:+.1f}]。这一条从多条边里事后挑出，区间未作多重比较修正，且进入该簇本身未必有利。它提示专家有时确实使下一状态朝不同方向走，而这仍需与下一位专家的实际接手效果连接。

{e}

[各条件的全部转移读数及接手点物理节点](edge_audit.json)
"""
    fresh=metrics['fresh']
    examples=fresh['rescued_outside']
    extra='\n## 能力超出原专家并集的真实病例\n\n'
    extra+=f"在两个单专家从头执行都失败的 {fresh['outside']} 个新房间里，有 **{len(examples)} 个** 在预定分叉改用两条命令的某个组合、随后还给 A，最终获得真实成功。"
    if examples:
        extra+=f"例如 `{examples[0]['room']}` 的成功分支为 {', '.join(examples[0]['routes'])}。"
    extra+='这证明重组已有命令有时能产生原库外成功；如何在执行前选出对应分支仍待验证。[逐房间成功分支](edge_audit.json)\n'
    path=exp.DOC/'README.md'; orig=path.read_text()
    if '## 对节点解释的离线复核' not in orig:path.write_text(orig+'\n'+text)
    if '## 能力超出原专家并集的真实病例' not in path.read_text():path.write_text(path.read_text()+extra)
    print(correct,totals,v)


if __name__=='__main__':main()
