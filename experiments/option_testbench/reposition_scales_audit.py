"""Audit new distance coverage without conflating it with a controller improvement."""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .reposition_arc_audit import audit
else:
    from reposition_arc_audit import audit


def main(out):
    records=[json.loads(p.read_text()) for p in sorted((out/'logs').glob('*.json'))]
    results=[audit(r) for r in records if r['metric']['status']=='completed']
    m=json.loads((out/'manifest.json').read_text())
    report={'version':'reposition-scales-v2-audit','action_version':m['version'],
            'scales_m':[.2,.4,.8,1.2,1.8],'results':results,
            'source_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                             for n in ('reposition_scales_audit.py','reposition_arc_audit.py','reposition_combinations_audit.py')},
            'opposite_face_label':'offline perpendicular projection onto opposite long face, separate from distance completion'}
    (out/'scale_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 新尺度覆盖与反侧长面标签','','实测行程完成和达到反侧长面分别报告。新旧菜单按各自冻结参数解码。','',
           '| 场景 | RP 请求 → 计程 | RP 退出 | 反侧长面 | 耗时 s |','|---|---|---|---|---:|']
    for r in results:
        options=[o for o in r['options'] if o['kind']=='reposition']
        progress=' / '.join(f"{o['target']:.2f} → {o['recorded_progress']:.4f} m" for o in options)
        exits=' / '.join(o['exit'] for o in options);back=' / '.join(str(o['opposite_long_face_reached']) for o in r['physical'])
        lines.append(f"| {r['case']} | {progress} | {exits} | {back} | {r['seconds']:.2f} |")
    (out/'SCALE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    plot(records,out)
    print(json.dumps({'audited':len(results),'periods':sum(r['samples'] for r in results),
                      'max_error_m':max(r['max_reconstruction_error_m'] for r in results)}))


def plot(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    colors=['#5391b8','#c89540','#9467bd','#c96a70','#218d70']
    for ax,direction in zip(axes,('CW','CCW')):
        ax.add_patch(Rectangle((2.,1.875),1.,.25,color='#aeb9c9',alpha=.5))
        ax.add_patch(Rectangle((2.,2.305),1.,.24,color='#218d70',alpha=.08))
        for distance,color in zip((.2,.4,.8,1.2,1.8),colors):
            r=next(r for r in records if r['metric']['case']==f'scale2_rp_{direction}_solo_{distance:.2f}')
            rows=[x for x in r['rows'] if x['r0']['kind']=='reposition']
            points=[x['true']['robots'][0] for x in rows]
            if distance==1.8:ax.plot([p[0] for p in points],[p[1] for p in points],color=color,label='rounded path, 1.8 m')
            x,y=points[-1][:2];ax.scatter(x,y,color=color,s=35,zorder=5,label=f'{distance:.1f} m end')
            ax.annotate(f'{distance:.1f}',(x,y),xytext=(7,7),textcoords='offset points',color=color)
        ax.set(title=f'{direction}: requested distances and measured endpoints',aspect='equal',
               xlim=(1.45,3.55),ylim=(1.45,2.65),xlabel='World x / m',ylabel='World y / m')
        ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower '+('right' if direction=='CW' else 'left'))
    fig.suptitle('New distance menu: 1.8 m reaches the opposite long face in these solo scenes')
    fig.savefig(out/'reposition_scales.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
