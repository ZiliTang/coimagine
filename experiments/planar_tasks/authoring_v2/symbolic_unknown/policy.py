"""One literal-rule representation for runtime choice and symbolic obligations."""
from experiments.planar_tasks.authoring_v2.symbolic_general.core import Policy


def actions():
    out={'brake':dict(kind='brake',robot=None,direction=None,ticks=4)}
    for robot in (0,1):
        for direction in range(8):
            out[f'push_{robot}_{direction}']=dict(kind='push',robot=robot,direction=direction,ticks=4)
            out[f'view_{robot}_{direction}']=dict(kind='observe',robot=robot,direction=direction,ticks=4)
    return out


def schema():
    metrics={'need_observation':dict(unit='bool'),'obstacle_blocks_goal':dict(unit='bool')}
    options={}
    for name,a in actions().items():
        for field in ('legal','known','free','dynamic','bound_valid'):
            metrics[field+':'+name]=dict(unit='bool')
        metrics['score:'+name]=dict(unit='1',domain=[-1000,1000])
        options[name]=dict(requires=['and',*[['var',f+':'+name]for f in ('legal','known','free','dynamic','bound_valid')]],
            writes=list(metrics),frame=[],productive=a['kind']!='brake')
    for option in options.values():option['writes']=list(metrics)
    return dict(metrics=metrics,options=options)


def program():
    host=schema();rules=[]
    for name,a in actions().items():
        when=host['options'][name]['requires']
        if a['kind']=='observe':when=['and',when,['var','need_observation'],['not',['var','obstacle_blocks_goal']],['gt',['var','score:'+name],['lit',0,'1']]]
        elif a['kind']=='push':when=['and',when,['gt',['var','score:'+name],['lit',99,'1']]]
        rules.append(dict(name=name,option=name,when=when,score=['var','score:'+name]))
    return dict(version=1,derived={},rules=rules,expect={})


def compiled(program_value=None):return Policy(program_value or program(),schema())
