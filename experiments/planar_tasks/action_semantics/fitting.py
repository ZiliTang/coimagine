"""Build labelled imitation sets separately from verified physical preferences."""
import json
from pathlib import Path
from experiments.planar_tasks.relational_search.pipeline import read,unpack
from experiments.planar_tasks.relational_search.pipeline_execution import history_at
from experiments.planar_tasks.relational_search.pipeline_semantics import embedding
from experiments.planar_tasks.relational_search.contract import canonical,digest,History
from experiments.planar_tasks.relational_search.dataset import verify
from .program import Program,action_encoding
from .network import fit


def build(campaign,source,demonstration_refs,comparison_refs,progress=None,on_feature=None):
    program=Program(source,on_feature);origins={};provenance=[]
    for rid,aid in demonstration_refs:
        round=campaign._round(rid)
        if round['split']!='train':raise ValueError('demonstrations must be training group')
        path=campaign._apath(rid,aid);summary=read(path/'summary.json')
        if summary['status']!='success':raise ValueError('imitation witness must be successful')
        rows=unpack(path/'public.json.gz');commands=read(path/'request.json')['commands'];h=History()
        for i,row in enumerate(rows[:-1]):
            h.append(row['observation'],row['action'])
            if h.sha256!=row['history_hash']:raise ValueError('history checksum')
            key=h.sha256;item=origins.setdefault(key,dict(history=h.fork(),teachers=[],refs=[]))
            if commands[i] not in item['teachers']:item['teachers'].append(commands[i])
            item['refs'].append([rid,aid,i])
    if len(origins)>600:raise ValueError('at most600 distinct training histories per fit')
    items=[]
    def features(h,extras):
        packet=program.evaluate(h,extras)
        return packet,[embedding(h,a)+action_encoding(h.events[-1]['observation'],a) for a in packet['actions']]
    for index,(key,item) in enumerate(origins.items()):
        h=item['history'];extras=[]
        for teacher in item['teachers']:
            extras.append(teacher)
            for ticks in (4,10,20,40):extras.append(dict(teacher,ticks=ticks))
        packet,raw=features(h,extras)
        items.append(dict(kind='demonstration',origin=key,refs=item['refs'],actions=packet['actions'],positive=[packet['actions'].index(a) for a in item['teachers']],semantic=packet['semantic'],raw=raw,proposals=packet['proposals']))
        if progress and index%20==0:progress(index,len(origins))
    for rid,cid in comparison_refs:
        for line in (campaign._rpath(rid)/'comparisons'/(cid+'.jsonl')).read_text().splitlines():
            row=json.loads(line);h=verify(row)
            if row['split']!='train':raise ValueError('no held-out fitting')
            if row['comparison']['status']!='preference':continue
            c=row['comparison'];preferred=c['preferred']['action'];other=c['other']['action'];packet,raw=features(h,[preferred,other])
            items.append(dict(kind='physical_preference',origin=h.sha256,refs=[[rid,cid]],actions=packet['actions'],preferred=packet['actions'].index(preferred),other=packet['actions'].index(other),semantic=packet['semantic'],raw=raw,proposals=packet['proposals']))
    if not items:raise ValueError('no successful demonstration histories')
    return dict(items=items,feature_names=packet['names'],feature_batches=program.calls,dataset_sha256=digest(items),
                label_semantics='Demonstrations designate witnessed successful choices; alternatives are NOT physical failures. Physical preference uses matched continuation only.')


def fit_dataset(dataset,source):
    models=fit(dataset['items'])
    return dict(version='action-semantics-mlp-v1',semantic_sha256=digest(source),semantic_names=dataset['feature_names'],dataset_sha256=dataset['dataset_sha256'],models=models,
                demonstrations=sum(i['kind']=='demonstration' for i in dataset['items']),physical_pairs=sum(i['kind']=='physical_preference' for i in dataset['items']),
                label_semantics=dataset['label_semantics'],architecture='two hidden layers32/32 tanh; shared per candidate',seed=17,epochs=400)
