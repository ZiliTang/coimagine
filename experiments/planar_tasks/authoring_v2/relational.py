"""Separate experimental relational profile; old v2 release seals stay unchanged.

Provider-neutral tool dispatch for an external/native author. No model transport
is started here. Host owns fixtures; author receives only tool responses/knowledge.
"""
import argparse
import json
import time
import uuid
from pathlib import Path
from experiments.planar_tasks.relational_search.pipeline import Campaign, read, identifier, save


class RelationalSession:
    def __init__(self,path):self.campaign=Campaign(path)

    def tool(self,name,arguments):
        record=dict(id=uuid.uuid4().hex,name=name,arguments=arguments,started=time.time(),status='running')
        path=self.campaign.path/'public_tools'/(record['id']+'.json')
        if read(self.campaign.path/'campaign.json')['status']=='frozen':
            if name not in ('status','knowledge','context','fit_detail','semantic_source','trace'):raise ValueError('frozen campaign')
            return self._tool(name,arguments)
        save(path,record)
        try:
            result=self._tool(name,arguments);record.update(status='complete',result=result)
            return result
        except Exception as exc:
            record.update(status='error',error=str(exc));raise
        finally:record['finished']=time.time();save(path,record)

    def _tool(self,name,arguments):
        c=self.campaign
        methods=dict(new_round=c.new_round,context=c.context,replay=c.replay,
            brief=c.author_turn,repair=c.repair,compare=c.compare,
            submit_semantics=c.submit_semantics,fit=c.fit,apply_ranker=c.apply_ranker,test_plan=c.test_plan)
        if name in methods:return methods[name](**arguments)
        if name=='status':
            s=read(c.path/'campaign.json')
            return {k:s[k] for k in ('version','model','limits','used','status','rounds','fits','semantics','llm_usage')}
        if name=='knowledge':
            return {p.name:p.read_text() for p in sorted((c.path/'knowledge').glob('*.md'))}
        if name=='fit_detail':return read(c.path/'fits'/(identifier(arguments['fit_id'])+'.json'))
        if name=='semantic_source':return (c.path/'semantics'/(identifier(arguments['semantic_id'])+'.py')).read_text()
        if name=='trace':
            from experiments.planar_tasks.relational_search.pipeline import unpack
            start=arguments.get('start',0);count=arguments.get('count',1)
            if type(start) is not int or start<0 or type(count) is not int or not 1<=count<=5:raise ValueError('trace pagination')
            rows=unpack(c._apath(arguments['rid'],arguments['attempt_id'])/'public.json.gz')
            return dict(total=len(rows),rows=rows[start:start+count])
        raise ValueError('unknown relational author tool')


def verify_frozen(path):return Campaign(path).verify_frozen()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign',type=Path);p.add_argument('operation',choices=('create','tool','audit','freeze','verify'))
    p.add_argument('--request',type=Path,help='JSON tool {name, arguments}, or create kwargs')
    args=p.parse_args();request=read(args.request) if args.request else {}
    if args.operation=='create':
        Campaign.create(args.campaign,**request);result=dict(created=str(args.campaign))
    elif args.operation=='tool':result=RelationalSession(args.campaign).tool(**request)
    else:result=getattr(Campaign(args.campaign),'verify_frozen' if args.operation=='verify' else args.operation)()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))


if __name__=='__main__':main()
