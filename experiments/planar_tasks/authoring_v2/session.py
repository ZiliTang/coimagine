"""Budget settlement, legal batch experiments, selective evidence, and durable tool records."""
from pathlib import Path
import json,math,re,time
from copy import deepcopy
from experiments.planar_tasks.authoring import session as old
from experiments.planar_tasks.baseline import session as base, seal
from experiments.planar_tasks.interface import validate_action
from . import release, reports
from .profile import DEFAULT_PROFILE, validate_profile


def load(p):return json.loads(Path(p).read_text())
def save(p,x):base.write_json(Path(p),x)
def valid_int(x,lo,hi):return type(x)is int and lo<=x<=hi


def verify_profile(path):
    release.verify();path=Path(path);s=load(path/'state.json');p=load(path/'experiment_profile.json')
    if seal.digest(path/'experiment_profile.json')!=s.get('experiment_profile_sha256'):raise ValueError('experiment profile changed')
    if p['release_sha256']!=seal.digest(release.LOCK):raise ValueError('experiment release mismatch')
    validate_profile(p['settings'])
    for name,h in p['materials'].items():
        f=path/'workspace'/name
        if f.is_symlink() or seal.digest(f)!=h:raise ValueError('experiment material changed: '+name)
    return p


class ExperimentMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        try:
            self.experiment_profile=verify_profile(self.path)
            self.settings=self.experiment_profile['settings']
            if self.predictor:self.predictor.ledger.limits=deepcopy(self.settings['prediction_limits'])
        except Exception:self.close();raise

    def _active(self):
        super()._active();verify_profile(self.path)

    def budget_status(self):
        b=super().budget_status()
        b['accounting']='resets: explicit development reset only; policy_tests include their own reset. Test ticks and script time reserved then settled to actual; unknown failures keep full reservation. Model output remains reserved.'
        if hasattr(self,'settings'):b['prediction_limits']=deepcopy(self.settings['prediction_limits'])
        return b

    def _charge(self,key,value=1):
        if getattr(self,'_inside_test',False) and key=='resets':
            u=self.state['usage'];u['policy_resets']=u.get('policy_resets',0)+value;self._save();return
        super()._charge(key,value)
        if getattr(self,'_inside_test',False) and key=='control_ticks':self._test_reserved+=value

    def test_policy(self,files,task,max_ticks):
        self._inside_test=True;self._test_reserved=0;before=len(self.state['tests']);result=None
        try:
            result=super().test_policy(files,task,max_ticks)
        finally:
            reserved=self._test_reserved;self._inside_test=False
            actual=None
            if len(self.state['tests'])>before:
                tid=self.state['tests'][before]['id'];p=self.path/'private'/tid/'episode/summary.json'
                if p.exists():actual=load(p).get('executed_control_ticks')
            if reserved:
                if actual is not None and valid_int(actual,0,reserved):
                    self.state['usage']['control_ticks']-=reserved-actual
                else:actual=None
                self.state.setdefault('settlements',[]).append(dict(kind='policy_test',test_index=before,reserved_ticks=reserved,actual_ticks=actual,charged_ticks=reserved if actual is None else actual))
                self._save()
        if result is not None:result['overview']=self.test_summary(result['test_id'])
        return result

    def run_python(self,path,seconds):
        before=self.state['usage'].get('python_total_seconds',0);result=None
        try:result=super().run_python(path,seconds)
        finally:
            reserved=self.state['usage'].get('python_total_seconds',0)-before
            actual=result.get('wall_seconds') if result else None
            if reserved and isinstance(actual,(int,float)) and math.isfinite(actual) and actual>=0:
                self.state['usage']['python_total_seconds']+=actual-reserved
                self.state.setdefault('settlements',[]).append(dict(kind='python',run_id=result['run_id'],reserved_seconds=reserved,actual_seconds=actual))
                self._save()
        return result

    def write_file(self,path,text):
        if path in ('experiment_api.md','experiment_profile.json','batch_client.py'):raise ValueError('host-owned experiment material')
        return super().write_file(path,text)

    def _test_path(self,test_id):
        if test_id not in {t['id'] for t in self.state['tests']}:raise ValueError('unknown test')
        return self.path/'private'/test_id/'episode'

    def _batch_path(self,batch_id):
        if batch_id not in self.state.get('batches',[]):raise ValueError('unknown batch')
        return self.path/'private'/batch_id

    def _observations(self,record_id,branch=None):
        if record_id in self.state.get('batches',[]):
            p=self._batch_path(record_id);rows=load(p/'rows.json')
            if not valid_int(branch,0,len(rows)-1):raise ValueError('branch index required')
            path=p/f'branch_{branch:04d}.jsonl'
            if not path.exists():return []
            return [json.loads(l) for l in path.read_text().splitlines()]
        if branch is not None:raise ValueError('branch only applies to batches')
        p=self._test_path(record_id)/'sensors.jsonl'
        if not p.exists():return []
        return [json.loads(l)['request']['observation'] for l in p.read_text().splitlines()]

    def test_summary(self,test_id):
        p=self._test_path(test_id);test=next(t for t in self.state['tests'] if t['id']==test_id)
        summary=base.public_summary(load(p/'summary.json')) if (p/'summary.json').exists() else dict(status=test['status'])
        facts=reports.summarize(self._observations(test_id));facts.pop('events')
        traces=self._search_traces(test_id)
        return dict(test_id=test_id,task=test['task'],summary=summary,observations=facts,
                    search=reports.search_summary(traces),detail_tools=['test_events','read_trace','read_log'])

    def _search_traces(self,test_id):
        p=self._test_path(test_id);traces=[]
        if (p/'policy.stderr').exists():
            for l in (p/'policy.stderr').read_text().splitlines():
                try:t=json.loads(l)
                except ValueError:continue
                if isinstance(t,dict) and t.get('kind')=='option_search_v3':traces.append(t)
        return traces

    def test_events(self,record_id,branch=None,start=0,count=20):
        if not valid_int(start,0,1000000) or not valid_int(count,1,50):raise ValueError('invalid page')
        events=reports.summarize(self._observations(record_id,branch))['events']
        if record_id not in self.state.get('batches',[]):
            events+=reports.prediction_events(self._search_traces(record_id))
        return dict(record_id=record_id,branch=branch,total=len(events),rows=events[start:start+count],next_start=start+count if start+count<len(events) else None)

    def read_trace(self,record_id,branch=None,start=0,count=5,fields=None,robots=None):
        if not valid_int(start,0,1000000) or not valid_int(count,1,20):raise ValueError('invalid page')
        fields=fields if fields is not None else ['own','execution','contacts'];robots=robots if robots is not None else [0,1]
        reports.project({'robots':[]},fields,robots)
        obs=self._observations(record_id,branch);rows=[];size=0
        for i in range(start,min(len(obs),start+count)):
            row=dict(row=i,**reports.project(obs[i],fields,robots));n=len(json.dumps(row).encode())
            if size+n>self.settings['tool_reply_bytes']-1500:
                if not rows:raise ValueError('one row too large; select fewer fields/robots or read_log raw slice')
                break
            rows.append(row);size+=n
        next_start=start+len(rows)
        return dict(record_id=record_id,branch=branch,total=len(obs),rows=rows,next_start=next_start if next_start<len(obs) else None)

    def _validate_commands(self,commands,maximum):
        if not isinstance(commands,list) or len(commands)>maximum:raise ValueError('command list limit')
        for a in commands:
            if not isinstance(a,dict) or set(a)!={'a0','a1','ticks'} or any(type(v)is not int for v in a.values()) or not 1<=a['ticks']<=40:raise ValueError('exact integer command required')

    def run_branches(self,task,prefix,candidates,deadline=None):
        if task not in self.config['development_tasks']:raise ValueError('fixture unavailable')
        self._validate_commands(prefix,self.settings['max_prefix_commands'])
        if not isinstance(candidates,list) or not 1<=len(candidates)<=self.settings['max_branches']:raise ValueError('branch count limit')
        for c in candidates:self._validate_commands(c,self.settings['max_branch_commands'])
        if any(not c for c in candidates):raise ValueError('empty branch')
        requested=sum(a['ticks'] for a in prefix)*len(candidates)+sum(a['ticks'] for c in candidates for a in c)
        if requested>self.available('control_ticks') or len(candidates)>self.available('resets'):raise ValueError('batch worst-case ticks/resets exceed remaining quota; split batch')
        end=min(deadline or float('inf'),time.monotonic()+self.settings['batch_wall_seconds'],time.monotonic()+self.experiment_wall())
        ident=f"batch_{len(self.state.get('batches',[])):04d}";dest=self.path/'private'/ident;dest.mkdir()
        self.state.setdefault('batches',[]).append(ident);self._save()
        save(dest/'request.json',dict(task=task,prefix=prefix,candidates=candidates))
        rows=[];save(dest/'rows.json',rows);start_usage=self.state['usage'].get('control_ticks',0)
        for index,commands in enumerate(candidates):
            if time.monotonic()>=end or self.state.get('finalizing'):break
            row=dict(branch=index,commands=commands,status='starting',error=None);observations=[];branch_start=None
            tick_start=self.state['usage'].get('control_ticks',0)
            try:
                r=self.rpc({'method':'reset','task':task},end);observations.append(r['observation'])
                for a in prefix:
                    r=self.rpc({'method':'step','action':a},end);observations.append(r['observation'])
                branch_start=len(observations)-1
                for a in commands:
                    r=self.rpc({'method':'step','action':a},end);observations.append(r['observation'])
                    if r.get('terminated'):break
                row.update(status='complete',terminated=r.get('terminated'),success=r.get('success'),termination=r.get('termination'))
            except Exception as exc:
                row.update(status='incomplete',error=dict(type=type(exc).__name__,message=str(exc)[:500]))
            facts=reports.summarize(observations[branch_start:] if branch_start is not None else []);facts.pop('events')
            row.update(executed_ticks=self.state['usage'].get('control_ticks',0)-tick_start,
                       branch_start_row=branch_start,observations=facts)
            with (dest/f'branch_{index:04d}.jsonl').open('w') as f:
                for o in observations:f.write(json.dumps(o,allow_nan=False)+'\n')
            rows.append(row);save(dest/'rows.json',rows)
        result=dict(batch_id=ident,requested_branches=len(candidates),attempted=len(rows),
                    completed=sum(r['status']=='complete' for r in rows),
                    errors=sum(r['error'] is not None for r in rows),
                    executed_ticks=self.state['usage'].get('control_ticks',0)-start_usage,
                    prefix_ticks_included=True,status='complete' if len(rows)==len(candidates) and all(r['status']=='complete' for r in rows) else 'incomplete',
                    detail_tools=['batch_rows','test_events','read_trace'])
        save(dest/'summary.json',result);return dict(result,preview=self.batch_rows(ident,count=5)['rows'])

    def batch_rows(self,batch_id,start=0,count=10,sort_by='branch',descending=False):
        if not valid_int(start,0,1000000) or not valid_int(count,1,20) or type(descending)is not bool:raise ValueError('invalid page')
        if sort_by not in ('branch','executed_ticks','estimated_translation','estimated_rotation_abs'):raise ValueError('unsupported sort metric')
        rows=load(self._batch_path(batch_id)/'rows.json')
        def key(r):
            if sort_by in ('branch','executed_ticks'):return r[sort_by]
            d=r['observations']['estimated_object_delta']
            return None if d is None else math.hypot(*d[:2]) if sort_by=='estimated_translation' else abs(d[2])
        present=[r for r in rows if key(r) is not None];missing=[r for r in rows if key(r) is None]
        ordered=sorted(present,key=key,reverse=descending)+missing
        page=[];size=0
        for r in ordered[start:start+count]:
            d=r['observations']['estimated_object_delta']
            row=dict(branch=r['branch'],commands=r['commands'],status=r['status'],error=r['error'],executed_ticks=r['executed_ticks'],success=r.get('success'),estimated_object_delta=d,robot_net_displacement=r['observations']['robot_net_displacement'],events=r['observations']['event_count'],branch_start_row=r['branch_start_row'])
            n=len(json.dumps(row).encode())
            if size+n>self.settings['tool_reply_bytes']-2000:break
            page.append(row);size+=n
        return dict(batch_id=batch_id,total=len(rows),sort_by=sort_by,descending=descending,estimate_source='legal_sensor_estimate_not_truth',rows=page,next_start=start+len(page) if start+len(page)<len(rows) else None)

    def rpc(self,request,deadline):
        if request.get('method')=='run_branches':
            self._active();self.check_finalization(include_model=False)
            if self.state.get('finalizing'):raise ValueError('development closed')
            if set(request)!={'method','task','prefix','candidates'}:raise ValueError('unknown batch fields')
            self._charge('tool_calls');result=self.run_branches(request['task'],request['prefix'],request['candidates'],deadline=deadline)
            self.event('rpc',request,{'ok':True,'result':result});return result
        return super().rpc(request,deadline)

    def notebook(self,text):
        if not isinstance(text,str) or len(text.encode())>self.settings['notebook_bytes']:raise ValueError('notebook size limit')
        self.state['author_notebook']=text;self._save();return dict(bytes=len(text.encode()),source='author_written_not_host_inference')

    def _record(self,record_id):
        if not valid_int(record_id,0,self.state.get('tool_records',0)-1):raise ValueError('unknown tool record')
        return self.path/'private/tool_records'/f'{record_id:06d}.json'

    def read_result(self,record_id,start=0,count=8000):
        if not valid_int(start,0,1000000000) or not valid_int(count,1,self.settings['detail_bytes']):raise ValueError('invalid byte slice')
        data=self._record(record_id).read_bytes()
        if start<len(data) and data[start]&0xC0==0x80:raise ValueError('start must be a UTF-8 boundary; use returned next_start')
        end=min(start+count,len(data))
        while end<len(data) and end>start and data[end]&0xC0==0x80:end-=1
        if start<len(data) and end==start:raise ValueError('count cannot fit the next UTF-8 character')
        part=data[start:end]
        return dict(record_id=record_id,start=start,next_start=end if end<len(data) else None,bytes=len(data),text=part.decode('utf-8'))

    def list_results(self,start=0,count=20):
        if not valid_int(start,0,1000000) or not valid_int(count,1,50):raise ValueError('invalid page')
        total=self.state.get('tool_records',0);rows=[]
        for i in range(start,min(total,start+count)):
            r=load(self._record(i));v=r['response'];out=v.get('result') or {}
            rows.append(dict(record_id=i,tool=r['request']['name'],ok=v['ok'],error=v.get('error'),
                             test_id=out.get('test_id') if isinstance(out,dict) else None,
                             batch_id=out.get('batch_id') if isinstance(out,dict) else None))
        return dict(total=total,rows=rows,next_start=start+len(rows) if start+len(rows)<total else None)

    def tool(self,name,arguments):
        request=dict(name=name,arguments=arguments)
        try:
            self._active();self.check_finalization(include_model=False);self._charge('tool_calls')
            if self.state.get('finalizing') and name!='submit':raise ValueError('finalization phase: only submit is available')
            if not isinstance(arguments,dict):raise ValueError('tool arguments must be an object')
            names=('list_files','read_file','write_file','run_python','test_policy','read_log','submit',
                   'test_summary','test_events','read_trace','run_branches','batch_rows','notebook','read_result','list_results')
            if name not in names:raise ValueError('unknown tool')
            if name=='run_branches' and 'deadline' in arguments:raise ValueError('deadline is host-owned')
            result=dict(ok=True,result=getattr(self,name)(**arguments))
        except Exception as exc:result=dict(ok=False,error=dict(type=type(exc).__name__,message=str(exc)))
        # Archive complete public request/response before bounding context output.
        ident=self.state.get('tool_records',0);d=self.path/'private/tool_records';d.mkdir(exist_ok=True)
        save(d/f'{ident:06d}.json',dict(request=request,response=result));self.state['tool_records']=ident+1;self._save()
        self.event('tool',request,dict(result,record_id=ident))
        out=dict(result,record_id=ident)
        if len(json.dumps(out,ensure_ascii=False).encode())>self.settings['tool_reply_bytes']-1800:
            out=dict(ok=result['ok'],record_id=ident,archived=True,bytes=(d/f'{ident:06d}.json').stat().st_size,
                     message='Full public result saved; use read_result(record_id,start,count) or summary/detail tools.')
            if not result['ok']:out['error']=dict(type=result['error']['type'],message=result['error']['message'][:300])
        b=self.budget_status();out['budget']=dict(remaining=b['remaining'],phase=b['phase'],accounting=b['accounting'])
        return out

    def submit(self,files,contents=None):
        p=verify_profile(self.path);r=super().submit(files,contents)
        r['experiment_profile_sha256']=seal.digest(self.path/'experiment_profile.json');r['experiment_release_sha256']=p['release_sha256']
        save(self.path/'frozen/manifest.json',r);return r


class FreeSession(ExperimentMixin,old.FreeSession):
    @classmethod
    def create(cls,path,config,**kwargs):return create_session(path,config,method='free',**kwargs)

class SearchSession(ExperimentMixin,old.SearchSession):
    @classmethod
    def create(cls,path,config,**kwargs):return create_session(path,config,method='option_search_v3',**kwargs)


def create_session(path,config,*,method='option_search_v3',**kwargs):
    release.verify();s=old.create_session(path,config,method=method,**kwargs)
    try:
        for n in ('experiment_api.md','batch_client.py'):(s.workspace/n).write_bytes((release.HERE/n).read_bytes())
        save(s.workspace/'experiment_profile.json',DEFAULT_PROFILE)
        p=dict(release_sha256=seal.digest(release.LOCK),settings=deepcopy(DEFAULT_PROFILE),
               materials={n:seal.digest(s.workspace/n) for n in ('experiment_api.md','experiment_profile.json','batch_client.py')})
        save(s.path/'experiment_profile.json',p);s.state['experiment_profile_sha256']=seal.digest(s.path/'experiment_profile.json');s._save()
        s.event('experiment_profile_installed',{},p)
    finally:s.close()
    return open_session(path,**({'world_factory':kwargs['world_factory']} if 'world_factory' in kwargs else {}))


def open_session(path,**kwargs):
    verify_profile(path);p=old.knowledge.verify_session(path)
    return (FreeSession if p['method']=='free' else SearchSession)(path,**kwargs)


def verify_frozen(path):
    p=verify_profile(path);m=old.verify_frozen(path)
    if m.get('experiment_profile_sha256')!=seal.digest(Path(path)/'experiment_profile.json') or m.get('experiment_release_sha256')!=p['release_sha256']:raise ValueError('submitted experiment provenance mismatch')
    return m
