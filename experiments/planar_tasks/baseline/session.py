"""Host-owned autonomous tools; generated code sees only a workspace and legal RPC."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

from . import VERSION
from .config import validate_config
from . import seal
from .runtime import run_script
from ..interface import TaskEnvironment, contract, validate_action
from ..runner import RunBudget, run_episode, write_json
from ..policy import JsonlPolicy
from ..tasks import task_catalog

BASE=Path(__file__).resolve().parent
REAL_RUN_EPISODE=run_episode
RESERVED={'client.py','predictor_client.py','prediction_api.md','contract.json','campaign.json','instructions.txt','__policy_host__.py'}


def read_json(path):return json.loads(Path(path).read_text())


def safe_path(root,name):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or any(p in ('','..','.') for p in name.split('/')):
        raise ValueError('normalized workspace-relative path required')
    root=Path(root).resolve();path=root/name
    # Reject every symlink component, even one pointing back into the workspace.
    for p in (path,*path.parents):
        if p==root:break
        if p.is_symlink():raise ValueError('symlinks are not supported')
    if not path.resolve().is_relative_to(root):raise ValueError('workspace escape')
    return path


def public_summary(summary):
    allowed=('status','success','termination','error','attempted_episodes','successful_episodes',
             'decisions','executed_control_ticks','simulated_seconds','wall_seconds',
             'task_cost_seconds','cost_definition','prediction_usage')
    return {k:summary[k] for k in allowed if k in summary}


def legal_svg(observation):
    """Raw visible scan points, own/peer poses and public goal; no fitted geometry."""
    robots=observation['robots'];width,height=robots[0]['arena']
    if not all(math.isfinite(x) and x>0 for x in (width,height)):raise ValueError('invalid public arena')
    elements=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
              f'<rect width="{width}" height="{height}" fill="#10151c"/>',
              f'<g transform="translate(0 {height}) scale(1 -1)">']
    colours={1:'#49b9e8',2:'#a5aab3',3:'#de9251'}
    for r in robots:
        scan=r['scan'];ox,oy=scan['origin']
        for a,d,valid,category in zip(scan['ray_angles'],scan['distances'],scan['valid'],scan['category']):
            if not valid or d is None or category not in colours:continue
            x,y=ox+d*math.cos(a),oy+d*math.sin(a)
            elements.append(f'<circle cx="{x}" cy="{y}" r="0.012" fill="{colours[category]}"/>')
        own=r['own']
        elements.append(f'<circle cx="{own["x"]}" cy="{own["y"]}" r="0.035" fill="#ffffff"/>')
    gx,gy=robots[0]['goal_xy'];radius=robots[0]['goal_radius']
    elements.append(f'<circle cx="{gx}" cy="{gy}" r="{radius}" stroke="#af7ed0" stroke-width="0.015" fill="none"/>')
    return ''.join(elements)+ '</g></svg>'


def package_command(package,*,validate=False):
    # Same execution environment at development tests and deployment; no dev RPC.
    command=['prlimit','--as=536870912','--cpu=30','--fsize=16777216','--nofile=64','--',
             'bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--clearenv']
    for root in ('/usr','/lib','/lib64'):
        if Path(root).exists():command+=['--ro-bind',root,root]
    return command+['--proc','/proc','--dev','/dev','--tmpfs','/tmp',
        '--ro-bind',str(Path(package).resolve()),'/work','--chdir','/work',
        '--setenv','PYTHONDONTWRITEBYTECODE','1','/usr/bin/python3','-u','__policy_host__.py']+(['--validate'] if validate else [])


class Session:
    """One live synthesis session. Simulator state is intentionally not reloadable."""
    def __init__(self,path,*,release_path=seal.LOCK,world_factory=TaskEnvironment):
        self.path=Path(path).resolve();self.release_path=Path(release_path)
        self.release=seal.verify_release(self.release_path)
        self.config=read_json(self.path/'session.json');self.budget=validate_config(self.config)
        self.state=read_json(self.path/'state.json')
        if seal.digest(self.path/'session.json')!=self.state['config_sha256']:
            raise ValueError('session config changed')
        if seal.digest(self.release_path)!=self.state['release_sha256']:
            raise ValueError('session release changed')
        self.workspace=self.path/'workspace';self.world=None;self.episode=None
        self.python_run=None
        self.world_factory=world_factory
        self.predictor = None
        if self.config.get('prediction'):
            from .prediction import PredictionService, PredictionLedger, DEVELOPMENT_LIMITS
            ledger = PredictionLedger(DEVELOPMENT_LIMITS, self.state.setdefault('prediction_usage', {}), self._save)
            self.predictor = PredictionService(ledger=ledger, log_path=self.path/'private/prediction.jsonl')
        # Prevent two hosts with independent counters/worlds opening one session.
        self.owner=(self.path/'.owner').open('a')
        fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self._save()

    @classmethod
    def create(cls,path,config,*,release_path=seal.LOCK,world_factory=TaskEnvironment):
        release=seal.verify_release(release_path);validate_config(config)
        if set(config['development_tasks'])-set(release['development_tasks']):raise ValueError('unknown development fixture')
        path=Path(path).resolve();path.mkdir(parents=True,exist_ok=False)
        (path/'workspace').mkdir();(path/'private').mkdir()
        write_json(path/'session.json',config)
        state={'version':VERSION,'created_unix':time.time(),'started_monotonic':time.monotonic(),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'config_sha256':seal.digest(path/'session.json'),'release_sha256':seal.digest(release_path),
            'status':'developing','usage':{},'events':0,'last_event_hash':None,
            'episodes':[],'tests':[],'python_runs':0}
        write_json(path/'state.json',state)
        from coimagine.random_start.env import EnvConfig
        c=EnvConfig();packet=contract()
        packet['public_success_rule']={k:getattr(c,k) for k in ('success_distance','success_linear','success_angular','success_hold')}
        packet['policy_interface']={'module':'policy.py','class':'Policy','constructor':'Policy(contract)','method':'act(observation) -> {a0, a1, ticks}'}
        write_json(path/'workspace/contract.json',packet)
        write_json(path/'workspace/campaign.json',config)
        shutil.copyfile(BASE/'prompt.txt',path/'workspace/instructions.txt')
        shutil.copyfile(BASE/'client.py',path/'workspace/client.py')
        if config.get('prediction'):
            for name in ('predictor_client.py', 'prediction_api.md'):
                shutil.copyfile(BASE/name, path/'workspace'/name)
        return cls(path,release_path=release_path,world_factory=world_factory)

    def close(self):
        if getattr(self,'owner',None):
            self.owner.close();self.owner=None

    def _save(self):
        temp=self.path/'state.tmp';write_json(temp,self.state);temp.replace(self.path/'state.json')

    def _active(self):
        if not self.owner:raise ValueError('session host is closed')
        seal.verify_release(self.release_path)
        if seal.digest(self.release_path)!=self.state['release_sha256']:
            raise ValueError('session release changed')
        if seal.digest(self.path/'session.json')!=self.state['config_sha256']:raise ValueError('session config changed')
        if self.state['status']!='developing':raise ValueError('session is closed or submitted')
        if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=self.state['boot_id']:
            raise ValueError('host reboot invalidates session deadline; no automatic resume')
        if self.available_wall()<=0:
            raise TimeoutError('session wall budget exhausted')

    def remaining_wall(self):
        return max(0.,self.budget.wall_seconds-(time.monotonic()-self.state['started_monotonic']))

    def budget_status(self):
        limits=self.budget.as_dict();used=dict(self.state['usage'])
        used['wall_seconds']=self.budget.wall_seconds-self.remaining_wall()
        cumulative=('wall_seconds','tool_calls','model_calls','model_input_bytes','model_output_tokens',
                    'python_total_seconds','resets','control_ticks','policy_tests')
        status={'limits':limits,'used':used,'remaining':{k:max(0,limits[k]-used.get(k,0)) for k in cumulative},
                'finalization':self.config.get('finalization',{}),'finalizing':self.state.get('finalizing',False),
                'accounting':'output tokens, Python seconds and policy-test ticks are reservations; no refund',
                'reported_model_usage':self.state.get('reported_model_usage',[])}
        status['phase']='submission' if self.state.get('finalizing') else 'development'
        if self.state.get('finalizing'):
            phase_used={k:max(0,used.get(k,0)-self.state['finalization_start_usage'].get(k,0))
                        for k in self.config.get('finalization',{})}
            phase_used['wall_seconds']=time.monotonic()-self.state['finalization_started_monotonic']
            status['phase_used']=phase_used
            status['phase_remaining']={k:max(0,min(status['remaining'][k],v-phase_used[k]))
                                       for k,v in self.config.get('finalization',{}).items()}
        return status

    def available_wall(self):
        remaining=self.remaining_wall()
        if self.state.get('finalizing'):
            remaining=min(remaining,self.config['finalization']['wall_seconds']-
                          (time.monotonic()-self.state['finalization_started_monotonic']))
        return max(0.,remaining)

    def available(self,key):
        status=self.budget_status()
        return status.get('phase_remaining',{}).get(key,status['remaining'][key])

    def finalize(self,reason):
        if not self.state.get('finalizing'):
            self.state['finalizing']=True;self.state['finalization_reason']=reason
            self.state['finalization_start_usage']=dict(self.state['usage'])
            self.state['finalization_started_monotonic']=time.monotonic();self._save()
            self.event('finalization',{'reason':reason},{'budget':self.budget_status()})

    def check_finalization(self,*,include_model=True):
        reserve=self.config.get('finalization',{})
        remaining=self.budget_status()['remaining']
        keys=reserve if include_model else ('tool_calls','wall_seconds')
        if reserve and any(remaining[k]<=reserve[k] for k in keys):
            self.finalize('predeclared reserve reached')

    def experiment_wall(self):
        return max(0.,self.remaining_wall()-self.config.get('finalization',{}).get('wall_seconds',0))

    def _charge(self,key,value=1):
        used=self.state['usage'].get(key,0)
        if self.state.get('finalizing') and key in self.config.get('finalization',{}) and value>self.available(key):
            self.state['status']='resource_exhausted';self.state['stop_reason']='submission '+key+' budget exhausted';self._save()
            raise ValueError(self.state['stop_reason'])
        if used+value>getattr(self.budget,key):
            if self.config.get('finalization') and key in ('resets','control_ticks','policy_tests','python_total_seconds'):
                self.finalize(key+' budget exhausted')
            else:
                self.state['status']='resource_exhausted';self.state['stop_reason']=key+' budget exhausted';self._save()
            raise ValueError(key+' budget exhausted')
        self.state['usage'][key]=used+value;self._save()

    def event(self,kind,request,result):
        row={'index':self.state['events'],'kind':kind,'request':request,'result':result,
             'time_unix':time.time(),'previous_sha256':self.state['last_event_hash']}
        data=json.dumps(row,sort_keys=True,allow_nan=False)
        with (self.path/'private/events.jsonl').open('a') as stream:stream.write(data+'\n')
        if kind=='rpc' and self.python_run is not None:
            with (self.path/'private'/self.python_run/'rpc.jsonl').open('a') as stream:stream.write(data+'\n')
        self.state['events']+=1;self.state['last_event_hash']=hashlib.sha256(data.encode()).hexdigest();self._save()

    def tool(self,name,arguments):
        request={'name':name,'arguments':arguments}
        try:
            self._active();self.check_finalization(include_model=False);self._charge('tool_calls')
            if self.state.get('finalizing') and name!='submit':
                raise ValueError('finalization phase: only submit(files, contents) is available')
            if not isinstance(arguments,dict):raise ValueError('tool arguments must be an object')
            dispatch={'list_files':self.list_files,'read_file':self.read_file,'write_file':self.write_file,
                'run_python':self.run_python,'test_policy':self.test_policy,'read_log':self.read_log,'submit':self.submit}
            if name not in dispatch:raise ValueError('unknown tool')
            result={'ok':True,'result':dispatch[name](**arguments)}
        except Exception as exc:
            result={'ok':False,'error':{'type':type(exc).__name__,'message':str(exc)}}
        result['budget']=self.budget_status()
        self.event('tool',request,result)
        return result

    def list_files(self):
        return [str(p.relative_to(self.workspace)) for p in sorted(self.workspace.rglob('*')) if p.is_file() and not p.is_symlink()]

    def read_file(self,path,start=0,limit=16000):
        if type(start)is not int or start<0 or type(limit)is not int or not 1<=limit<=64000:raise ValueError('invalid read range')
        p=safe_path(self.workspace,path)
        if not p.is_file():raise ValueError('regular file required')
        with p.open('rb') as f:f.seek(start);data=f.read(limit)
        return {'text':data.decode('utf-8',errors='replace'),'start':start,'bytes':len(data),'size':p.stat().st_size}

    def workspace_size(self):
        return sum(p.lstat().st_size for p in self.workspace.rglob('*') if p.is_file() and not p.is_symlink())

    def write_file(self,path,text):
        if not isinstance(text,str) or len(text.encode())>self.budget.output_bytes:raise ValueError('file write size/type limit')
        if path in RESERVED:raise ValueError('reserved common interface file')
        p=safe_path(self.workspace,path);old=p.stat().st_size if p.exists() else 0
        if p.exists() and not p.is_file():raise ValueError('regular file required')
        if self.workspace_size()-old+len(text.encode())>self.budget.workspace_bytes:raise ValueError('workspace size budget exhausted')
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
        return {'path':path,'sha256':seal.digest(p)}

    def _result(self):
        env=self.world._env;terminal=env.termination is not None
        return {'observation':self.world.observe(),'terminated':terminal,
            'termination':env.termination,'success':bool(env.success) if terminal else None}

    def rpc(self,request,deadline):
        try:
            self._active();self.check_finalization(include_model=False)
            if self.state.get('finalizing'):raise ValueError('finalization phase: simulator tools closed')
            self._charge('tool_calls')
            if time.monotonic()>=deadline:raise TimeoutError('script deadline exceeded')
            method=request.get('method')
            if method == 'predict':
                if self.predictor is None:
                    raise ValueError('prediction profile disabled')
                self.predictor.begin_decision()  # development RPC: one request per window
                result = self.predictor.rpc(request, deadline)
                self.event('rpc', request, {'ok': True, 'result': result})
                return result
            required={'reset':{'method','task'},'step':{'method','action'},'observe':{'method'},'render':{'method'}}
            if method not in required or set(request)!=required[method]:raise ValueError('unknown RPC or fields')
            if method=='reset':
                name=request['task']
                if name not in self.config['development_tasks']:raise ValueError('fixture unavailable')
                if self.config['purpose']=='infrastructure_smoke' and self.world_factory is TaskEnvironment:
                    raise ValueError('smoke profile forbids real simulator creation')
                self._charge('resets')
                if self.episode is not None:
                    previous=self.state['episodes'][self.episode]
                    if previous['status']=='active':previous['status']='abandoned_by_agent'
                self.episode=len(self.state['episodes'])
                self.state['episodes'].append({'task':name,'status':'starting','ticks':0});self._save()
                self.world=None
                try:
                    self.world=self.world_factory(task_catalog()[name].instance)
                except Exception:
                    self.state['episodes'][-1]['status']='initialization_error';self._save()
                    raise
                self.state['episodes'][-1]['status']='active';self._save()
                result=self._result()
            else:
                if self.world is None:raise ValueError('reset required; no episode restored after host restart')
                if method=='step':
                    if self.world._env.termination is not None:raise ValueError('episode terminated; reset required')
                    a0,a1,ticks=validate_action(request['action'],self.world.observe())
                    for _ in range(ticks):
                        if time.monotonic()>=deadline:raise TimeoutError('script deadline exceeded during step')
                        if self.remaining_wall()<=0:raise TimeoutError('session wall budget exhausted')
                        self._charge('control_ticks')
                        self.world._env.joint_step(a0,a1,record=True)
                        self.state['episodes'][self.episode]['ticks']+=1
                        if self.world._env.termination is not None:break
                    self.world.decisions+=1;self.world._env.skills_used=self.world.decisions
                    if self.world._env.termination is not None:
                        self.state['episodes'][self.episode].update(status='complete',success=bool(self.world._env.success),termination=self.world._env.termination)
                    self._save()
                result=legal_svg(self.world.observe()) if method=='render' else self._result()
            self.event('rpc',request,{'ok':True,'result':result})
            return result
        except Exception as exc:
            self.event('rpc',request,{'ok':False,'error':{'type':type(exc).__name__,'message':str(exc)}})
            raise

    def run_python(self,path,seconds):
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:raise ValueError('positive finite seconds required')
        left=self.budget.python_total_seconds-self.state['usage'].get('python_total_seconds',0)
        timeout=min(seconds,self.budget.python_call_seconds,left,self.experiment_wall())
        if timeout<=0:raise ValueError('Python time budget exhausted')
        # Reserve whole timeout before execution; a crash cannot avoid accounting.
        self._charge('python_total_seconds',timeout)
        index=self.state['python_runs'];self.state['python_runs']+=1;self._save()
        snapshot=self.path/'private'/f'python_{index:04d}'
        snapshot.mkdir()
        self._copy_files(self.list_files(),snapshot/'workspace',allow_common=True)
        deadline=time.monotonic()+timeout
        self.python_run=f'python_{index:04d}'
        try:
            result=run_script(self.workspace,path,lambda req:self.rpc(req,deadline),timeout,self.budget.output_bytes)
            result['run_id']=self.python_run
        finally:
            self.python_run=None
        if self.workspace_size()>self.budget.workspace_bytes:
            result['status']='workspace_limit';self.state['status']='resource_exhausted';self._save()
        write_json(snapshot/'result.json',result)
        return result

    def _copy_files(self,files,target,allow_common=False):
        if not isinstance(files,list) or not files or not all(isinstance(n,str) for n in files) or len(set(files))!=len(files):raise ValueError('distinct file list required')
        target.mkdir(parents=True,exist_ok=False);hashes={}
        for name in files:
            if not allow_common and (name in RESERVED or name.split('/')[0].startswith('__')):raise ValueError('reserved file cannot be submitted')
            source=safe_path(self.workspace,name)
            if not source.is_file():raise ValueError('regular file required')
            dest=safe_path(target,name);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest);hashes[name]=seal.digest(dest)
        return hashes

    def _package(self,files,target,contents=None):
        if 'policy.py' not in files:raise ValueError('policy.py required')
        if contents is None:
            hashes=self._copy_files(files,target)
        else:
            if not isinstance(files,list) or not files or not all(isinstance(n,str) for n in files) or len(set(files))!=len(files):raise ValueError('distinct file list required')
            if not isinstance(contents,dict) or set(contents)-set(files) or not all(isinstance(v,str) for v in contents.values()):
                raise ValueError('contents must map listed filenames to UTF-8 text')
            if sum(len(v.encode()) for v in contents.values())>self.budget.output_bytes:raise ValueError('inline submission size limit')
            target.mkdir(parents=True,exist_ok=False);hashes={};total=0
            for name in files:
                if name in RESERVED or name.split('/')[0].startswith('__'):raise ValueError('reserved file cannot be submitted')
                dest=safe_path(target,name);dest.parent.mkdir(parents=True,exist_ok=True)
                if name in contents:dest.write_text(contents[name])
                else:
                    source=safe_path(self.workspace,name)
                    if not source.is_file():raise ValueError('regular file required')
                    shutil.copyfile(source,dest)
                total+=dest.stat().st_size
                if total>self.budget.workspace_bytes:raise ValueError('submission size limit')
                hashes[name]=seal.digest(dest)
        shutil.copyfile(BASE/'policy_host.py',target/'__policy_host__.py')
        hashes['__policy_host__.py']=seal.digest(target/'__policy_host__.py')
        if self.config.get('prediction'):
            shutil.copyfile(BASE/'predictor_client.py', target/'predictor_client.py')
            hashes['predictor_client.py'] = seal.digest(target/'predictor_client.py')
        return hashes

    def test_policy(self,files,task,max_ticks):
        if task not in self.config['development_tasks']:raise ValueError('fixture unavailable')
        if self.config['purpose']=='infrastructure_smoke' and run_episode is REAL_RUN_EPISODE:
            raise ValueError('smoke profile forbids real physics tests')
        if type(max_ticks)is not int or not 1<=max_ticks<=1800:raise ValueError('max_ticks must be 1..1800')
        self._charge('policy_tests');self._charge('resets')
        self._charge('control_ticks',max_ticks)  # reservation includes failed tests
        index=len(self.state['tests']);test_id=f'test_{index:04d}'
        self.state['tests'].append({'id':test_id,'task':task,'status':'starting','reserved_ticks':max_ticks});self._save()
        directory=self.path/'private'/test_id
        try:
            hashes=self._package(files,directory/'package')
            # Tick reservation also bounds a requested multi-tick batch.
            summary=run_episode(task_catalog()[task],package_command(directory/'package'),directory/'episode',
                RunBudget(1800,min(self.budget.episode_wall_seconds,self.experiment_wall()),self.budget.response_seconds),
                provenance={'generated_agent_baseline':True,'method':VERSION,'files_sha256':hashes},
                security_sandbox={'kind':'bubblewrap','read_only_package':True,'development_rpc':False},
                max_control_ticks=max_ticks,wall_deadline=time.monotonic()+self.experiment_wall(),
                **({'prediction_profile': self.config['prediction'],
                    'prediction_ledger': self.predictor.ledger} if self.predictor else {}))
        except Exception as exc:
            self.state['tests'][index].update(status='infrastructure_error',success=False,
                task_cost_seconds=180.,error={'type':type(exc).__name__,'message':str(exc)})
            self._save()
            raise
        result={'test_id':test_id,'summary':public_summary(summary),'files_sha256':hashes}
        self.state['tests'][index].update(status='finished',result=result);self._save()
        return result

    def read_log(self,test_id,stream,start=0,count=10):
        if type(start)is not int or start<0 or type(count)is not int or not 1<=count<=50:raise ValueError('invalid log range')
        if test_id in {f'python_{i:04d}' for i in range(self.state['python_runs'])}:
            directory=self.path/'private'/test_id
            if stream in ('stdout','stderr'):
                return {'text':read_json(directory/'result.json')[stream][start:start+min(count*1024,self.budget.output_bytes)]}
            if stream!='rpc':raise ValueError('Python log stream must be rpc/stdout/stderr')
            from itertools import islice
            logfile=directory/'rpc.jsonl'
            if not logfile.exists():return []
            with logfile.open() as f:rows=[json.loads(line) for line in islice(f,start,start+count)]
            if len(json.dumps(rows).encode())>self.budget.output_bytes:raise ValueError('log slice too large; reduce count')
            return rows
        if test_id not in {t['id'] for t in self.state['tests']}:raise ValueError('unknown test')
        if stream not in ('sensors','stderr','decisions'):raise ValueError('log stream unavailable')
        directory=self.path/'private'/test_id/'episode'
        if stream=='stderr':
            with (directory/'policy.stderr').open('rb') as f:f.seek(start);data=f.read(min(count*1024,self.budget.output_bytes))
            return {'text':data.decode(errors='replace')}
        if stream=='decisions':return read_json(directory/'decisions.json')[start:start+count]
        from itertools import islice
        with (directory/'sensors.jsonl').open() as f:rows=[json.loads(line) for line in islice(f,start,start+count)]
        if len(json.dumps(rows).encode())>self.budget.output_bytes:raise ValueError('log slice too large; reduce count')
        return rows

    def submit(self,files,contents=None):
        index=self.state.get('submission_attempts',0);self.state['submission_attempts']=index+1;self._save()
        staging=self.path/'private'/f'submission_{index:06d}'
        hashes=self._package(files,staging/'package',contents)
        for name in hashes:
            if name.endswith('.py'):compile(safe_path(staging/'package',name).read_bytes(),name,'exec')
        timeout=min(self.budget.response_seconds,self.available_wall())
        if timeout<=0:raise TimeoutError('no submission validation time remains')
        validator=JsonlPolicy(package_command(staging/'package',validate=True),staging/'validation.stderr')
        try:
            checked=validator.request({'contract':contract()},timeout)
            if checked!={'validation':'load_and_interface_only'}:raise ValueError('invalid interface validation response')
        except Exception as exc:
            validator.close()
            detail=(staging/'validation.stderr').read_bytes()[-4000:].decode(errors='replace')
            write_json(staging/'validation.json',{'status':'failed','error':type(exc).__name__,'stderr':detail})
            raise ValueError('submission interface validation failed: '+type(exc).__name__+'\n'+detail) from None
        else:validator.close()
        write_json(staging/'validation.json',{'status':'passed','check':'load_and_interface_only','physics_evaluated':False})
        manifest={'version':VERSION,'files_sha256':hashes,'release_sha256':self.state['release_sha256'],
                  'config_sha256':self.state['config_sha256'],'submitted_unix':time.time(),'selection':'agent_explicit_submission',
                  'validation':'syntax, isolated Policy(contract) construction and act signature; no physics or success guarantee'}
        write_json(staging/'manifest.json',manifest)
        staging.rename(self.path/'frozen')
        self.state['status']='submitted'
        self.state['finished_monotonic']=time.monotonic();self.state['finished_unix']=time.time();self._save()
        return manifest


def verify_frozen(path,release_path=seal.LOCK):
    """Evaluation entry gate. Never exposes a development RPC or selects a policy."""
    seal.verify_release(release_path)
    path=Path(path);manifest=read_json(path/'frozen/manifest.json')
    if manifest['release_sha256']!=seal.digest(release_path):raise ValueError('frozen release mismatch')
    if manifest['config_sha256']!=seal.digest(path/'session.json'):raise ValueError('frozen campaign mismatch')
    package=path/'frozen/package'
    files={str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    if files!=set(manifest['files_sha256']):raise ValueError('frozen file set changed')
    for name,expected in manifest['files_sha256'].items():
        if seal.digest(safe_path(package,name))!=expected:raise ValueError('frozen file hash changed: '+name)
    return manifest
