"""Heuristic-only authoring over the immutable v1.4.0 accounting/runtime."""
from pathlib import Path
import json
import shutil
import tempfile
import time
from dataclasses import asdict
from experiments.planar_tasks.baseline import session as base
from experiments.planar_tasks.baseline import seal
from experiments.planar_tasks.interface import TaskEnvironment
from .runner import RunBudget,run_episode,write_json
from experiments.planar_tasks.tasks import task_catalog
from . import build
from .search import Budget


class Session(base.Session):
    @classmethod
    def create(cls,path,config,*,planner_budget=None,world_factory=TaskEnvironment):
        build.verify()
        if config.get('prediction')!={'version':'nominal-v1','runtime':True}:
            raise ValueError('fixed-search authoring requires nominal-v1 runtime prediction')
        budget=planner_budget or Budget()
        if not isinstance(budget,Budget):raise ValueError('host must supply a validated planner Budget')
        session=base.Session.create(path,config,world_factory=world_factory)
        try:
            profile=dict(kind='option-search-v3',planner_budget=asdict(budget),
                         framework_release_sha256=seal.digest(build.LOCK))
            write_json(session.path/'framework.json',profile)
            session.state['framework_profile_sha256']=seal.digest(session.path/'framework.json');session._save()
            for name,source in [('heuristic.py',build.HERE/'heuristic.py'),('instructions.txt',build.HERE/'prompt.txt'),('heuristic_api.md',build.HERE/'README.md')]:
                shutil.copyfile(source,session.workspace/name)
            write_json(session.workspace/'framework.json',profile)
            packet=base.read_json(session.workspace/'contract.json')
            packet['policy_interface']={'module':'heuristic.py','functions':['rank_actions(parent,candidates,memory)','evaluate_state(node,memory)'],
                'returns':'rank_actions -> {costs:[finite total remaining seconds per candidate], proposals:[{action,cost}]}; evaluate_state -> finite remaining seconds',
                'priority':'candidate cost at its current parent; no extra prefix/time cost before prediction',
                'output_selection':'predicted edge time + backed-up successor value; unpredicted real roots retain explicit prior costs'}
            write_json(session.workspace/'contract.json',packet)
        finally:session.close()
        return cls(path,world_factory=world_factory)

    def __init__(self,path,*,world_factory=TaskEnvironment):
        build.verify()
        super().__init__(path,world_factory=world_factory)
        try:self._verify_framework()
        except Exception:self.close();raise

    def _verify_framework(self):
        build.verify()
        if seal.digest(self.path/'framework.json')!=self.state['framework_profile_sha256']:
            raise ValueError('host framework profile changed')
        self.profile=base.read_json(self.path/'framework.json')
        if self.profile['framework_release_sha256']!=seal.digest(build.LOCK):raise ValueError('framework release changed')
        self.planner_budget=Budget(**self.profile['planner_budget'])

    def _active(self):
        super()._active();self._verify_framework()

    def budget_status(self):
        result=super().budget_status()
        if hasattr(self,'profile'):result['fixed_planner']=self.profile
        result['prediction_usage']=dict(self.state.get('prediction_usage',{}))
        from experiments.planar_tasks.baseline.prediction import DEVELOPMENT_LIMITS
        result['prediction_limits']=dict(DEVELOPMENT_LIMITS)
        return result

    def write_file(self,path,text):
        base.safe_path(self.workspace,path)
        allowed=path=='heuristic.py' or (path.startswith('experiments/') and path.endswith('.py')) or (path.startswith('notes/') and path.endswith('.txt'))
        if not allowed:raise ValueError('editable: heuristic.py, experiments/*.py, notes/*.txt only')
        return super().write_file(path,text)

    def run_python(self,path,seconds):
        if not isinstance(path,str) or not path.startswith('experiments/') or not path.endswith('.py'):
            raise ValueError('development script must be experiments/*.py')
        return super().run_python(path,seconds)

    def _package(self,files,target,contents=None):
        self._verify_framework()
        if files!=['heuristic.py']:raise ValueError('deployment files must be exactly [heuristic.py]')
        if contents is not None and (not isinstance(contents,dict) or set(contents)-{'heuristic.py'} or any(not isinstance(v,str) for v in contents.values())):
            raise ValueError('inline contents may only contain heuristic.py text')
        if contents and 'heuristic.py' in contents:text=contents['heuristic.py']
        else:
            source=base.safe_path(self.workspace,'heuristic.py')
            if not source.is_file():raise ValueError('heuristic.py required')
            text=source.read_text()
        if len(text.encode())>min(self.budget.output_bytes,self.budget.workspace_bytes):raise ValueError('heuristic submission size limit')
        target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
        # Author workspace may be modified by experiment scripts. Fixed files are
        # always rebuilt from the frozen host sources, never copied from workspace.
        with tempfile.TemporaryDirectory(dir=target.parent) as temp:
            source=Path(temp)/'heuristic.py';source.write_text(text)
            build.build_package(target,source,self.planner_budget)
        build.verify_package(target)
        return {p.name:seal.digest(p) for p in target.iterdir() if p.is_file()}

    def test_policy(self,files,task,max_ticks):
        self._verify_framework()
        if files!=['heuristic.py']:raise ValueError('test files must be exactly [heuristic.py]')
        if task not in self.config['development_tasks']:raise ValueError('fixture unavailable')
        if type(max_ticks)is not int or not 1<=max_ticks<=1800:raise ValueError('max_ticks must be 1..1800')
        self._charge('policy_tests');self._charge('resets');self._charge('control_ticks',max_ticks)
        index=len(self.state['tests']);test_id=f'test_{index:04d}'
        self.state['tests'].append(dict(id=test_id,task=task,status='starting',reserved_ticks=max_ticks));self._save()
        directory=self.path/'private'/test_id
        try:
            hashes=self._package(files,directory/'package')
            summary=run_episode(task_catalog()[task],base.package_command(directory/'package'),directory/'episode',
                RunBudget(1800,min(self.budget.episode_wall_seconds,self.experiment_wall()),self.budget.response_seconds),
                provenance=dict(generated_agent_baseline=True,method='option-search-v3',files_sha256=hashes,framework=self.profile),
                security_sandbox=dict(kind='bubblewrap',read_only_package=True,development_rpc=False),
                max_control_ticks=max_ticks,wall_deadline=time.monotonic()+self.experiment_wall(),
                prediction_profile=self.config['prediction'],prediction_ledger=self.predictor.ledger)
        except Exception as exc:
            self.state['tests'][index].update(status='infrastructure_error',success=False,task_cost_seconds=180.,error=dict(type=type(exc).__name__,message=str(exc)));self._save();raise
        public=base.public_summary(summary);public['process_exit']=summary.get('process_exit')
        result=dict(test_id=test_id,summary=public,files_sha256=hashes)
        self.state['tests'][index].update(status='finished',result=result);self._save()
        return result

    def submit(self,files,contents=None):
        result=super().submit(files,contents)
        result.update(version='option-search-v3',framework_profile_sha256=self.state['framework_profile_sha256'],
            framework_release_sha256=seal.digest(build.LOCK),
            validation='syntax, isolated Policy and both hook signatures; no physics/success guarantee')
        write_json(self.path/'frozen/manifest.json',result)
        validation=base.read_json(self.path/'frozen/validation.json')
        validation['check']='load_and_two_hook_signatures_only'
        write_json(self.path/'frozen/validation.json',validation)
        return result


def verify_frozen(path,*,release_path=seal.LOCK):
    if Path(release_path).resolve()!=seal.LOCK.resolve():raise ValueError('this framework requires the frozen v1.4.0 baseline')
    path=Path(path);build.verify()
    manifest=base.verify_frozen(path)
    if manifest.get('version')!='option-search-v3':raise ValueError('not a heuristic-only submission')
    if manifest.get('framework_release_sha256')!=seal.digest(build.LOCK):raise ValueError('submission framework release mismatch')
    if manifest.get('framework_profile_sha256')!=seal.digest(path/'framework.json'):raise ValueError('submission profile changed')
    profile=base.read_json(path/'framework.json')
    if profile['framework_release_sha256']!=seal.digest(build.LOCK):raise ValueError('profile release mismatch')
    package=build.verify_package(path/'frozen/package')
    if package['budget']!=profile['planner_budget']:raise ValueError('submitted budget changed')
    return manifest
