"""Durable seed → perturb → replay → repair → compare → fit development pipeline.

Author tools return legal evidence only. The read-only human UI has a separate
privileged replay route. No provider is called implicitly by this module.
"""
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import time
import uuid

from .contract import digest, canonical, History, validate_action
from . import pipeline_execution as X
from . import pipeline_semantics as F
from .evidence import Outcome
from . import dataset

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'relational-seed-repair-v1'
GUIDES = ['experiments/planar_tasks/authoring/option_guide.md',
          'experiments/planar_tasks/authoring_v2/README.md',
          'experiments/planar_tasks/authoring_v2/experiment_api.md',
          'experiments/planar_tasks/relational_search/option_guide.md',
          'experiments/planar_tasks/relational_search/pipeline_guide.md']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with temp.open('w') as f:
        f.write(canonical(value)+'\n'); f.flush()
        import os
        os.fsync(f.fileno())
    temp.replace(path)


def pack(path, value, *, compresslevel=9):
    # A closed file is only published after complete serialization.
    path=Path(path); tmp=path.with_suffix('.pending')
    with gzip.open(tmp,'wt',compresslevel=compresslevel) as f:f.write(canonical(value))
    tmp.replace(path)


def unpack(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def identifier(value):
    if not isinstance(value,str) or not value or len(value)>80 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise ValueError('invalid artifact ID')
    return value


def perturb(base, delta, bound):
    value=deepcopy(base)
    if not isinstance(delta,dict):raise ValueError('perturbation dictionary required')
    for key,amount in delta.items():
        parts=key.split('.')
        if (type(amount) not in (float,int) or not abs(amount)<=bound or len(parts)!=2
                or parts[1] not in ('x','y','angle')):
            raise ValueError('bounded position/angle perturbation required')
        who,field=parts
        if who=='object':body=value['object']
        elif who in ('robot0','robot1'):body=value['robots'][int(who[-1])]
        else:
            body=next((b for b in value['movable_objects'] if b['label']==who),None)
            if body is None:raise ValueError('unknown perturbation entity')
        body[field]+=amount
    return value


def expectation(spec, observation):
    kind=spec['kind']
    if kind=='action_legal':
        try:validate_action(spec['action'],observation);return True
        except ValueError:return False
    if kind=='entity_visible':
        return any(e['id']==spec['entity_id'] and e['points'] for e in observation['entities'])
    raise ValueError('expectation kind; use action_legal or entity_visible')


class Campaign:
    def __init__(self,path):
        self.path=Path(path).resolve()
        if read(self.path/'campaign.json')['version']!=VERSION:raise ValueError('campaign version')

    @classmethod
    def create(cls,path,*,fixture=None,sequence=None,limits=None,group='clear-seed-01',split='train',
               model=None,perturbation_bound=.35):
        if split not in ('train','validation','test'):raise ValueError('split')
        identifier(group)
        if type(perturbation_bound) not in (float,int) or not 0<perturbation_bound<=1:raise ValueError('perturbation bound')
        initial=deepcopy(X.seed_fixture() if fixture is None else fixture);X.fixture(initial)
        seq=X.validate_commands(X.commands(X.T.default_sequence()) if sequence is None else sequence)
        caps=dict(execution_ticks=16000,resets=50,attempts=50,rounds=12,fits=12,author_turns=40)
        if limits:
            if set(limits)-set(caps):raise ValueError('unknown budget')
            caps.update(limits)
        if any(type(v) is not int or v<0 for v in caps.values()):raise ValueError('integer budget')
        dest=Path(path);dest.mkdir(parents=True,exist_ok=False)
        (dest/'knowledge').mkdir();(dest/'sources').mkdir()
        sources={}
        paths=set(ROOT.glob('experiments/**/*.py'))|set(ROOT.glob('src/coimagine/**/*.py'))
        paths.add(ROOT/'src/coimagine/random_start/lidar_kernel.c')
        paths|={ROOT/p for p in GUIDES}
        paths|=set((Path(__file__).parent/'pipeline_static').glob('*'))
        for p in sorted(paths):
            rel=str(p.relative_to(ROOT)); data=p.read_bytes();sources[rel]=hashlib.sha256(data).hexdigest()
            target=dest/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        for index,g in enumerate(GUIDES):(dest/'knowledge'/f'{index}_{Path(g).name}').write_bytes((ROOT/g).read_bytes())
        save(dest/'sources.json',sources)
        save(dest/'fixture.json',initial)
        save(dest/'campaign.json',dict(version=VERSION,created=time.time(),group=group,split=split,
            model=model or dict(provider='none',name='engineering acceptance; no LLM calls'),
            limits=caps,used=dict.fromkeys(caps,0),status='development',rounds=[],fits=[],semantics=[],
            reservations={},source_hash=digest(sources),seed_sequence=seq,perturbation_bound=perturbation_bound,
            task_cost='success elapsed seconds; failure 2 * environment time budget; errors unknown',
            history_window=64,episode_max_ticks=1800,episode_wall_seconds=120,
            llm_usage=dict(requests=0 if model is None or model.get('provider')=='none' else None,
                input_tokens=None,output_tokens=None,provenance='no provider transport attached; native usage unmetered if used')))
        return cls(dest)

    @contextmanager
    def locked(self):
        with (self.path/'.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            self.state=read(self.path/'campaign.json')
            if self.state['status']=='frozen':raise ValueError('frozen campaign is read-only')
            yield

    def _save(self):save(self.path/'campaign.json',self.state)

    def verify_sources(self):
        for rel,sha in read(self.path/'sources.json').items():
            if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=sha:
                raise ValueError('campaign source changed; start a new version: '+rel)

    def _charge(self,**amounts):
        for k,v in amounts.items():
            if type(v) is not int or v<0 or self.state['used'][k]+v>self.state['limits'][k]:
                raise RuntimeError('budget exhausted: '+k)
        for k,v in amounts.items():self.state['used'][k]+=v
        self._save()

    def _round(self,rid):return read(self.path/'rounds'/identifier(rid)/'round.json')
    def _rpath(self,rid):return self.path/'rounds'/identifier(rid)
    def _apath(self,rid,aid):return self._rpath(rid)/'attempts'/identifier(aid)

    def new_round(self,delta=None,*,parent=None,expectations=None):
        with self.locked():
            seq=self.state['seed_sequence'];parent_attempt=None
            if parent:
                prior=self._round(parent)
                good=[a for a in prior['attempts'] if a['status']=='success']
                if not good:raise ValueError('parent needs a completed successful witness')
                parent_attempt=good[-1]['id']
                seq=read(self._apath(parent,parent_attempt)/'request.json')['commands']
            # Deltas are always relative to the campaign seed, so repeated local
            # repairs cannot silently wander beyond the declared neighborhood.
            value=perturb(read(self.path/'fixture.json'),delta or {},self.state['perturbation_bound'])
            s,boxes=X.fixture(value)
            from .environment import MultiObjectOptionEnv
            obs=MultiObjectOptionEnv(s,boxes,overrides=value['overrides']).public()
            ex=deepcopy(expectations or [])
            if len(ex)>100:raise ValueError('expectation limit')
            for x in ex:
                if type(x.get('boundary')) is not int or not 0<=x['boundary']<=len(seq):raise ValueError('expectation boundary')
                expectation(x,obs)
            self._charge(rounds=1,resets=1)
            rid=f'r{len(self.state["rounds"]):03d}';p=self._rpath(rid);p.mkdir(parents=True)
            save(p/'fixture.json',value);save(p/'initial_public.json',obs)
            save(p/'scene.json',X.scene_metadata(s,boxes))
            record=dict(id=rid,parent=parent,parent_attempt=parent_attempt,delta=delta or {},
                group=self.state['group'],split=self.state['split'],created=time.time(),status='ready',
                original_commands=seq,expectations=ex,expectation_provenance='declared before this round execution',
                attempts=[],author_turns=[],comparisons=[],fits=[],semantics=[])
            save(p/'round.json',record);self.state['rounds'].append(rid);self._save()
            return dict(round_id=rid,observation=obs,commands=seq,expectations=ex)

    def _execute(self,rid,sequence,kind,metadata=None):
        """Caller holds the campaign lock. Persist reservation before reset/step."""
        self.verify_sources()
        X.validate_commands(sequence);total=sum(a['ticks'] for a in sequence)
        if total>self.state['episode_max_ticks']:raise ValueError('episode tick cap')
        r=self._round(rid);aid=f'a{len(r["attempts"]):03d}'
        key=rid+'/'+aid
        self._charge(attempts=1,resets=1,execution_ticks=total)
        self.state['reservations'][key]=total;self._save()
        p=self._apath(rid,aid);p.mkdir(parents=True)
        save(p/'request.json',dict(commands=sequence,kind=kind,metadata=metadata or {}))
        item=dict(id=aid,kind=kind,status='running',reserved_ticks=total,created=time.time(),metadata=metadata or {})
        r['attempts'].append(item);r['status']='running';save(self._rpath(rid)/'round.json',r)
        try:
            def progress(boundary,tick):
                if boundary%10==0:save(p/'progress.json',dict(boundary=boundary,tick=tick,updated=time.time()))
            result=X.execute(read(self._rpath(rid)/'fixture.json'),sequence,
                             max_ticks=self.state['episode_max_ticks'],on_boundary=progress)
            pack(p/'public.json.gz',result['public_rows']);pack(p/'replay.json.gz',result['replay'])
            save(p/'events.json',result['events']);save(p/'summary.json',result['summary'])
            item.update(result['summary'])
            mismatches=[]
            # These expectations refer to the original command indexing.
            # A longer repair shifts indices; do not silently reinterpret them.
            item['expectations_checked']=kind=='original_replay'
            for e in (r['expectations'] if item['expectations_checked'] else []):
                k=e['boundary']
                if k>=len(result['public_rows']):continue
                o=result['public_rows'][k]['observation']
                actual=expectation(e,o)
                if not actual:mismatches.append(dict(boundary=k,tick=o['tick'],expectation=e,actual=actual))
            if item['error']:
                mismatches.append(dict(boundary=item['error']['command_index'],tick=item['ticks'],error=item['error']))
            item['mismatches']=mismatches
            item['detected_boundary']=min((m['boundary'] for m in mismatches),default=None)
            self.state['used']['execution_ticks']-=total-item['ticks']
            del self.state['reservations'][key];self._save()
        except Exception as exc:
            item.update(status='infrastructure_error',error=str(exc),reservation_retained=True)
            save(p/'summary.json',item)
            raise
        finally:
            r['status']='awaiting_author';save(self._rpath(rid)/'round.json',r)
        return deepcopy(item)

    def replay(self,rid):
        with self.locked():return self._execute(rid,self._round(rid)['original_commands'],'original_replay')

    def author_turn(self,rid,turn):
        """Store a public decision brief + structured proposal, never hidden reasoning."""
        allowed={'actor','model','summary','rationale','evidence','output','usage'}
        if not isinstance(turn,dict) or set(turn)-allowed or not {'actor','summary','output'}<=set(turn):
            raise ValueError('author turn schema')
        if turn['actor'] not in ('human','engineering','llm'):raise ValueError('author provenance')
        if not isinstance(turn['summary'],str) or not 0<len(turn['summary'])<=500:raise ValueError('one short summary required')
        if len(canonical(turn).encode())>65536:raise ValueError('author turn byte cap')
        if turn['actor']=='llm' and not turn.get('model'):raise ValueError('explicit model/settings required')
        with self.locked():
            r=self._round(rid);self._charge(author_turns=1)
            tid=f't{len(r["author_turns"]):03d}'
            entry=dict(id=tid,created=time.time(),**deepcopy(turn))
            save(self._rpath(rid)/'authors'/(tid+'.json'),entry)
            r['author_turns'].append(entry);save(self._rpath(rid)/'round.json',r)
            return dict(turn_id=tid)

    def repair(self,rid,*,attempt_id,boundary,suffix,turn_id):
        with self.locked():
            r=self._round(rid)
            if not any(t['id']==turn_id for t in r['author_turns']):raise ValueError('record decision brief first')
            old=self._apath(rid,attempt_id);seq=read(old/'request.json')['commands']
            rows=unpack(old/'public.json.gz');h=X.history_at(rows,boundary)
            suffix=X.validate_commands(suffix)
            # The intervention is before action boundary, even if detection was later.
            meta=dict(source_attempt=attempt_id,repair_boundary=boundary,turn_id=turn_id,
                      prefix_hash=digest(seq[:boundary]),origin_history_hash=h.sha256,
                      supervision_scope='whole repaired sequence; not a local action preference')
            result=self._execute(rid,seq[:boundary]+suffix,'repair',meta)
            new=unpack(self._apath(rid,result['id'])/'public.json.gz')
            if len(new)<=boundary or X.history_at(new,boundary).sha256!=h.sha256:
                raise RuntimeError('replayed intervention prefix differs; no matched repair evidence')
            return result

    def compare(self,rid,*,attempt_id,boundary,candidates,continuation):
        """Exactly one candidate action then the identical bounded suffix per branch."""
        with self.locked():
            r=self._round(rid);old=self._apath(rid,attempt_id)
            seq=read(old/'request.json')['commands'];rows=unpack(old/'public.json.gz')
            h=X.history_at(rows,boundary);suffix=X.validate_commands(continuation,True)
            if not isinstance(candidates,list) or not 2<=len(candidates)<=12:raise ValueError('2..12 candidates')
            for a in candidates:validate_action(a,h.events[-1]['observation'])
            if len({digest(a) for a in candidates})!=len(candidates):raise ValueError('duplicate candidates')
            # Same *total* allowed horizon; each branch must fit it before launching.
            maximum=max(sum(a['ticks'] for a in seq[:boundary])+a['ticks']+sum(b['ticks'] for b in suffix) for a in candidates)
            if maximum>self.state['episode_max_ticks']:raise ValueError('comparison episode cap')
            required=sum(sum(a['ticks'] for a in seq[:boundary])+c['ticks']+sum(a['ticks'] for a in suffix) for c in candidates)
            for key,n in [('execution_ticks',required),('resets',len(candidates)),('attempts',len(candidates))]:
                if self.state['used'][key]+n>self.state['limits'][key]:raise RuntimeError('comparison budget: '+key)
            cid=f'c{len(r["comparisons"]):03d}'
            c=dict(id=cid,source_attempt=attempt_id,boundary=boundary,origin=h.sha256,
                   continuation=suffix,continuation_hash=digest(suffix),prefix_hash=digest(seq[:boundary]),
                   candidates=candidates,branches=[],status='running',dataset=None)
            r['comparisons'].append(c);save(self._rpath(rid)/'round.json',r)
            outcomes=[]
            try:
                for a in candidates:
                    result=self._execute(rid,seq[:boundary]+[a]+suffix,'comparison',dict(comparison=cid))
                    branchrows=unpack(self._apath(rid,result['id'])/'public.json.gz')
                    matched=len(branchrows)>boundary and X.history_at(branchrows,boundary).sha256==h.sha256
                    status=result['status'] if matched else 'tool_error'
                    outcomes.append(Outcome(h.sha256,a,digest(suffix),dict(episode_max_ticks=self.state['episode_max_ticks']),
                                            'execution',status,result['task_cost_seconds'] if matched else None))
                    c['branches'].append(dict(attempt_id=result['id'],prefix_verified=matched,**result))
                    rr=self._round(rid);rr['comparisons'][-1]=c;save(self._rpath(rid)/'round.json',rr)
                records=[dataset.record(r['group'],r['split'],h,a,b) for i,a in enumerate(outcomes) for b in outcomes[i+1:]]
                file=self._rpath(rid)/'comparisons'/(cid+'.jsonl')
                c['dataset']=dataset.write(file,records)
                c['preference_pairs']=sum(x['comparison']['status']=='preference' for x in records)
                c['status']='complete'
            finally:
                if c['status']=='running':c['status']='interrupted'
                rr=self._round(rid);rr['comparisons'][-1]=c;save(self._rpath(rid)/'round.json',rr)
            return c

    def submit_semantics(self,rid,*,source,turn_id):
        with self.locked():
            self.verify_sources()
            r=self._round(rid)
            if not any(t['id']==turn_id for t in r['author_turns']):raise ValueError('record author brief first')
            h=History();h.append(read(self._rpath(rid)/'initial_public.json'))
            probe=F.evaluate(source,[(h,dict(target_id=None,a0=44,a1=44,ticks=4))])
            sid=f's{len(self.state["semantics"]):03d}'
            p=self.path/'semantics';p.mkdir(exist_ok=True)
            (p/(sid+'.py')).write_text(source)
            meta=dict(id=sid,round_id=rid,turn_id=turn_id,sha256=digest(source),names=probe['names'],
                      validation='isolated schema probe only; scientific adequacy untested')
            self.state['semantics'].append(meta);self._save()
            r['semantics'].append(sid);save(self._rpath(rid)/'round.json',r)
            return meta

    def fit(self,rid,semantic_id,*,comparison_refs=None):
        with self.locked():
            self.verify_sources()
            source=(self.path/'semantics'/(identifier(semantic_id)+'.py')).read_text()
            refs=comparison_refs
            if refs is None:
                refs=[(rr['id'],c['id']) for rr in (self._round(x) for x in self.state['rounds'])
                      for c in rr['comparisons'] if c['status']=='complete']
            if len(refs)!=len(set(tuple(r) for r in refs)):raise ValueError('duplicate comparison ref')
            rows=[]
            for r,c in refs:
                p=self._rpath(r)/'comparisons'/(identifier(c)+'.jsonl')
                rows.extend(json.loads(line) for line in p.read_text().splitlines())
            self._charge(fits=1)
            fid=f'f{len(self.state["fits"]):03d}'
            meta=dict(id=fid,round_id=rid,semantic_id=semantic_id,comparisons=refs,status='running')
            self.state['fits'].append(meta);self._save()
            try:
                result=F.fit_models(rows,source);result.update(meta,status='complete')
                save(self.path/'fits'/(fid+'.json'),result);meta['status']='complete'
            except Exception as exc:
                meta.update(status='error',error=str(exc));save(self.path/'fits'/(fid+'.json'),meta);raise
            finally:
                self._save()
                r=self._round(rid);r['fits'].append(fid);save(self._rpath(rid)/'round.json',r)
            return dict(fit_id=fid,training_pairs=result['training_pairs'],
                        models={k:v['metrics'] for k,v in result['models'].items()})

    def context(self,rid,*,attempt_id=None,boundary=None):
        """Whitelisted public context, NOT the human dashboard round/fixture object."""
        r=self._round(rid)
        result=dict(round_id=rid,commands=r['original_commands'],expectations=r['expectations'],
            author_turns=r['author_turns'],attempts=[{k:a[k] for k in ('id','kind','status','ticks','success','error','detected_boundary') if k in a} for a in r['attempts']],
            instruction='Read knowledge files; use short decision briefs. Request only needed trace boundaries. No truth/fixture access.')
        state=read(self.path/'campaign.json')
        result['fits']=state['fits'];result['semantic_versions']=state['semantics']
        if attempt_id:
            rows=unpack(self._apath(rid,attempt_id)/'public.json.gz')
            h=X.history_at(rows,len(rows)-1 if boundary is None else boundary)
            result.update(history=h.events,history_hash=h.sha256)
        else:result['observation']=read(self._rpath(rid)/'initial_public.json')
        return result

    def apply_ranker(self,rid,*,fit_id,attempt_id,boundary,candidates,continuation,condition='M'):
        """Bounded ranking-to-execution wiring check, not a complete search panel."""
        with self.locked():
            if not isinstance(candidates,list) or not 1<=len(candidates)<=128:raise ValueError('candidate batch 1..128')
            fitted=read(self.path/'fits'/(identifier(fit_id)+'.json'))
            source=(self.path/'semantics'/(identifier(fitted['semantic_id'])+'.py')).read_text()
            old=self._apath(rid,attempt_id);seq=read(old/'request.json')['commands']
            h=X.history_at(unpack(old/'public.json.gz'),boundary)
            scores=F.score_candidates(source,fitted,h,candidates,condition)
            winner=max(range(len(candidates)),key=lambda i:scores[i])
            suffix=X.validate_commands(continuation,True)
            meta=dict(fit_id=fit_id,condition=condition,source_attempt=attempt_id,boundary=boundary,
                      candidates=candidates,scores=scores,selected=winner,origin=h.sha256,
                      scope='ranked candidate plus supplied continuation; not full learned-search evaluation')
            return self._execute(rid,seq[:boundary]+[candidates[winner]]+suffix,'ranked_execution',meta)

    def test_plan(self,rid,*,source,turn_id):
        """One common function across fixtures; initial observation only, no replanning."""
        from .pipeline_plans import generate
        with self.locked():
            self.verify_sources();r=self._round(rid)
            if not any(t['id']==turn_id for t in r['author_turns']):raise ValueError('record author brief first')
            sequence=generate(source,read(self._rpath(rid)/'initial_public.json'),self.state['seed_sequence'])
            sha=digest(source);p=self.path/'plans';p.mkdir(exist_ok=True)
            target=p/(sha+'.py')
            if not target.exists():target.write_text(source)
            return self._execute(rid,sequence,'generated_plan',dict(plan_sha256=sha,turn_id=turn_id,
                scope='same initial-observation sequence generator; no online replanning'))

    def audit(self):
        with self.locked():
            # Missing settlement is not proof of zero work. Keep reservation charged.
            for rid in self.state['rounds']:
                r=self._round(rid)
                for a in r['attempts']:
                    if a['status']=='running':a.update(status='interrupted',reservation_retained=True)
                if r['status']=='running':r['status']='interrupted'
                save(self._rpath(rid)/'round.json',r)
            return dict(reservations=self.state['reservations'],used=self.state['used'])

    def freeze(self):
        with self.locked():
            if self.state['reservations']:raise ValueError('unsettled work; audit and resolve before freeze')
            self.verify_sources()
            self.state['status']='frozen';self._save()
            hashes={str(p.relative_to(self.path)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.path.rglob('*') if p.is_file() and p.name not in ('.lock','manifest.json')}
            save(self.path/'manifest.json',hashes)
        return self.verify_frozen()

    def verify_frozen(self):
        if read(self.path/'campaign.json')['status']!='frozen':raise ValueError('not frozen')
        hashes=read(self.path/'manifest.json')
        actual={str(p.relative_to(self.path)) for p in self.path.rglob('*') if p.is_file() and p.name not in ('.lock','manifest.json')}
        if actual!=set(hashes):raise ValueError('frozen artifact inventory changed')
        for rel,sha in hashes.items():
            p=self.path/rel
            if p.resolve().is_relative_to(self.path) is False or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
                raise ValueError('frozen artifact changed: '+rel)
        sources=read(self.path/'sources.json')
        if digest(sources)!=read(self.path/'campaign.json')['source_hash']:
            raise ValueError('source inventory hash mismatch')
        for rel,sha in sources.items():
            p=(self.path/'sources'/rel).resolve()
            if not p.is_relative_to(self.path/'sources') or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
                raise ValueError('source snapshot mismatch: '+rel)
        return dict(status='verified',files=len(hashes),sha256=digest(hashes))
