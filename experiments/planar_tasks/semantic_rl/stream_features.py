"""Incremental semantic runtime: one new legal event, bounded explicit memory.

Host retains compact vectors/geometry, never reloads old observation windows.
The broker can be restarted without losing state; all author state is explicit.
"""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import pickle
import select
import os
import struct
import subprocess
import tempfile
import time
from .fast_features import FastProgram
from .stream_history import StreamHistory
from .typed_compiler import compile_source, stream_config
from experiments.planar_tasks.sandbox import policy_command
from experiments.planar_tasks.relational_search.contract import canonical, digest, validate_action


class StreamProgram(FastProgram):
    def __init__(self,source,max_names=64,timeout=30.):
        super().__init__(source,max_names,timeout,typed=True)
        self.protocol=stream_config(source)
        self.names=next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign)
                        and any(isinstance(t,ast.Name) and t.id=='NAMES' for t in n.targets))
        self.reset()

    def reset(self):
        self.episode=None;self.index=-1;self.memory=None;self.frames=[];self.context=None
        self.history_hash=None;self.observation=None;self.prepared_for=None
        self.update_count=0
        self.action_index=None

    def start(self):
        self.close()
        self.directory=tempfile.TemporaryDirectory(prefix='coimagine_stream_')
        path=Path(self.directory.name)/'worker.py'
        directory,self.compilation=compile_source(self.source)
        base=Path(__file__).with_name('feature_worker.py').read_text()
        path.write_text(base+'\n'+Path(__file__).with_name('stream_worker.py').read_text()+
                        '\nstream_serve("/work/semantics.so")\n')
        self.process=subprocess.Popen(policy_command(path,artifacts=((directory/'semantics.so','semantics.so'),)),
             stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,start_new_session=True,bufsize=0)

    def request(self,value):
        if self.process is None or self.process.poll() is not None or self.calls%256==0:self.start()
        self.calls+=1
        packet=pickle.dumps(value,protocol=4)
        deadline=time.monotonic()+self.timeout
        try:
            fd=self.process.stdin.fileno();os.set_blocking(fd,False)
            remaining=memoryview(struct.pack('!I',len(packet))+packet)
            while remaining:
                wait=deadline-time.monotonic()
                if wait<=0 or not select.select([],[fd],[],wait)[1]:raise TimeoutError('stream write deadline')
                remaining=remaining[os.write(fd,remaining):]
            size=struct.unpack('!I',self.read(4,deadline))[0]
            if size>1024*1024:raise ValueError('stream reply limit')
            result=json.loads(self.read(size,deadline))
            if 'error' in result:raise ValueError('stream worker: '+result['error'])
            return result['result']
        except BaseException:
            self.close();raise

    def prepare(self,history):
        if not isinstance(history,StreamHistory):raise ValueError('stream requires StreamHistory; replay from episode start or restore snapshot')
        if history.episode!=self.episode:
            if history.index!=0:raise ValueError('missing stream prefix; replay or restore memory snapshot')
            self.reset();self.episode=history.episode
        if history.index==self.index:
            if history.sha256!=self.history_hash:raise ValueError('changed committed stream event')
            return
        if history.index!=self.index+1 or (self.index>=0 and history.previous_chain!=self.history_hash):
            raise ValueError('skipped or divergent stream boundary')
        from .memory_flow import current_event,MemoryHistory
        event=current_event(history);o=event['observation']
        result=self.request(dict(op='advance',memory=self.memory,event=dict(event,
            executed_ticks=None if history.previous_tick is None else o['tick']-history.previous_tick)))
        if result['names']!=self.names or result['state_names']!=self.protocol['state_names']:
            raise ValueError('stream schema changed')
        # Validate proposals when used, retaining explicit invalid-proposal diagnostics.
        frame=dict(tick=o['tick'],state=result['state'],entities={e['id']:dict(
            xy=[sum(p[k] for p in e['points'])/len(e['points']) for k in (0,1)],count=len(e['points']),
            last_seen_tick=e['last_seen_tick']) for e in o['entities'] if e['points']})
        self.frames=(self.frames+[frame])[-self.protocol['history_length']:]
        self.memory=result['memory'];self.context=result;self.observation=o
        self.action_index=history.action_index if isinstance(history,MemoryHistory) else None
        self.index=history.index;self.history_hash=history.sha256;self.update_count+=1

    def _validate_action(self,action):
        if self.action_index is not None:return self.action_index.validate(action,self.observation)
        return validate_action(action,self.observation)

    def recommendation(self):
        legal=[];invalid=[]
        for a in self.context['proposals']:
            try:self._validate_action(a)
            except (ValueError,TypeError,KeyError) as exc:invalid.append(dict(action=a,error=str(exc)));continue
            if a not in legal:legal.append(a)
        raw=self.context['proposals'][0] if self.context['proposals'] else None
        return dict(names=self.names,proposals=legal,invalid_proposals=invalid,raw_first=raw,
                    first_legal=raw is not None and raw in legal)

    def evaluate(self,history,extras,mapping):
        if self.prepared_for is not history:self.prepare(history)
        supplied=[]
        for a in extras:
            self._validate_action(a)
            if a not in supplied:supplied.append(deepcopy(a))
        if len(supplied)>128:raise ValueError('host candidate limit')
        supplied.sort(key=digest)
        result=self.request(dict(op='score',values=self.context['values'],proposals=self.context['proposals'],
             ids=[e['id'] for e in self.observation['entities']],candidates=supplied,mapping=mapping,max_names=self.max_names))
        if result['names']!=self.names:raise ValueError('stream schema changed')
        packet=self.recommendation();actions=[];values=[]
        for a,v in zip(result['actions'],result['values']):
            try:self._validate_action(a)
            except (ValueError,TypeError,KeyError):continue
            if a not in actions:actions.append(a);values.append(v)
        return dict(packet,actions=actions,values=values)

    def vector_window(self):
        capacity=self.protocol['state_capacity'];length=self.protocol['history_length']
        rows=[[None]*capacity+[0.,None] for _ in range(length-len(self.frames))]
        tick=self.observation['tick']
        rows += [f['state']+[0.]*(capacity-len(f['state']))+[1.,(tick-f['tick'])*.05] for f in self.frames]
        return [v for row in rows for v in row]

    def geometry(self,action):
        """Same 18-coordinate style residual, built from compact entity frames."""
        o=self.observation;label=action['target_id']
        samples=[(f['tick'],f['entities'][label]) for f in self.frames if label in f['entities']]
        p=samples[-1][1]['xy'] if samples else None
        origin=[sum(r['own'][k] for r in o['robots'])/2 for k in ('x','y')]
        vectors=[]
        for start in (0,47):
            v=[0.,0.]
            for a in (action['a0'],action['a1']):
                if start<=a<start+32:
                    d,s=divmod(a-start,4);angle=d*math.pi/4
                    v[0]+=.1*(s+1)*math.cos(angle);v[1]+=.1*(s+1)*math.sin(angle)
            vectors+=v
        delta=[None,None] if p is None else [p[k]-origin[k] for k in (0,1)]
        goal=[None,None] if p is None else [o['goal']['xy'][k]-p[k] for k in (0,1)]
        movement=[None,None] if len(samples)<2 else [p[k]-samples[0][1]['xy'][k] for k in (0,1)]
        return [action['ticks']*.05,float(label==o['goal']['target_id']),*vectors,
                0 if not samples else samples[-1][1]['count'],*delta,*goal,
                None if not samples else (o['tick']-samples[-1][1]['last_seen_tick'])*.05,*movement,
                *[float(r['own'][k]) for r in o['robots'] for k in ('vx','vy')]]

    def snapshot(self):
        if self.index<0:raise ValueError('no stream state')
        value=dict(source=digest(self.source),protocol=self.protocol,episode=self.episode,index=self.index,
                   memory=self.memory,frames=self.frames,context=self.context,observation=self.observation,
                   history_hash=self.history_hash,update_count=self.update_count)
        return json.loads(canonical(dict(value=value,checksum=digest(value))))

    def restore(self,packet,history):
        if packet.get('checksum')!=digest(packet['value']):raise ValueError('stream snapshot checksum')
        value=deepcopy(packet['value'])
        if value['source']!=digest(self.source) or value['protocol']!=self.protocol:
            raise ValueError('stream semantic version mismatch; re-extract from raw trajectory')
        if not isinstance(history,StreamHistory) or (value['episode'],value['index'],value['history_hash'])!=(history.episode,history.index,history.sha256):
            raise ValueError('stream snapshot cursor mismatch')
        if canonical(value['observation'])!=canonical(history.events[0]['observation']):raise ValueError('snapshot observation mismatch')
        self.close();self.reset()
        for key in ('episode','index','memory','frames','context','observation','history_hash','update_count'):
            setattr(self,key,value[key])
