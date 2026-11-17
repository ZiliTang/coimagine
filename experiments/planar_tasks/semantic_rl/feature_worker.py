"""Standard-library sandbox broker. Author code runs in a fresh fork per query.

Only projected public packets cross this process boundary. No torch, simulator,
repository, network or credentials. Host never unpickles worker-controlled output.
"""
import copy
import gc
import json
import math
import marshal
import os
import pickle
import struct
import sys
import tempfile
from array import array
import importlib.util

MAX_REPLY = 1024 * 1024


def prepare(source, history, compiled_path=None):
    namespace = {'__name__': '__main__'}
    if compiled_path:
        spec=importlib.util.spec_from_file_location('_coimagine_semantics',compiled_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        namespace=module.__dict__
    else:exec(compile(source, '/work/author.py', 'exec'), namespace)
    # Match the legacy adapter, including author-visible globals.
    if not compiled_path:
        namespace['_PREFIX_MAP'] = {}
        namespace['_COMPLETE_FEATURES'] = namespace['semantic_features']
        exec('''import json
def semantic_features(m, o, a, proposals):
    if 'prefix_semantic_features' in globals():
        prefix = _PREFIX_MAP.get(json.dumps(a,sort_keys=True,separators=(',',':')), a)
        return prefix_semantic_features(m, o, prefix, proposals)
    return _COMPLETE_FEATURES(m, o, a, proposals)
''', namespace)
    memory = namespace['initial_memory'](); previous = None
    for event, encoded_observation in history:
        o = event['observation']
        memory = namespace['update_memory'](memory, dict(
            observation=marshal.loads(encoded_observation),
            action=None if previous is None else copy.deepcopy(event['action']),
            executed_ticks=None if previous is None else o['tick']-previous))
        if not isinstance(memory, dict) or len(json.dumps(memory, allow_nan=False).encode()) > 8192:
            raise ValueError('memory limit')
        previous = o['tick']
    if not compiled_path:return namespace, memory, encoded_observation
    observation=marshal.loads(encoded_observation)
    context=namespace['prepare_context'](copy.deepcopy(memory),observation)
    if not isinstance(context,dict) or set(context)!={'values','proposals'}:
        raise ValueError('prepare_context must return values/proposals')
    values=context['values']
    if not isinstance(values,(list,tuple)) or not 1<=len(values)<=4096 or any(
        type(v) not in (float,int,bool) or math.isinf(v) for v in values):
        raise ValueError('context requires1..4096 numeric values; NaN means unknown')
    # bytes-backed views cannot be made writable by the author kernel.
    state=array('d',values).tobytes()
    ids=[e['id'] for e in history[-1][0]['observation']['entities']]
    return namespace, state, context['proposals'], ids


def evaluate(prepared, candidates, mapping, max_names):
    if len(prepared)==4:return evaluate_typed(prepared,candidates,mapping,max_names)
    namespace, memory, encoded_observation = prepared
    namespace['_PREFIX_MAP'] = mapping
    names = namespace['NAMES']
    if (not isinstance(names, (tuple, list)) or not 1 <= len(names) <= max_names
            or len(set(names)) != len(names)
            or any(not isinstance(n, str) or not 0 < len(n) <= 100 for n in names)):
        raise ValueError('feature names')
    proposals = namespace['propose_actions'](copy.deepcopy(memory), marshal.loads(encoded_observation))
    if not isinstance(proposals, list) or len(proposals) > 16:
        raise ValueError('at most16 proposals')
    actions = []
    for action in proposals + candidates:
        if action not in actions:
            actions.append(action)
    values = []
    for action in actions:
        v = namespace['semantic_features'](copy.deepcopy(memory), marshal.loads(encoded_observation),
                                          copy.deepcopy(action), copy.deepcopy(proposals))
        if not isinstance(v, dict) or set(v) != set(names):
            raise ValueError('feature keys')
        row = [v[n] for n in names]
        if any(x is not None and (type(x) not in (float, int, bool) or not math.isfinite(x)) for x in row):
            raise ValueError('features numeric')
        values.append(row)
    return dict(names=list(names), proposals=proposals, actions=actions, values=values)


def evaluate_typed(prepared,candidates,mapping,max_names):
    namespace,state,proposals,ids=prepared
    names=namespace['NAMES']
    if (not isinstance(names,(list,tuple)) or not 1<=len(names)<=max_names or len(set(names))!=len(names)
            or any(not isinstance(n,str) or not 0<len(n)<=100 for n in names)):
        raise ValueError('feature names')
    if not isinstance(proposals,list) or len(proposals)>16:raise ValueError('at most16 proposals')
    actions=[]
    for a in proposals+candidates:
        if a not in actions:actions.append(a)
    if not actions:return dict(names=list(names),proposals=proposals,actions=[],values=[])
    encoded=[]
    for action in actions:
        prefix=mapping.get(json.dumps(action,sort_keys=True,separators=(',',':')),action)
        target=prefix.get('target_id')
        # -1 unselected, -2 target-free, -3 invalid recommendation (host rejects).
        entity=(-1 if 'target_id' not in prefix else -2 if target is None
                else ids.index(target) if target in ids else -3)
        encoded.extend([entity]+[prefix.get(k,-1) for k in ('a0','a1','ticks')])
    matrix=memoryview(array('d',encoded).tobytes()).cast('d',shape=(len(actions),4))
    output=array('d',[float('nan')])*(len(actions)*len(names))
    view=memoryview(output).cast('B').cast('d',shape=(len(actions),len(names)))
    namespace['evaluate_candidates'](memoryview(state).cast('d'),matrix,view)
    values=[]
    for i in range(len(actions)):
        row=output[i*len(names):(i+1)*len(names)]
        if any(math.isinf(v) for v in row):raise ValueError('features finite or NaN required')
        values.append([None if math.isnan(v) else v for v in row])
    return dict(names=list(names),proposals=proposals,actions=actions,values=values)


def frame_write(fd, data):
    packet = memoryview(struct.pack('!I', len(data))+data)
    while packet:
        packet = packet[os.write(fd, packet):]


def frame_read(fd):
    def exact(n):
        value = bytearray()
        while len(value)<n:
            chunk=os.read(fd,n-len(value))
            if not chunk:raise EOFError()
            value.extend(chunk)
        return bytes(value)
    size=struct.unpack('!I',exact(4))[0]
    if size>16*1024*1024:raise ValueError('request limit')
    return exact(size)


def context_loop(source, history, incoming, outgoing, compiled_path=None):
    # Initialization runs once per immutable window, inside its own process.
    # Its globals/imports are inherited by a new fork for EVERY prefix query.
    null=os.open('/dev/null',os.O_WRONLY);os.dup2(null,1);os.dup2(null,2)
    # This window and every scoring query have bounded process lifetimes.
    # Refcounts still release the many acyclic observation copies immediately;
    # repeatedly scanning them for cycles costs more than the feature arithmetic.
    # Do not disable GC in the long-lived broker or training process.
    gc.disable()
    try:prepared=prepare(source,history,compiled_path)
    except BaseException as exc:
        frame_read(incoming)
        frame_write(outgoing,json.dumps(dict(error=type(exc).__name__+': '+str(exc))).encode())
        return
    gc.collect()
    while True:
        try:request=pickle.loads(frame_read(incoming))
        except EOFError:return
        with tempfile.TemporaryFile() as reply:
            pid=os.fork()
            if pid==0:
                try:
                    result=evaluate(prepared,request['candidates'],request['mapping'],request['max_names'])
                    data=json.dumps(dict(result=result),allow_nan=False).encode()
                except BaseException as exc:
                    data=json.dumps(dict(error=type(exc).__name__+': '+str(exc))).encode()
                if len(data)>MAX_REPLY:data=b'{"error":"reply limit"}'
                reply.write(data);reply.flush();os._exit(0)
            _,status=os.waitpid(pid,0);reply.seek(0);data=reply.read(MAX_REPLY+1)
            if status or not data or len(data)>MAX_REPLY:data=b'{"error":"isolated feature child failed"}'
            frame_write(outgoing,data)


def serve(source,compiled_path=None):
    cache={};context=None;previous=None
    try:
        while True:
            try:request=pickle.loads(frame_read(0))
            except EOFError:return
            for key,encoded in request['new']:
                event=json.loads(encoded)
                cache[key]=(event,marshal.dumps(event['observation']))
            history=[cache[key] for key in request['keys']]
            cache={key:cache[key] for key in request['keys']}
            if previous!=request['keys']:
                if context:
                    pid,send,receive=context;os.close(send);os.close(receive);os.waitpid(pid,0)
                read_request,write_request=os.pipe();read_reply,write_reply=os.pipe()
                pid=os.fork()
                if pid==0:
                    os.close(write_request);os.close(read_reply)
                    try:context_loop(source,history,read_request,write_reply,compiled_path)
                    finally:os._exit(0)
                os.close(read_request);os.close(write_reply)
                context=(pid,write_request,read_reply);previous=request['keys']
            frame_write(context[1],pickle.dumps(request,protocol=4))
            try:data=frame_read(context[2])
            except EOFError:data=b'{"error":"feature context preparation failed"}'
            frame_write(1,data)
    finally:
        if context:
            pid,send,receive=context;os.close(send);os.close(receive);os.waitpid(pid,0)
