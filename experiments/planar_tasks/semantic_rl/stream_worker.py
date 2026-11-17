"""Trusted worker additions to feature_worker.py; loaded only inside bubblewrap.

A pristine native module is loaded once. Each update/query forks from that module;
only explicit JSON memory persists, not author globals. No historical observation
window, no per-boundary full GC, and no authored pickle returns to the host.
"""

def stream_request(namespace, request):
    if request['op']=='advance':
        memory=request['memory']
        if memory is None:memory=namespace['initial_memory']()
        memory=namespace['update_memory'](memory,copy.deepcopy(request['event']))
        if not isinstance(memory,dict) or len(json.dumps(memory,allow_nan=False).encode())>8192:
            raise ValueError('explicit memory limit')
        context=namespace['prepare_context'](copy.deepcopy(memory),request['event']['observation'])
        if not isinstance(context,dict) or set(context)!={'values','state','proposals'}:
            raise ValueError('stream context requires values/state/proposals')
        values=context['values'];state=context['state']
        if (not isinstance(values,(list,tuple)) or not 1<=len(values)<=4096 or
            any(type(v) not in (int,float,bool) or math.isinf(v) for v in values)):
            raise ValueError('invalid numerical context')
        if (not isinstance(state,(list,tuple)) or len(state)!=len(namespace['STATE_NAMES']) or
            any(v is not None and (type(v) not in (int,float,bool) or not math.isfinite(v)) for v in state)):
            raise ValueError('invalid state vector')
        if not isinstance(context['proposals'],list) or len(context['proposals'])>16:raise ValueError('proposal limit')
        return dict(memory=memory,values=[None if math.isnan(v) else v for v in values],state=state,
                    proposals=context['proposals'],names=namespace['NAMES'],state_names=namespace['STATE_NAMES'])
    if request['op']=='score':
        state=array('d',[float('nan') if v is None else v for v in request['values']]).tobytes()
        return evaluate_typed((namespace,state,request['proposals'],request['ids']),
                              request['candidates'],request['mapping'],request['max_names'])
    raise ValueError('unknown stream operation')


def stream_serve(compiled_path):
    spec=importlib.util.spec_from_file_location('_coimagine_semantics',compiled_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace=module.__dict__
    gc.disable()
    while True:
        try:request=pickle.loads(frame_read(0))
        except EOFError:return
        with tempfile.TemporaryFile() as reply:
            pid=os.fork()
            if pid==0:
                null=os.open('/dev/null',os.O_WRONLY);os.dup2(null,1);os.dup2(null,2)
                try:data=json.dumps(dict(result=stream_request(namespace,request)),allow_nan=False).encode()
                except BaseException as exc:data=json.dumps(dict(error=type(exc).__name__+': '+str(exc))).encode()
                if len(data)>MAX_REPLY:data=b'{"error":"reply limit"}'
                reply.write(data);reply.flush();os._exit(0)
            _,status=os.waitpid(pid,0);reply.seek(0);data=reply.read(MAX_REPLY+1)
            if status or not data or len(data)>MAX_REPLY:data=b'{"error":"isolated stream child failed"}'
            frame_write(1,data)
