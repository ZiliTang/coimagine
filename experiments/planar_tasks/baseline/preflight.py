"""Explicit single-call native-tool protocol check; never synthesizes a robot policy."""
import json
from pathlib import Path
import time
from . import seal
from .config import validate_config
from .driver import bounded_completion, schema
from .transports import deepseek_complete
from ..runner import write_json


def preflight(config, output, *, release_path=seal.LOCK, complete=deepseek_complete):
    seal.verify_release(release_path);validate_config(config)
    if config['model']['provider']!='deepseek':raise ValueError('preflight transport is DeepSeek only')
    model=config['model'];settings=dict(model['settings'])
    if set(settings)&{'model','messages','tools','tool_choice','max_tokens','stream','api_key','authorization'}:
        raise ValueError('reserved transport setting')
    # Thinking accepts native tools with auto choice, but not forced functions.
    choice='auto' if settings.get('thinking',{}).get('type')=='enabled' else {'type':'function','function':{'name':'protocol_echo'}}
    body=dict(settings,model=model['name'],messages=[{'role':'user','content':
        'Protocol connectivity check only. Call protocol_echo with value coimagine.'}],
        tools=[schema('protocol_echo','Return the supplied value.',{'value':{'type':'string'}},['value'])],
        tool_choice=choice,max_tokens=1024,stream=False)
    size=len(json.dumps(body,ensure_ascii=False).encode())
    if size>65536:raise ValueError('preflight request exceeds 65536 bytes')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'manifest.json',dict(kind='native_tool_protocol_preflight',created_unix=time.time(),
        model=model,release_sha256=seal.digest(release_path),budget=dict(model_calls=1,
            input_bytes=65536,reserved_output_tokens=1024,wall_seconds=60),physics_episodes=0))
    write_json(output/'request.json',body)
    started=time.monotonic();response=bounded_completion(complete,body,60)
    write_json(output/'response.json',response)
    error=None
    try:
        if not response.get('ok'):raise ValueError(response.get('error','transport failed'))
        data=response['response']
        if data['model']!=model['name']:raise ValueError('returned model identity mismatch')
        choice=data['choices'][0];calls=choice['message'].get('tool_calls',[])
        if choice['finish_reason']!='tool_calls' or len(calls)!=1:raise ValueError('expected one native tool call')
        call=calls[0]['function']
        if call['name']!='protocol_echo' or json.loads(call['arguments'])!={'value':'coimagine'}:
            raise ValueError('native tool arguments differ')
    except (ValueError,KeyError,TypeError,IndexError) as exc:error=type(exc).__name__+': '+str(exc)
    result=dict(status='passed' if error is None else 'failed',error=error,model_calls=1,
        input_bytes=size,reserved_output_tokens=1024,wall_seconds=time.monotonic()-started,
        reported_usage=response.get('response',{}).get('usage'),physics_episodes=0,
        meaning='native tool transport only; not autonomous synthesis or policy performance')
    write_json(output/'summary.json',result)
    return result
