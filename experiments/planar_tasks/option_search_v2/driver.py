"""Provider-neutral autonomous tool loop. No fixed feedback/revision schedule.

The injected trusted completion transport accepts a Chat-Completions-shaped body
and returns its JSON response. No credentials/provider SDK are included here.
API transport runs outside the agent workspace and is killed on timeout. This
module itself never selects a model or opens a network connection.
"""
import json
import multiprocessing as mp
import os
import signal
import time

from .build import HERE as BASE
from experiments.planar_tasks.baseline.config import request_settings


def schema(name,description,properties,required):
    return {'type':'function','function':{'name':name,'description':description,
        'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}}


STRING={'type':'string'}
INTEGER={'type':'integer'}
FILES={'type':'array','items':STRING,'minItems':1}
TOOLS=[
    schema('list_files','List this development workspace.',{},[]),
    schema('read_file','Read a bounded byte range of a workspace text file.',
           {'path':STRING,'start':{'type':'integer','minimum':0},'limit':{'type':'integer','minimum':1,'maximum':64000}},['path']),
    schema('write_file','Edit heuristic.py, experiments/*.py or notes/*.txt. Framework and budgets are fixed.',{'path':STRING,'text':STRING},['path','text']),
    schema('run_python','Run your Python script with the development simulator client. Choose experiments yourself.',
           {'path':STRING,'seconds':{'type':'number'}},['path','seconds']),
    schema('test_policy','Run fixed search with files=[heuristic.py] on one development task. No automatic repair or selection.',
           {'files':FILES,'task':STRING,'max_ticks':{'type':'integer','minimum':1,'maximum':1800}},['files','task','max_ticks']),
    schema('read_log','Read your policy test sensors/decisions/stderr, or Python run rpc/stdout/stderr. Pass returned test_id or run_id.',
           {'test_id':STRING,'stream':{'enum':['sensors','decisions','stderr','rpc','stdout']},'start':{'type':'integer','minimum':0},'count':{'type':'integer','minimum':1,'maximum':50}},['test_id','stream']),
    schema('submit','Atomically submit your selected files. Optional contents supplies complete new/replacement text for listed files, only heuristic.py. Checks loading/interface only, no physics; accepted submission ends synthesis.',
           {'files':FILES,'contents':{'type':'object','additionalProperties':STRING}},['files']),
]


def _worker(complete,payload,timeout,pipe):
    os.setsid()
    try:pipe.send({'ok':True,'response':complete(payload,timeout)})
    except Exception as exc:pipe.send({'ok':False,'error':type(exc).__name__})
    finally:pipe.close()


def bounded_completion(complete,payload,timeout):
    context=mp.get_context('fork')
    receive,send=context.Pipe(duplex=False)
    process=context.Process(target=_worker,args=(complete,payload,timeout,send));process.start();send.close()
    try:
        if not receive.poll(timeout):return {'ok':False,'error':'model_wall_timeout','billing':'unknown; no retry'}
        try:return receive.recv()
        except EOFError:return {'ok':False,'error':'model_worker_exited','billing':'unknown; no retry'}
    finally:
        # Kill the provider process group even if its entry process has exited.
        # Nested HTTP workers must not survive a model-call deadline.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        if process.is_alive():process.terminate()
        process.join(timeout=1.)
        if process.is_alive():process.kill();process.join()
        receive.close()


def drive(session,complete):
    """Run one autonomous session; transport is caller-supplied, never implicit.

    A model lacking tool-calling capability cannot be silently replaced by the
    legacy fixed-feedback loop. Model failure/budget exhaustion retains no-submit.
    """
    if session.state.get('driver_started'):raise ValueError('driver already attempted; no implicit resume')
    session._active();session.state['driver_started']=True;session._save()
    model=session.config['model'];settings=dict(model['settings'])
    if set(settings)&{'model','messages','tools','tool_choice','max_tokens','stream','api_key','authorization'}:
        raise ValueError('model settings override reserved request fields')
    messages=[{'role':'system','content':(BASE/'prompt.txt').read_text()},
              {'role':'user','content':'Use the workspace tools to inspect campaign.json and contract.json, then autonomously develop and submit your policy.'}]
    reason=None
    while session.state['status']=='developing':
        try:
            session._active()
            session.check_finalization()
            reserve=session.config.get('finalization',{})
            left=session.budget_status()['remaining']
            if reserve and any(left[k]<=0 for k in ('control_ticks','resets','policy_tests','python_total_seconds')):
                session.finalize('experimental resource exhausted')
            if reserve and not session.state.get('finalizing') and left['model_output_tokens']-session.budget.model_call_output_tokens<reserve['model_output_tokens']:
                session.finalize('next request reaches reserved output tokens')
            remaining=session.available('model_output_tokens')
            cap=min(session.budget.model_call_output_tokens,remaining)
            if cap<=0:raise ValueError('model_output_tokens budget exhausted')
            while True:
                finalizing=session.state.get('finalizing',False)
                if finalizing and not session.state.get('submission_notice_sent'):
                    messages.append({'role':'user','content':
                        'DEVELOPMENT IS CLOSED. Use submit now to deliver your own chosen program. '
                        'The only available tool is submit. You may include complete heuristic.py text '
                        'in contents together with the explicit files list; existing listed files are copied as-is. '
                        'No further experiments or separate write_file calls are available. '
                        'If validation fails, correct and resubmit only within the remaining submission budget. '
                        'No automatic choice or fallback policy will be supplied.'})
                    session.state['submission_notice_sent']=True;session._save()
                messages[1]['content']=('Autonomously develop then submit heuristic.py containing evaluate_nodes and score_actions. '
                    'Read campaign.json and contract.json. Submission reserves are part of total limits. '
                    'Stage 2 only permits atomic submit(files, contents), not separate writes or tests. '
                    'Authoritative remaining budgets (before this request):\n'+json.dumps(session.budget_status()))
                exposed=[t for t in TOOLS if not finalizing or t['function']['name']=='submit']
                choice={'type':'function','function':{'name':'submit'}} if finalizing else 'auto'
                payload=dict(request_settings(session.config,finalizing=finalizing),model=model['name'],messages=messages,tools=exposed,tool_choice=choice,max_tokens=cap,stream=False)
                size=len(json.dumps(payload,ensure_ascii=False).encode())
                if reserve and not finalizing and size>left['model_input_bytes']-reserve['model_input_bytes']:
                    session.finalize('next request reaches reserved input bytes');continue
                break
            # Reserve before making a request. Failed/incomplete calls count too.
            session._charge('model_input_bytes',size);session._charge('model_output_tokens',cap);session._charge('model_calls')
            available=session.available_wall() if session.state.get('finalizing') else session.experiment_wall()
            timeout=min(session.budget.model_call_seconds,available)
            started=time.monotonic();response=bounded_completion(complete,payload,timeout)
            session.event('model',payload,dict(response,wall_seconds=time.monotonic()-started))
            if not response.get('ok'):
                reason=response.get('error','model_transport_error');break
            data=response['response']
            if not isinstance(data,dict) or data.get('model')!=model['name']:
                raise ValueError('returned model identity differs from frozen campaign configuration')
            choice=data['choices'][0]
            # A truncated response was still billed; record usage before checking
            # whether any tool output can be accepted.
            session.state.setdefault('reported_model_usage',[]).append(data.get('usage'));session._save()
            if choice.get('finish_reason')=='length' and reserve:
                session.event('model_output_truncated',{'phase': 'submission' if session.state.get('finalizing') else 'development'},
                              {'tools_executed':0,'action':'use only predeclared submission allowance'})
                if not session.state.get('finalizing'):
                    session.finalize('model_output_token_cap_reached')
                # Do not install a partial tool call or partial reasoning message
                # into the conversation. Retain the exact raw response in events.
                messages.append({'role':'user','content':
                    'Your previous response reached its output-token limit. None of its tools were executed. '
                    'Development is closed. Deliver your chosen complete program with submit using only '
                    'the remaining predeclared submission allowance; no further experiments are available.'})
                continue
            if choice.get('finish_reason') not in ('stop','tool_calls'):raise ValueError('model response incomplete')
            message=choice['message'];calls=message.get('tool_calls') or []
            assistant={'role':'assistant','content':message.get('content'),**({'tool_calls':calls} if calls else {})}
            if 'reasoning_content' in message:assistant['reasoning_content']=message['reasoning_content']
            messages.append(assistant)
            if session.state.get('finalizing') and (len(calls)!=1 or calls[0].get('function',{}).get('name')!='submit'):
                # Do not interpret a list of candidates as an ordered fallback.
                # An invalid delivery response consumes this model call; only a
                # remaining, predeclared submission call can correct it.
                result={'ok':False,'error':{'type':'ValueError',
                    'message':'submission phase requires exactly one submit tool call; none of these calls executed'}}
                session.event('submission_response_rejected',{'tool_call_count':len(calls)},result)
                for call in calls:
                    messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result)})
                if not calls:messages.append({'role':'user','content':result['error']['message']})
                continue
            if not calls:
                reason='model_ended_without_submission';break
            for call in calls:
                if session.state['status']!='developing':break
                try:
                    arguments=json.loads(call['function']['arguments'])
                    result=session.tool(call['function']['name'],arguments)
                except (TypeError,KeyError,ValueError) as exc:
                    session._charge('tool_calls')
                    result={'ok':False,'error':{'type':type(exc).__name__,'message':str(exc)}}
                    session.event('malformed_model_tool',call,result)
                messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result)})
        except Exception as exc:
            reason=type(exc).__name__+': '+str(exc);break
    if session.state['status']=='developing':session.state['status']='stopped_without_submission'
    session.state['stop_reason']=reason or session.state['status']
    session.state['finished_monotonic']=time.monotonic();session.state['finished_unix']=time.time();session._save()
    return {'status':session.state['status'],'stop_reason':session.state['stop_reason'],
            'usage':session.state['usage'],'reported_model_usage':session.state.get('reported_model_usage',[])}
