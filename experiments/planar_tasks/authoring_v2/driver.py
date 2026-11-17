"""Current backend-neutral loop; public skill knowledge is included before budget accounting."""
import json
import time
from experiments.planar_tasks.baseline.config import request_settings
from experiments.planar_tasks.baseline.driver import bounded_completion, TOOLS as FREE_TOOLS
from experiments.planar_tasks.option_search_v3.driver import TOOLS as SEARCH_TOOLS
from experiments.planar_tasks.authoring import knowledge
from . import context
from .session import verify_profile
from experiments.planar_tasks.baseline.driver import schema, STRING, INTEGER


EXTRA_TOOLS=[
 schema('test_summary','Compact deterministic overview of one policy test.',{'test_id':STRING},['test_id']),
 schema('test_events','Page boundary-sampled stage/contact/failure changes. Branch is required for batch records.',{'record_id':STRING,'branch':INTEGER,'start':INTEGER,'count':INTEGER},['record_id']),
 schema('read_trace','Read legal observation fields for selected rows/robots; default excludes scans.',{'record_id':STRING,'branch':INTEGER,'start':INTEGER,'count':INTEGER,'fields':{'type':'array','items':{'type':'string','enum':['own','execution','contacts','scan','patches','goal','legal_pairs']}},'robots':{'type':'array','items':{'type':'integer','enum':[0,1]}}},['record_id']),
 schema('run_branches','Reset and replay the same prefix for each branch. All actual ticks including prefixes count. Returns compact overview, not raw sensor streams.',{'task':STRING,'prefix':{'type':'array','items':{'type':'object'}},'candidates':{'type':'array','items':{'type':'array','items':{'type':'object'}}}},['task','prefix','candidates']),
 schema('batch_rows','Page/explicitly sort batch outcomes; estimates are not truth.',{'batch_id':STRING,'start':INTEGER,'count':INTEGER,'sort_by':{'type':'string','enum':['branch','executed_ticks','estimated_translation','estimated_rotation_abs']},'descending':{'type':'boolean'}},['batch_id']),
 schema('notebook','Replace bounded persistent author notes; include hypotheses, evidence IDs, failures and next checks. Notes are not host-verified.',{'text':STRING},['text']),
 schema('list_results','Page durable public tool-result IDs, including failures.',{'start':INTEGER,'count':INTEGER},[]),
 schema('read_result','Read at most 8000 bytes of a full archived public tool request/result; no arbitrary file paths.',{'record_id':INTEGER,'start':INTEGER,'count':INTEGER},['record_id']),
]


def drive(session,complete):
    """Run one autonomous session; transport is caller-supplied, never implicit.

    A model lacking tool-calling capability cannot be silently replaced by the
    legacy fixed-feedback loop. Model failure/budget exhaustion retains no-submit.
    """
    verify_profile(session.path)
    profile = knowledge.verify_session(session.path)
    is_search = profile['method'] == 'option_search_v3'
    tools = list(SEARCH_TOOLS if is_search else FREE_TOOLS) + EXTRA_TOOLS
    entry = 'heuristic.py containing rank_actions and evaluate_state' if is_search else 'policy.py and explicit helper files'
    if session.state.get('driver_started'):raise ValueError('driver already attempted; no implicit resume')
    session._active();session.state['driver_started']=True;session._save()
    model=session.config['model'];settings=dict(model['settings'])
    if set(settings)&{'model','messages','tools','tool_choice','max_tokens','stream','api_key','authorization'}:
        raise ValueError('model settings override reserved request fields')
    messages=[{'role':'system','content':(session.workspace/'instructions.txt').read_text() + '\n\n' + (session.workspace/'option_guide.md').read_text() + '\n\n' + (session.workspace/'experiment_api.md').read_text()},
              {'role':'user','content':'Use the workspace tools to inspect campaign.json and contract.json, then autonomously develop and submit your policy.'}]
    reason=None
    while session.state['status']=='developing':
        try:
            session._active()
            session.check_finalization()
            reserve=session.config.get('finalization',{})
            left=session.budget_status()['remaining']
            if reserve and any(left[k]<=0 for k in ('control_ticks',)):
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
                        f'The only available tool is submit. You may include complete {entry} text '
                        'in contents together with the explicit files list; existing listed files are copied as-is. '
                        'No further experiments or separate write_file calls are available. '
                        'If validation fails, correct and resubmit only within the remaining submission budget. '
                        'No automatic choice or fallback policy will be supplied.'})
                    session.state['submission_notice_sent']=True;session._save()
                messages[1]['content']=(f'Autonomously develop then submit {entry}. '
                    'Read campaign.json, contract.json, option_guide.md and experiment_api.md. Submission reserves are part of total limits. '
                    'Stage 2 only permits atomic submit(files, contents), not separate writes or tests. '
                    'Authoritative remaining budgets (before this request):\n'+json.dumps(session.budget_status()))
                exposed=[t for t in tools if not finalizing or t['function']['name']=='submit']
                choice={'type':'function','function':{'name':'submit'}} if finalizing else 'auto'
                view,projection=context.project(messages,session.settings,session.state.get('author_notebook',''))
                payload=dict(request_settings(session.config,finalizing=finalizing),model=model['name'],messages=view,tools=exposed,tool_choice=choice,max_tokens=cap,stream=False)
                size=len(json.dumps(payload,ensure_ascii=False).encode())
                if reserve and not finalizing and size>left['model_input_bytes']-reserve['model_input_bytes']:
                    session.finalize('next request reaches reserved input bytes');continue
                break
            # Reserve before making a request. Failed/incomplete calls count too.
            session._charge('model_input_bytes',size);session._charge('model_output_tokens',cap);session._charge('model_calls')
            available=session.available_wall() if session.state.get('finalizing') else session.experiment_wall()
            timeout=min(session.budget.model_call_seconds,available)
            started=time.monotonic();response=bounded_completion(complete,payload,timeout)
            session.event('context_projection',projection,{'input_accounted_bytes':size})
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
