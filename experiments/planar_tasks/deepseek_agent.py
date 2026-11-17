"""Explicit, bounded DeepSeek program-synthesis calls. Never called by deployment.

Credential: DEEPSEEK_API_KEY or ~/.config/coimagine/deepseek_api_key (0600).
Fixed official endpoint, no redirects/retries, no credential in manifests/transcripts.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import stat
import time
import urllib.error
import urllib.request

from .agentic import freeze, trial, locked
from .runner import write_json

ENDPOINT = 'https://api.deepseek.com/chat/completions'
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


def _credential():
    value = os.environ.get('DEEPSEEK_API_KEY')
    if value:
        return value
    path = Path.home()/'.config/coimagine/deepseek_api_key'
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError('credential file must not be accessible to group/other')
    return path.read_text().strip()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('API redirects disabled')


def _request_worker(body, connection, timeout=120.):
    try:
        request = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
            headers={'Authorization':'Bearer '+_credential(), 'Content-Type':'application/json'})
        with urllib.request.build_opener(NoRedirect).open(request,timeout=timeout) as response:
            raw=response.read(MAX_RESPONSE_BYTES+1)
        if len(raw)>MAX_RESPONSE_BYTES:
            raise ValueError('API response size limit')
        connection.send(dict(ok=True,response=json.loads(raw)))
    except urllib.error.HTTPError as exc:
        # Do not record response headers/body, which might echo credentials.
        connection.send(dict(ok=False,error='http_error',http_status=exc.code))
    except Exception as exc:
        connection.send(dict(ok=False,error=type(exc).__name__))
    finally:
        connection.close()


def request(body, timeout=180.):
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
        raise ValueError('positive finite request timeout required')
    context=mp.get_context('spawn')
    parent,child=context.Pipe(duplex=False)
    worker=context.Process(target=_request_worker,args=(body,child,timeout))
    worker.start();child.close()
    try:
        if not parent.poll(timeout):
            return dict(ok=False,error='wall_timeout',billing='unknown; no retry')
        try:
            return parent.recv()
        except EOFError:
            return dict(ok=False,error='request_worker_exited')
    finally:
        if worker.is_alive():worker.terminate()
        worker.join(timeout=3.)
        if worker.is_alive():worker.kill();worker.join()
        parent.close()


def synthesize(session):
    session=Path(session)
    # Separate lifetime lock prevents accidentally running two billable loops.
    with (session/'.api_lock').open('a') as lockfile:
        import fcntl
        fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (session/'api_budget.json').exists():
            raise ValueError('API session already attempted; automatic resumption/retry disabled')
        config=json.loads((session/'session.json').read_text())
        budget=dict(max_calls=config['maximum_candidates'],max_completion_tokens_per_call=8192,
            max_request_characters=200000,wall_seconds_per_call=180,thinking='disabled',
            retries=0,model=config['model'],endpoint=ENDPOINT,started_unix=time.time())
        write_json(session/'api_budget.json',budget)
        packet=json.loads((session/'packet.json').read_text())
        messages=[dict(role='system',content=packet.pop('instructions')+
            '\nReturn a JSON object with exactly code (complete Python program as string) and note (brief design/change explanation).'),
            dict(role='user',content=json.dumps(packet)+'\nSubmit candidate 1 of at most '+str(budget['max_calls'])+'.')]
        statuses=[]
        for index in range(budget['max_calls']):
            directory=session/f'api_call_{index:02d}'
            directory.mkdir()
            body=dict(model=config['model'],messages=messages,max_tokens=8192,
                thinking={'type':'disabled'},response_format={'type':'json_object'},stream=False)
            if len(json.dumps(body))>budget['max_request_characters']:
                statuses.append(dict(call=index,error='request_character_budget'))
                break
            write_json(directory/'request.json',body)
            started=time.monotonic()
            try:
                response=request(body,budget['wall_seconds_per_call'])
            except Exception as exc:
                response=dict(ok=False,error='request_exception:'+type(exc).__name__)
            write_json(directory/'response.json',response)
            status=dict(call=index,wall_seconds=time.monotonic()-started,ok=response['ok'])
            statuses.append(status)
            if not response['ok']:
                status.update(error=response.get('error'),http_status=response.get('http_status'))
                write_json(session/'api_status.json',statuses)
                break
            try:
                data=response['response']
                if not isinstance(data,dict):
                    raise ValueError('invalid response envelope')
                status.update(returned_model=data.get('model'),usage=data.get('usage'),
                              response_id=data.get('id'))
                choice=data['choices'][0]
                if choice['finish_reason']!='stop':
                    raise ValueError('completion did not finish normally')
                content=choice['message']['content']
                proposal=json.loads(content)
                if set(proposal)!={'code','note'} or not all(isinstance(v,str) for v in proposal.values()):
                    raise ValueError('invalid proposal schema')
                if len(proposal['code'])>65536:raise ValueError('program exceeds size limit')
            except (KeyError,IndexError,TypeError,ValueError) as exc:
                status['error']=str(exc)
                messages.append(dict(role='user',content='Previous response was invalid/incomplete; return valid JSON with complete code and note. This consumed one model call.'))
                write_json(session/'api_status.json',statuses)
                continue
            try:
                feedback=trial(session,proposal['code'],proposal['note'])
            except Exception as exc:
                status['error']='trial_exception:'+type(exc).__name__
                write_json(session/'api_status.json',statuses)
                break
            messages.append(dict(role='assistant',content=content))
            messages.append(dict(role='user',content='Development trial feedback:\n'+json.dumps(feedback)+
                '\nRevise the single policy using this evidence. Remaining candidate calls: '+str(budget['max_calls']-1-index)))
            status['candidate']=feedback['candidate']
            status['successes']=sum(e['summary']['success'] for e in feedback['episodes'])
            write_json(session/'api_status.json',statuses)
            print(json.dumps(status),flush=True)
        write_json(session/'api_status.json',statuses)
        freeze_error=None
        try:
            frozen=freeze(session)
        except ValueError as exc:
            frozen=None
            freeze_error=str(exc)
        result=dict(api_calls=len(list(session.glob('api_call_*/request.json'))),statuses=statuses,
                    frozen=frozen,freeze_error=freeze_error,status='frozen' if frozen else 'incomplete')
        write_json(session/'synthesis.json',result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True)
    args=parser.parse_args()
    print(json.dumps(synthesize(args.session),indent=2))


if __name__=='__main__':main()
