"""Frozen experiment tools; only explicit DeepSeek subcommands start model requests."""
import argparse
import json
from pathlib import Path
import sys

from . import seal
from .session import Session


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='op',required=True)
    release=sub.add_parser('seal');release.add_argument('--out',default=str(seal.LOCK))
    verify=sub.add_parser('verify');verify.add_argument('--release',default=str(seal.LOCK))
    init=sub.add_parser('init');init.add_argument('--session',required=True);init.add_argument('--config',required=True)
    serve=sub.add_parser('serve');serve.add_argument('--session',required=True)
    run=sub.add_parser('run-deepseek',help='explicit billable tool-calling synthesis; no retries')
    run.add_argument('--session',required=True)
    synth=sub.add_parser('synthesize-deepseek',help='create session and immediately run explicit billable synthesis')
    synth.add_argument('--config',required=True);synth.add_argument('--session',required=True)
    pre=sub.add_parser('preflight-deepseek',help='explicit one-call billable native-tool transport check; no physics')
    pre.add_argument('--config',required=True);pre.add_argument('--out',required=True)
    panel=sub.add_parser('create-panel',help='freeze new held-out roots; geometry checks only, no simulation')
    panel.add_argument('--out',required=True);panel.add_argument('--seed',type=int,required=True)
    panel.add_argument('--per-family',type=int,default=10)
    evaluate=sub.add_parser('evaluate',help='evaluate explicit frozen submission; no LLM calls or selection')
    evaluate.add_argument('--session',required=True);evaluate.add_argument('--panel',required=True)
    evaluate.add_argument('--out',required=True);evaluate.add_argument('--budget',required=True)
    args=parser.parse_args()
    if args.op=='seal':result=seal.build_release(args.out);print(json.dumps({'version':result['version'],'files':len(result['files'])}));return
    if args.op=='verify':result=seal.verify_release(args.release);print(json.dumps({'verified':True,'version':result['version'],'files':len(result['files'])}));return
    if args.op=='create-panel':
        seal.verify_release()
        from .panels import create_panel
        result=create_panel(args.out,seed=args.seed,per_family=args.per_family)
        print(json.dumps({'panel':args.out,'content_sha256':result['content_sha256']}));return
    if args.op=='evaluate':
        from .evaluation import evaluate,EvaluationBudget
        result=evaluate(args.session,args.panel,args.out,EvaluationBudget(**json.loads(Path(args.budget).read_text())))
        print(json.dumps({k:v for k,v in result.items() if k!='results'}));return
    if args.op=='preflight-deepseek':
        from .preflight import preflight
        print(json.dumps(preflight(json.loads(Path(args.config).read_text()),args.out)));return
    if args.op=='init':
        session=Session.create(args.session,json.loads(Path(args.config).read_text()))
        session.close();print(json.dumps({'session':args.session,'status':'initialized; no model or simulator started'}));return
    if args.op=='synthesize-deepseek':
        config=json.loads(Path(args.config).read_text())
        if config['purpose']!='development' or config['model']['provider']!='deepseek':
            raise ValueError('explicit DeepSeek development config required')
        session=Session.create(args.session,config)
    else:session=Session(args.session)
    try:
        if args.op in ('run-deepseek','synthesize-deepseek'):
            if session.config['purpose']!='development' or session.config['model']['provider']!='deepseek':
                raise ValueError('run-deepseek requires explicit development config with provider=deepseek')
            from .driver import drive
            from .transports import deepseek_complete
            print(json.dumps(drive(session,deepseek_complete)));return
        for line in sys.stdin:
            try:
                request=json.loads(line)
                result=session.tool(request['name'],request['arguments'])
            except Exception as exc:result={'ok':False,'error':{'type':type(exc).__name__,'message':str(exc)}}
            print(json.dumps(result),flush=True)
    finally:session.close()


if __name__=='__main__':main()
