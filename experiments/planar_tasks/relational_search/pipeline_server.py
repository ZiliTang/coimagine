"""Read-only human pipeline console. Deliberately separate from public author RPC."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from .pipeline import read, unpack, identifier

STATIC=Path(__file__).parent/'pipeline_static'
DEFAULT_ROOT=Path(__file__).resolve().parents[3]/'outputs/relational_pipeline'


def handler_for(root,rl_root=None):
    root=Path(root).resolve()
    rl_root=Path(rl_root).resolve() if rl_root is not None else root.parent/'semantic_rl'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def send(self,value,kind='application/json; charset=utf-8',status=200):
            data=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'self'")
            self.end_headers();self.wfile.write(data)
        def do_POST(self):self.send(dict(error='Read-only human console; use authoring_v2.relational host tools.'),status=405)
        do_DELETE=do_PUT=do_PATCH=do_POST
        def do_GET(self):
            try:
                u=urlparse(self.path);q=parse_qs(u.query);parts=u.path.strip('/').split('/')
                files={'':'index.html','pipeline.js':'pipeline.js','pipeline.css':'pipeline.css',
                       'rl':'rl.html','rl.js':'rl.js','rl.css':'rl.css'}
                if u.path.strip('/') in files:
                    name=files[u.path.strip('/')]
                    kind='text/html' if name.endswith('html') else 'text/css' if name.endswith('css') else 'text/javascript'
                    return self.send((STATIC/name).read_bytes(),kind+'; charset=utf-8')
                if parts[:2]==['api','rl']:
                    from experiments.planar_tasks.semantic_rl.console import campaigns,detail,replay
                    if len(parts)==2:return self.send(campaigns(rl_root))
                    if len(parts)==3:return self.send(detail(rl_root,parts[2]))
                    if len(parts)==5 and parts[3]=='replay':return self.send(replay(rl_root,parts[2],parts[4]))
                    raise ValueError('route')
                if parts==['api','campaigns']:
                    entries=[]
                    for p in sorted(root.glob('*/campaign.json')):
                        if not p.resolve().is_relative_to(root):continue
                        try:
                            s=read(p)
                            entries.append(dict(id=p.parent.name,**{k:s[k] for k in ('created','status','model','used','limits','rounds')}))
                        except (ValueError,KeyError):continue
                    return self.send(entries)
                if len(parts)<3 or parts[:2]!=['api','campaigns']:raise ValueError('route')
                p=(root/identifier(parts[2])).resolve()
                if not p.is_relative_to(root):raise ValueError('path')
                if len(parts)==3:
                    s=read(p/'campaign.json')
                    s['round_details']=[read(p/'rounds'/identifier(r)/'round.json') for r in s['rounds']]
                    return self.send(s)
                if len(parts)==5 and parts[3]=='fits':return self.send(read(p/'fits'/(identifier(parts[4])+'.json')))
                if len(parts)==5 and parts[3]=='semantics':
                    return self.send(dict(source=(p/'semantics'/(identifier(parts[4])+'.py')).read_text()))
                if len(parts)>=5 and parts[3]=='rounds':
                    rp=p/'rounds'/identifier(parts[4])
                    if len(parts)==5:
                        r=read(rp/'round.json');r['fixture']=read(rp/'fixture.json')
                        if (rp/'scene.json').exists():r['scene']=read(rp/'scene.json')
                        else:
                            # Early development artifacts predate the standalone
                            # initial-scene file. Preserve them without rewriting.
                            first=next((rp/'attempts').glob('*/replay.json.gz'))
                            r['scene']=unpack(first)['scene']
                        r['initial_public']=read(rp/'initial_public.json')
                        return self.send(r)
                    if len(parts)==8 and parts[5]=='attempts':
                        ap=rp/'attempts'/identifier(parts[6]);kind=parts[7]
                        if kind=='replay':return self.send(unpack(ap/'replay.json.gz'))
                        if kind=='decisions':
                            start=int(q.get('start',['0'])[0]);count=int(q.get('count',['1'])[0])
                            if start<0 or not 1<=count<=5:raise ValueError('decision pagination')
                            rows=unpack(ap/'decisions.json.gz')
                            return self.send(dict(total=len(rows),start=start,decisions=rows[start:start+count]))
                        if kind=='trace':
                            start=int(q.get('start',['0'])[0]);count=int(q.get('count',['1'])[0])
                            if start<0 or not 1<=count<=5:raise ValueError('trace pagination')
                            rows=unpack(ap/'public.json.gz')
                            return self.send(dict(total=len(rows),start=start,rows=rows[start:start+count]))
                        if kind in ('request','summary','progress'):return self.send(read(ap/(kind+'.json')))
                raise ValueError('route')
            except (OSError,ValueError,KeyError,IndexError,StopIteration):self.send(dict(error='Artifact not available or invalid request'),status=404)
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--port',type=int,default=8767);a=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',a.port),handler_for(a.root))
    print(f'Relational Pipeline: http://127.0.0.1:{a.port}/',flush=True);server.serve_forever()


if __name__=='__main__':main()
