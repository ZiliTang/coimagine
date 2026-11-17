"""Local read-only research console; no synthesis, simulator or API transport import."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .catalog import Catalog, inside

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / 'static'
TESTBED = ROOT / 'experiments/option_rl_v19/testbed'


def handler_for(catalog):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, data, content_type='application/json; charset=utf-8', status=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'self'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self.send({'error': 'Read-only console: no run, edit, delete, or model-call endpoints.'}, status=405)

        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                path, query = parsed.path, parse_qs(parsed.query)
                if path == '/':
                    return self.send((STATIC / 'index.html').read_bytes(), 'text/html; charset=utf-8')
                if path in ('/console.js', '/console.css'):
                    return self.send((STATIC / path[1:]).read_bytes(),
                        'text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
                if path == '/api/experiments':
                    catalog.scan()
                    return self.send({'runs': catalog.list_runs(), 'evaluations': catalog.list_evaluations()})
                if path.startswith('/api/experiments/'):
                    parts = path.split('/')[3:]
                    run_id = parts[0]
                    if len(parts) == 1:
                        result = catalog.detail(run_id)
                        result['evaluations'] = catalog.list_evaluations(run_id)
                        return self.send(result)
                    if parts[1] == 'events' and len(parts) == 2:
                        start = int(query.get('start', ['0'])[0]);count = int(query.get('count', ['100'])[0])
                        if start < 0 or not 1 <= count <= 200:
                            raise ValueError('invalid page range')
                        return self.send(catalog.events(run_id, start, count))
                    if parts[1] == 'policy' and len(parts) == 3:
                        return self.send(catalog.policy(run_id, parts[2]))
                    if parts[1] == 'diff' and len(parts) == 2:
                        return self.send(catalog.diff(run_id, query['before'][0], query['after'][0]))
                if path == '/replay':
                    # Preserve the shared renderer, hide all editing affordances and
                    # deny mutations at the server boundary, including manual requests.
                    html = (TESTBED / 'index.html').read_text()
                    html = html.replace('</head>', '<link rel="stylesheet" href="/replay-readonly.css"></head>')
                    html = html.replace('</body>', '<script src="/replay-readonly.js"></script></body>')
                    return self.send(html.encode(), 'text/html; charset=utf-8')
                if path in ('/style.css', '/app.js'):
                    return self.send((TESTBED / path[1:]).read_bytes(),
                        'text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
                if path in ('/replay-readonly.js', '/replay-readonly.css'):
                    return self.send((STATIC / path[1:]).read_bytes(),
                        'text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
                if path == '/api/status':
                    return self.send({'controller_source_sha256': 'artifact-replay', 'runtime_source_sha256': 'read-only',
                                      'restart_required': False, 'pid': None})
                if path in ('/api/params', '/api/presets', '/api/overrides', '/api/runs'):
                    return self.send([])
                if path == '/api/options':
                    # Labels only. Do not import live controllers or instantiate an environment.
                    labels = []
                    for direction, start in zip(('E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'), range(0, 32, 4)):
                        labels += [{'action_index': start + i, 'label': f'PUSH_{direction}[v={speed:.2f}m/s]'}
                                   for i, speed in enumerate((.1, .2, .3, .4))]
                    labels += [{'action_index': i, 'label': name} for i, name in
                               ((32, 'ROTATE_CCW slow'), (33, 'ROTATE_CCW'), (34, 'ROTATE_CCW fast'),
                                (35, 'ROTATE_CW slow'), (36, 'ROTATE_CW'), (37, 'ROTATE_CW fast'),
                                (40, 'REPOSITION_CCW'), (43, 'REPOSITION_CW'), (44, 'BRAKE_HOLD'), (46, 'YIELD_CLEARANCE'))]
                    return self.send(labels)
                if path.startswith('/api/runs/'):
                    replay_id = path.split('/')[-1]
                    if replay_id not in catalog.replays:
                        # Direct replay bookmarks also work after server restart.
                        catalog.scan()
                        for run_id in catalog.runs:
                            catalog.episodes(run_id)
                        catalog.list_evaluations()
                    replay = inside(catalog.root, catalog.replays[replay_id])
                    return self.send(replay.read_bytes())
                self.send({'error': 'Not found'}, status=404)
            except (KeyError, ValueError, OSError) as exc:
                self.send({'error': type(exc).__name__ + ': unavailable artifact or invalid request'}, status=404)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/agentic')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    catalog = Catalog(args.root).scan()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(catalog))
    print(f'Read-only research console: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
