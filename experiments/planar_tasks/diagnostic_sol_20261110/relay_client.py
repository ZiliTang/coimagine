"""Restricted author entry: only incremental relay requests and tool replies."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('command', choices=('next', 'reply'), default='next', nargs='?')
    args = parser.parse_args()
    root = Path(args.root).resolve() / 'relay'
    cursor = root / 'cursor.json'
    state = json.loads(cursor.read_text()) if cursor.exists() else {'index': 0, 'messages': 0}
    if args.command == 'reply':
        calls = json.load(sys.stdin)
        if not isinstance(calls, list) or not calls:
            raise ValueError('nonempty list of tool requests required')
        n = state['index']
        if n < 1:
            raise ValueError('read a request before replying')
        path = root / f'response_{n:03d}.json'
        if path.exists():
            raise ValueError('request already answered')
        response = {'model': 'gpt-6.1-sol', 'choices': [{'finish_reason': 'tool_calls',
            'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': f'call_{n}_{i}', 'type': 'function', 'function': {
                    'name': c['name'], 'arguments': json.dumps(c['arguments'], ensure_ascii=False)}}
                for i, c in enumerate(calls)]}}]}
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(response, ensure_ascii=False)); tmp.replace(path)
    end = time.monotonic() + 25
    while time.monotonic() < end:
        path = root / f"request_{state['index'] + 1:03d}.json"
        if path.exists():
            payload = json.loads(path.read_text()); messages = payload['messages']
            view = {'request_index': state['index'] + 1, 'model': payload['model'],
                    'reasoning_effort_requested': payload.get('reasoning_effort'),
                    'max_tokens_requested_not_metered': payload['max_tokens'],
                    'tool_choice': payload['tool_choice'], 'tools': payload['tools'],
                    'budget_message': messages[1], 'new_messages': messages[state['messages']:]}
            cursor.write_text(json.dumps({'index': state['index'] + 1, 'messages': len(messages)}))
            print(json.dumps(view, ensure_ascii=False)); return
        execution = root.parent / 'execution.json'
        if execution.exists() and json.loads(execution.read_text()).get('phase') == 'finished':
            print('CAMPAIGN CLOSED. Do not inspect results or continue.'); return
        time.sleep(.1)
    print('WAITING: call next again; no new request available.')


if __name__ == '__main__':
    main()
