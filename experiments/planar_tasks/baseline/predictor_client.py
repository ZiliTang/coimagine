"""Standard-library client. All state belongs to an independent hypothetical world."""
import json
import sys


class Predictor:
    def _call(self, operation, **fields):
        request = dict(method='predict', operation=operation, **fields)
        sys.__stdout__.write('__COIMAGINE_RPC__' + json.dumps(request, allow_nan=False) + '\n')
        sys.__stdout__.flush()
        line = sys.__stdin__.readline(1048577)
        if not line.endswith('\n') or len(line.encode()) > 1048576:
            raise RuntimeError('prediction service disconnected or oversized response')
        reply = json.loads(line)
        if not reply.get('ok'):
            raise RuntimeError(str(reply.get('error')))
        return reply['result']

    def create(self, hypothesis):
        return self._call('create', hypothesis=hypothesis)

    def step(self, handle, action):
        return self._call('step', handle=handle, action=action)

    def fork(self, handle):
        return self._call('fork', handle=handle)

    def close(self, handle):
        return self._call('close', handle=handle)

    def status(self):
        return self._call('status')
