"""Predict entire prefixes in an independent hypothesis; charge all replay work."""
import time


class ReplayModel:
    def __init__(self, client, hypothesis):
        self.client=client;self.hypothesis=hypothesis
        self.calls=0;self.requested_ticks=0;self.cleanup_errors=[]

    def evaluate(self, plan, deadline):
        if self.hypothesis is None:raise ValueError('state hypothesis unavailable')
        handle=None
        def check():
            if time.monotonic()>=deadline:raise TimeoutError('search model deadline')
        try:
            check();self.calls+=1;result=self.client.create(self.hypothesis);handle=result['handle']
            for action in plan:
                check()
                if result.get('termination') is not None:
                    raise ValueError('prediction terminated before complete requested prefix')
                self.calls+=1;self.requested_ticks+=action['ticks']
                result=self.client.step(handle,action)
            return result
        finally:
            if handle is not None:
                self.calls+=1
                try:self.client.close(handle)
                except Exception as exc:self.cleanup_errors.append(str(exc))
