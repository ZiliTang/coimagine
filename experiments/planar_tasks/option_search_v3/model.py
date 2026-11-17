"""Reuse live synthetic prefixes when possible; charge all reconstructed work."""
import time

class PrefixModel:
    def __init__(self,client,hypothesis):
        self.client=client;self.hypothesis=hypothesis;self.handles={}
        self.calls=0;self.requested_ticks=0;self.cleanup_errors=[];self.last_reused=False
    def cost(self,parent,action):
        if parent['id'] in self.handles:return action['ticks'],2 # step + eventual close
        prefix=parent['path_context']
        return sum(a['ticks'] for a in prefix)+action['ticks'],len(prefix)+3
    def _close(self,handle):
        self.calls+=1
        try:self.client.close(handle)
        except Exception as exc:self.cleanup_errors.append(str(exc))
    def expand(self,parent,child_id,action,deadline):
        if self.hypothesis is None:raise ValueError('legal state estimate unavailable')
        def check():
            if time.monotonic()>=deadline:raise TimeoutError('prediction deadline')
        handle=None;self.last_reused=parent['id'] in self.handles
        try:
            check()
            if self.last_reused:handle=self.handles.pop(parent['id'])
            else:
                self.calls+=1;result=self.client.create(self.hypothesis);handle=result['handle']
                for a in parent['path_context']:
                    check()
                    if result.get('termination'):raise ValueError('prefix reconstruction terminated early')
                    self.calls+=1;self.requested_ticks+=a['ticks'];result=self.client.step(handle,a)
            check();self.calls+=1;self.requested_ticks+=action['ticks'];result=self.client.step(handle,action)
            self.handles[child_id]=handle;handle=None
            return result
        finally:
            if handle is not None:self._close(handle)
    def close(self):
        for handle in list(self.handles.values()):self._close(handle)
        self.handles.clear()
