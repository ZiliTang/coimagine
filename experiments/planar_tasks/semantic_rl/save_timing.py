"""Visible wall-clock accounting for serialization and file writes."""
import time


class SaveTiming:
    def __init__(self):self.seconds=0.;self.calls=0

    def call(self,fn,*args,**kwargs):
        start=time.perf_counter()
        try:return fn(*args,**kwargs)
        finally:self.seconds+=time.perf_counter()-start;self.calls+=1

    def log(self,label,**fields):
        print('[storage]',label,*(str(k)+'='+str(v) for k,v in fields.items()),
              f'save_seconds={self.seconds:.6f}',f'save_calls={self.calls}',flush=True)
