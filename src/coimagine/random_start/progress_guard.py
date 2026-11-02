"""Own-observation phase progress, independent from legacy per-frame heartbeats."""
from __future__ import annotations
import math

VERSION='meaningful-progress-v1'


class ProgressGuard:
    """A plateau deadline renewed by cumulative target-direction improvement.

    A phase value is larger when better. Small positive/negative jitter cannot
    repeatedly renew the deadline; braking/ready pauses do not erase a plateau.
    """
    def __init__(self,window=3.):
        self.window=window;self.phase=None;self.marker=None;self.elapsed=0.;self.total=0.
        self.info=None

    def update(self,phase,value,dt,minimum):
        if phase is None:
            self.info=dict(version=VERSION,phase=self.phase,paused=True,elapsed=self.elapsed,window=self.window)
            return False
        if not math.isfinite(value):raise ValueError('non-finite progress evidence')
        if phase!=self.phase:
            self.phase=phase;self.marker=value;self.elapsed=0.;self.total=0.
        else:
            self.elapsed+=dt;self.total+=dt
            if value>=self.marker+minimum:
                self.marker=value;self.elapsed=0.
        stalled=self.elapsed>=self.window-1e-9
        self.info=dict(version=VERSION,phase=phase,paused=False,value=value,marker=self.marker,
            elapsed=self.elapsed,window=self.window,minimum=minimum,stalled=stalled)
        return stalled
