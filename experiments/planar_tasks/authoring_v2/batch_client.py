"""Development-only batch RPC; all prefixes and branches are charged execution."""
from client import DevelopmentEnv

class BatchEnv(DevelopmentEnv):
    def branches(self, task, prefix, candidates):
        return self._call('run_branches',task=task,prefix=prefix,candidates=candidates)
