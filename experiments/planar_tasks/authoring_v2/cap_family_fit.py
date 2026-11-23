"""Use losslessly packed feature rows for the original GA V/Q trainer's random reads."""
import numpy as np

from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_learner
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import Store
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.resident_replay import ResidentStore


class Features:
    def __init__(self, resident, original):
        self.resident = resident
        self.original = original

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # The metrics pass is sequential; use original float32 mmap slices.
            return self.original[idx]
        return self.resident.decode(idx)


def train(path):
    train_path = path/'arrays/train'
    packed = ResidentStore(train_path, path/'packed_train')
    original = Store(train_path)
    features = Features(packed,original.arrays['x'])
    indices = np.random.default_rng(21).integers(len(original),size=128)
    assert original.arrays['x'][indices].tobytes() == features[indices].tobytes()
    # A process-local storage adapter keeps the frozen learner and its original
    # minibatch arithmetic, tensor shapes, sample order, and labels intact.
    packed.arrays['x'] = features
    original_store = full_learner.Store
    try:
        full_learner.Store = lambda source: packed if source==train_path else original_store(source)
        return full_learner.train_offline(path, 'GA', 17)
    finally:
        full_learner.Store = original_store
