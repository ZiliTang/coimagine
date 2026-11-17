"""Comparable continuation outcomes, group splits, and atomic budget admission."""
from dataclasses import asdict, dataclass
import hashlib
from .contract import finite


@dataclass(frozen=True)
class Outcome:
    origin: str
    action: dict
    continuation: str
    budget: dict
    source: str
    status: str
    cost: float | None

    def __post_init__(self):
        if self.source not in ('execution', 'prediction'):
            raise ValueError('evidence source')
        if self.status not in ('success', 'task_failure', 'search_unknown', 'tool_error'):
            raise ValueError('outcome status')
        if self.cost is not None and (not finite(self.cost) or self.cost < 0):
            raise ValueError('nonnegative finite cost')
        if self.status in ('success', 'task_failure') and self.cost is None:
            raise ValueError('completed task outcome requires original task cost')


def compare(left, right):
    for key in ('origin', 'continuation', 'budget', 'source'):
        if getattr(left, key) != getattr(right, key):
            raise ValueError('noncomparable ' + key)
    unknown = {'search_unknown', 'tool_error'}
    if left.status in unknown or right.status in unknown:
        return dict(status='unknown', left=asdict(left), right=asdict(right))
    if left.cost == right.cost:
        return dict(status='tie', left=asdict(left), right=asdict(right))
    good, bad = (left, right) if left.cost < right.cost else (right, left)
    return dict(status='preference', preferred=asdict(good), other=asdict(bad),
                scope='cost under the recorded bounded continuation, not optimal Q')


def group_split(group, salt):
    bucket = int(hashlib.sha256((salt + ':' + group).encode()).hexdigest(), 16) % 10
    return 'train' if bucket < 6 else 'validation' if bucket < 8 else 'test'


class Ledger:
    def __init__(self, limits):
        if any(type(v) is not int or v < 0 for v in limits.values()):
            raise ValueError('integer nonnegative limits')
        self.limits = dict(limits)
        self.used = dict.fromkeys(limits, 0)

    def charge(self, **amounts):
        if any(k not in self.used or type(v) is not int or v < 0 for k, v in amounts.items()):
            raise ValueError('invalid ledger charge')
        if any(self.used[k] + v > self.limits[k] for k, v in amounts.items()):
            raise RuntimeError('budget exhausted')
        for k, v in amounts.items():
            self.used[k] += v

    def record(self):
        return dict(limits=self.limits.copy(), used=self.used.copy())
