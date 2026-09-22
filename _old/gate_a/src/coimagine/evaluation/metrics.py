from __future__ import annotations

from math import sqrt
from typing import Mapping


def rank_agreement(values_a: Mapping[tuple, float], values_b: Mapping[tuple, float]) -> float:
    common = sorted(set(values_a) & set(values_b), key=str)
    if len(common) < 2:
        return 1.0
    concordant = 0
    total = 0
    for i, first in enumerate(common):
        for second in common[i + 1 :]:
            da = values_a[first] - values_a[second]
            db = values_b[first] - values_b[second]
            if da == 0 and db == 0:
                concordant += 1
            elif da * db > 0:
                concordant += 1
            total += 1
    return concordant / max(1, total)
