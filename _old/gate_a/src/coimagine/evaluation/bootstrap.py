from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    mean: float
    lower: float
    upper: float
    count: int


def bootstrap_mean(
    values: list[float] | tuple[float, ...],
    *,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapInterval:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return BootstrapInterval(float("nan"), float("nan"), float("nan"), 0)
    if array.size == 1:
        value = float(array[0])
        return BootstrapInterval(value, value, value, 1)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, array.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        mean=float(array.mean()),
        lower=float(np.quantile(draws, alpha)),
        upper=float(np.quantile(draws, 1.0 - alpha)),
        count=int(array.size),
    )
