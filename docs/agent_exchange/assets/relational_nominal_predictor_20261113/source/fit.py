"""Small weighted pairwise ranker; missingness explicit, normalization train-only."""
from dataclasses import dataclass
import numpy as np


def array(rows):
    x = np.asarray([[np.nan if v is None else v for v in row] for row in rows], dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or np.isinf(x).any():
        raise ValueError('nonempty rectangular finite-or-missing features')
    return x


@dataclass
class Ranker:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def transform(self, rows):
        x = array(rows)
        if x.shape[1] != len(self.mean):
            raise ValueError('feature dimension changed')
        missing = np.isnan(x)
        x = np.where(missing, self.mean, x)
        return np.concatenate(((x - self.mean) / self.scale, missing.astype(float)), axis=1)

    def score(self, rows):
        return self.transform(rows) @ self.weights

    def record(self):
        return dict(mean=self.mean.tolist(), scale=self.scale.tolist(), weights=self.weights.tolist())

    @classmethod
    def restore(cls, record):
        model = cls(*(np.asarray(record[k], dtype=float) for k in ('mean', 'scale', 'weights')))
        if (model.mean.ndim != 1 or model.scale.shape != model.mean.shape or
                model.weights.shape != (2 * len(model.mean),) or
                not all(np.isfinite(v).all() for v in (model.mean, model.scale, model.weights)) or
                (model.scale <= 0).any()):
            raise ValueError('invalid ranker checkpoint')
        return model


def fit(preferred, other, *, confidence=None, epochs=200, learning_rate=.1,
        ridge=.001, offsets=None, residual_penalty=0.):
    """Offsets are frozen semantic score differences when fitting a residual."""
    pos, neg = array(preferred), array(other)
    if pos.shape != neg.shape:
        raise ValueError('paired feature shape')
    if type(epochs) is not int or not 1 <= epochs <= 200:
        raise ValueError('bounded training epochs')
    if not all(np.isfinite(v) for v in (learning_rate, ridge, residual_penalty)) or learning_rate <= 0 or min(ridge, residual_penalty) < 0:
        raise ValueError('training configuration')
    x = np.concatenate((pos, neg))
    counts = np.sum(~np.isnan(x), axis=0)
    mean = np.nansum(x, axis=0) / np.maximum(counts, 1)
    scale = np.sqrt(np.sum(np.where(np.isnan(x), 0., (x - mean)**2), axis=0) / np.maximum(counts, 1))
    scale = np.where(scale < 1e-6, 1., scale)
    model = Ranker(mean, scale, np.zeros(2 * x.shape[1]))
    d = model.transform(preferred) - model.transform(other)
    w = np.ones(len(pos)) if confidence is None else np.asarray(confidence, float)
    base = np.zeros(len(pos)) if offsets is None else np.asarray(offsets, float)
    if w.shape != (len(pos),) or base.shape != w.shape or not np.isfinite(w).all() or not np.isfinite(base).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('finite pair weights and frozen offsets required')
    w = w / w.sum()
    losses = []
    for _ in range(epochs):
        contribution = d @ model.weights
        margin = base + contribution
        loss = np.sum(w * np.logaddexp(0., -margin)) + ridge * np.sum(model.weights**2) / 2
        loss += residual_penalty * np.sum(w * np.abs(contribution))
        losses.append(float(loss))
        probability = np.exp(-np.logaddexp(0., margin))
        grad = d.T @ (-w * probability + residual_penalty * w * np.sign(contribution)) + ridge * model.weights
        model.weights -= learning_rate * grad
    if not np.isfinite(model.weights).all():
        raise ValueError('nonfinite fit')
    return model, losses
