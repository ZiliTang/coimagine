"""The single target-transform interface for this round, and the visibility policy.

Everything that turns a physical quantity into a network target, or a network output back into a
physical quantity, goes through :func:`encode_target` and :func:`decode_prediction`. A model's
``forward`` returns **standardised numbers only**; it never adds a metric reference internally. The
previous round's diagnostic mixed the two (it added the metric current inside ``forward`` and then
decoded again outside), which left an extra ``sigma_delta * current`` term in the result.

Coordinate system, units and column names are fixed here so every consumer agrees:

* frame: each robot's own **option-start frame** ``R_i0``;
* units: metres for midpoints and displacements, radians for angles, m/s and rad/s for velocities;
* the midpoint columns are the two endpoints' midpoint of the **fitted segment** in ``R_i0``;
* a target with no visible patch is **never** scored as geometry. Its placeholder values must not
  enter a loss, a normalisation statistic, or a reported error.

Two parameterisations are supported for a position column ``y``:

* ``direct``   -- ``z = (y - mu) / sigma``,           ``y_hat = sigma * z + mu``
* ``residual`` -- ``delta = y - reference``,
  ``z = (delta - mu) / sigma``, ``y_hat = reference + sigma * z + mu``

``reference`` comes only from the current or an earlier **visible** observation. When the current
patch is not visible the reference is the fixed origin of ``R_i0`` (numerically zero), which turns an
invisible-to-visible sample into an ordinary regression against a fixed origin rather than a copy of
an invalid midpoint.

Note on the reference: with ``mu != 0`` a zero network output is *not* the hold prediction; the hold
prediction is ``z = -mu / sigma``. :func:`hold_output` returns that value, and :func:`apply_hold_bias`
offers an output-layer bias that makes a zero output mean "hold" exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "residual-prediction-v1/targets-1"

#: the midpoint columns inside a geometry vector: r0 mid x, r0 mid y, r1 mid x, r1 mid y
MIDPOINT_COLUMNS = (0, 1, 4, 5)
#: column names, in the order the geometry vectors use them
GEOMETRY_COLUMNS = ["r0_mid_x", "r0_mid_y", "r0_normal_sin", "r0_normal_cos",
                    "r1_mid_x", "r1_mid_y", "r1_normal_sin", "r1_normal_cos"]
#: only these two robots exist in this round
ROBOTS = (0, 1)
#: the minimum scale for a per-column statistic; a near-constant column must not be amplified
MIN_COLUMN_SIGMA = 1e-3
#: the finite default scale used when a column has no valid training label at all
DEFAULT_COLUMN_SIGMA = 1.0


@dataclass
class TransformSpec:
    """A frozen, auditable description of one target transform."""

    name: str
    parameterisation: str = "direct"          # direct | residual
    normalisation: str = "per_group"          # per_group | per_column
    reference: str = "current_visible_midpoint"  # current_visible_midpoint | origin
    columns: Tuple[int, ...] = MIDPOINT_COLUMNS
    mean: np.ndarray = field(default_factory=lambda: np.zeros(4))
    std: np.ndarray = field(default_factory=lambda: np.ones(4))
    lower_clamped_columns: Tuple[int, ...] = ()
    missing_columns: Tuple[int, ...] = ()
    version: str = VERSION

    def _describe(self) -> dict:
        return {
            "version": self.version, "name": self.name,
            "parameterisation": self.parameterisation,
            "normalisation": self.normalisation,
            "reference": self.reference,
            "columns": [GEOMETRY_COLUMNS[index] for index in self.columns],
            "mean": [float(v) for v in np.atleast_1d(self.mean)],
            "std": [float(v) for v in np.atleast_1d(self.std)],
            "lower_clamped_columns": [GEOMETRY_COLUMNS[i] for i in self.lower_clamped_columns],
            "missing_columns": [GEOMETRY_COLUMNS[i] for i in self.missing_columns],
            "min_column_sigma": MIN_COLUMN_SIGMA,
            "default_column_sigma": DEFAULT_COLUMN_SIGMA,
        }

    def descriptor(self) -> dict:
        """A JSON-safe description; the numbers here are what a checkpoint must carry."""

        payload = self._describe()
        payload["hash"] = self.hash()
        return payload

    def hash(self) -> str:
        blob = json.dumps(self._describe(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def sigma_for(self, position: int) -> float:
        values = np.atleast_1d(self.std)
        return float(values[0] if values.size == 1 else values[position])

    def mean_for(self, position: int) -> float:
        values = np.atleast_1d(self.mean)
        return float(values[0] if values.size == 1 else values[position])


def fit_transform(truth: np.ndarray, mask: np.ndarray, reference: np.ndarray, *,
                  name: str, parameterisation: str, normalisation: str,
                  reference_name: str = "current_visible_midpoint") -> TransformSpec:
    """Fit the statistics on **valid** training labels only.

    ``truth`` and ``reference`` are ``(n, 4)``: the four midpoint columns in R_i0. ``mask`` is
    ``(n, 4)`` and marks which columns are supervised. A residual target is ``truth - reference``;
    an invalid column never contributes to a statistic.
    """

    if parameterisation not in ("direct", "residual"):
        raise ValueError(parameterisation)
    if normalisation not in ("per_group", "per_column"):
        raise ValueError(normalisation)
    target = truth.astype(np.float64) if parameterisation == "direct" \
        else truth.astype(np.float64) - reference.astype(np.float64)
    valid = mask > 0
    if normalisation == "per_group":
        selected = target[valid]
        if selected.size:
            mean = np.asarray([selected.mean()])
            std = np.asarray([selected.std() or 1.0])
        else:
            mean = np.zeros(1)
            std = np.ones(1)
        return TransformSpec(name=name, parameterisation=parameterisation,
                             normalisation=normalisation, reference=reference_name,
                             mean=mean, std=std)
    mean = np.zeros(4)
    std = np.ones(4)
    clamped: List[int] = []
    missing: List[int] = []
    for column in range(4):
        keep = valid[:, column]
        if not keep.any():
            missing.append(column)
            mean[column] = 0.0
            std[column] = DEFAULT_COLUMN_SIGMA
            continue
        values = target[keep, column]
        mean[column] = float(values.mean())
        value = float(values.std())
        if value < MIN_COLUMN_SIGMA:
            clamped.append(column)
            std[column] = MIN_COLUMN_SIGMA
        else:
            std[column] = value
    return TransformSpec(name=name, parameterisation=parameterisation,
                         normalisation=normalisation, reference=reference_name,
                         mean=mean, std=std, lower_clamped_columns=tuple(clamped),
                         missing_columns=tuple(missing))


def encode_target(truth: np.ndarray, reference: np.ndarray, spec: TransformSpec) -> np.ndarray:
    """Physical midpoint columns -> standardised network targets."""

    truth = np.asarray(truth, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    delta = truth if spec.parameterisation == "direct" else truth - reference
    if spec.normalisation == "per_group":
        return (delta - float(np.atleast_1d(spec.mean)[0])) / float(np.atleast_1d(spec.std)[0])
    return (delta - spec.mean) / spec.std


def decode_prediction(output: np.ndarray, reference: np.ndarray, spec: TransformSpec) -> np.ndarray:
    """Standardised network output -> physical midpoint columns. The reference is added once."""

    output = np.asarray(output, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if spec.normalisation == "per_group":
        delta = float(np.atleast_1d(spec.std)[0]) * output + float(np.atleast_1d(spec.mean)[0])
    else:
        delta = spec.std * output + spec.mean
    if spec.parameterisation == "direct":
        return delta
    return reference + delta


def hold_output(spec: TransformSpec) -> np.ndarray:
    """The network output that decodes **exactly** to the reference (the residual parameterisation).

    With a non-zero mean a zero output is *not* hold: ``y = ref + sigma * z + mu`` equals ``ref``
    only at ``z = -mu / sigma``. Only the residual parameterisation has a reference, so this raises
    for ``direct``: there the hold baseline is a physical-space quantity, not a fixed network output.
    """

    if spec.parameterisation != "residual":
        raise ValueError("the hold output is only defined for the residual parameterisation; "
                         "the direct baseline is computed in physical space")
    if spec.normalisation == "per_group":
        return np.full(4, -float(np.atleast_1d(spec.mean)[0]) / float(np.atleast_1d(spec.std)[0]))
    return -spec.mean / spec.std


def apply_hold_bias(weight_bias: np.ndarray, spec: TransformSpec) -> np.ndarray:
    """Set a 4-column output bias so that a **zero** network output means "hold the reference".

    The effective standardised output is ``z_net + bias``; choosing ``bias = hold_output`` makes
    ``z_net = 0`` decode exactly to the reference, which is the inductive bias this round wants.
    """

    bias = np.asarray(weight_bias, dtype=np.float64).copy()
    if spec.parameterisation != "residual":
        return bias  # nothing to bias: a direct head has no reference to fall back to
    target = hold_output(spec)
    if spec.normalisation == "per_group":
        bias[:] = float(np.atleast_1d(target)[0])
    else:
        bias[:] = target
    return bias


def visible_reference(current_midpoint: np.ndarray, current_visible: np.ndarray) -> np.ndarray:
    """The reference used by the residual transform.

    Visible -> the current midpoint. Invisible -> the fixed origin of R_i0 (numerically zero), so an
    invisible-to-visible sample is a regression against a fixed origin and never a copy of an invalid
    midpoint. The decision uses the **current** visibility only; no future field is read.
    """

    current = np.asarray(current_midpoint, dtype=np.float64)
    visible = np.asarray(current_visible) > 0.5
    if visible.ndim == 1:
        visible = visible[:, None]
    if visible.shape[1] == 2 and current.shape[1] == 4:
        # per-robot visibility -> per-column: columns 0,1 are robot 0 and 2,3 are robot 1
        visible = np.concatenate([np.repeat(visible[:, 0:1], 2, axis=1),
                                  np.repeat(visible[:, 1:2], 2, axis=1)], axis=1)
    return np.where(visible, current, 0.0)


#: the four visibility transitions the report must count separately
VISIBILITY_TRANSITIONS = ("sustained_visible", "became_visible", "became_invisible",
                          "sustained_invisible")


def visibility_transition(current_visible: np.ndarray, target_visible: np.ndarray,
                          robot: int) -> np.ndarray:
    """Per-sample transition label for one robot, using only current and target visibility."""

    current = np.asarray(current_visible)[:, robot] > 0.5
    target = np.asarray(target_visible)[:, robot] > 0.5
    out = np.empty(len(current), dtype=object)
    out[current & target] = "sustained_visible"
    out[~current & target] = "became_visible"
    out[current & ~target] = "became_invisible"
    out[~current & ~target] = "sustained_invisible"
    return out


def motion_bin(speed: np.ndarray) -> np.ndarray:
    """Robot **translation** speed bins (not the object's), as required by the task book."""

    speed = np.asarray(speed, dtype=np.float64)
    out = np.empty(len(speed), dtype=object)
    out[speed < 0.01] = "near_static(<0.01)"
    out[(speed >= 0.01) & (speed < 0.05)] = "slow(0.01-0.05)"
    out[speed >= 0.05] = "moving(>=0.05)"
    return out


# ---------------------------------------------------------------------------
# the full 8-column geometry transform used by the multi-task model
# ---------------------------------------------------------------------------


@dataclass
class GeometryTransform:
    """The 8-column geometry transform: 4 midpoints (optionally residual) + 4 normal components.

    Only the midpoint columns change parameterisation this round. The surface normal keeps the
    round-6 per-group standardisation, and angles, velocities and the classification heads are
    untouched, so the comparison isolates the midpoint parameterisation.
    """

    midpoint: TransformSpec
    normal_mean: np.ndarray = field(default_factory=lambda: np.zeros(1))
    normal_std: np.ndarray = field(default_factory=lambda: np.ones(1))
    version: str = VERSION

    def descriptor(self) -> dict:
        return {"version": self.version, "midpoint": self.midpoint.descriptor(),
                "normal_mean": [float(v) for v in np.atleast_1d(self.normal_mean)],
                "normal_std": [float(v) for v in np.atleast_1d(self.normal_std)],
                "midpoint_columns": [GEOMETRY_COLUMNS[i] for i in self.midpoint.columns]}

    def hash(self) -> str:
        blob = json.dumps(self.descriptor(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()


def fit_geometry_transform(geometry_truth: np.ndarray, geometry_mask: np.ndarray,
                           reference: np.ndarray, *, midpoint: TransformSpec,
                           name: str = "geometry") -> GeometryTransform:
    """Fit the normal-component statistics on visible training labels only."""

    mask = np.asarray(geometry_mask) > 0
    selected = np.asarray(geometry_truth, dtype=np.float64)[mask]
    if selected.size:
        mean = np.asarray([float(selected.mean())])
        std = np.asarray([float(selected.std()) or 1.0])
    else:
        mean = np.zeros(1)
        std = np.ones(1)
    del reference, name
    return GeometryTransform(midpoint=midpoint, normal_mean=mean, normal_std=std)


def encode_geometry(geometry: np.ndarray, reference: np.ndarray,
                    transform: GeometryTransform) -> np.ndarray:
    """8-column physical geometry -> standardised targets."""

    geometry = np.asarray(geometry, dtype=np.float64)
    out = np.empty_like(geometry)
    columns = list(transform.midpoint.columns)
    out[:, columns] = encode_target(geometry[:, columns], reference, transform.midpoint)
    normal_columns = [i for i in range(8) if i not in columns]
    out[:, normal_columns] = ((geometry[:, normal_columns]
                               - float(np.atleast_1d(transform.normal_mean)[0]))
                              / float(np.atleast_1d(transform.normal_std)[0]))
    return out


def decode_geometry(output: np.ndarray, reference: np.ndarray,
                    transform: GeometryTransform) -> np.ndarray:
    """8-column standardised output -> physical geometry. The reference is added exactly once."""

    output = np.asarray(output, dtype=np.float64)
    single = output.ndim == 1
    if single:
        output = output[None, :]
        reference = np.asarray(reference, dtype=np.float64)
        # a single output pairs with a single reference row; do not flatten a whole batch
        reference = reference[None, :] if reference.ndim == 1 else reference[:1]
    out = np.empty_like(output)
    columns = list(transform.midpoint.columns)
    out[:, columns] = decode_prediction(output[:, columns], reference, transform.midpoint)
    normal_columns = [i for i in range(8) if i not in columns]
    out[:, normal_columns] = (output[:, normal_columns]
                              * float(np.atleast_1d(transform.normal_std)[0])
                              + float(np.atleast_1d(transform.normal_mean)[0]))
    return out[0] if single else out


def geometry_hold_output(transform: GeometryTransform) -> np.ndarray:
    """The 8-column output that decodes to the reference midpoint and the mean normal."""

    out = np.zeros(8)
    if transform.midpoint.parameterisation == "residual":
        out[list(transform.midpoint.columns)] = hold_output(transform.midpoint)
    return out
