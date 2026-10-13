"""C1: where the termination cross-entropy comes from (observation only)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..continuous2d_learning.contract import TERMINATION_CLASSES
from ..continuous2d_learning.model import JointWindowMLP, _tensors

SUCCESS = TERMINATION_CLASSES.index("success")
WINDOW = TERMINATION_CLASSES.index("window_complete")
FAILURE = TERMINATION_CLASSES.index("physical_failure")


def _family(record: Dict[str, object]) -> str:
    return str(record["scene_id"]).split("_")[0]


def termination_attribution(
    model: JointWindowMLP, records: Sequence[Dict[str, object]], *, batch_size: int = 512
) -> Dict[str, object]:
    """Per-true-class CE, true-label probability and CE contribution, plus success behaviour."""

    if not records:
        return {"n": 0, "null_reason": "empty record set"}
    tensors = _tensors(records)
    n = len(records)
    logits_all = np.zeros((n, len(TERMINATION_CLASSES)))
    true_class = np.zeros(n, dtype=np.int64)
    for index, record in enumerate(records):
        true_class[index] = TERMINATION_CLASSES.index(str(record["termination_class"]))
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            index = slice(start, min(start + batch_size, n))
            logits_all[start : index.stop] = model(tensors[0][index])["termination"].numpy()
    probabilities = torch.softmax(torch.tensor(logits_all, dtype=torch.float64), dim=1).numpy()
    rows = np.arange(n)
    per_sample_ce = -np.log(np.clip(probabilities[rows, true_class], 1e-12, None))
    predicted = probabilities.argmax(axis=1)
    families = np.asarray([_family(record) for record in records])

    def _class_table(mask: np.ndarray) -> Dict[str, object]:
        total = int(mask.sum())
        classes = {}
        for value, name in enumerate(TERMINATION_CLASSES):
            class_mask = mask & (true_class == value)
            count = int(class_mask.sum())
            classes[name] = {
                "count": count,
                "share_of_total": count / max(1, total),
                "mean_ce": float(per_sample_ce[class_mask].mean()) if count else None,
                "mean_true_label_probability": float(probabilities[class_mask, value].mean()) if count else None,
                "ce_contribution": float(per_sample_ce[class_mask].sum() / max(1, total)),
            }
        return {
            "n": total,
            "total_ce": float(per_sample_ce[mask].mean()) if total else None,
            "contribution_sum": float(sum(entry["ce_contribution"] for entry in classes.values())),
            "classes": classes,
        }

    def _success_stats(mask: np.ndarray) -> Dict[str, object]:
        true_success = int((mask & (true_class == SUCCESS)).sum())
        predicted_success = int((mask & (predicted == SUCCESS)).sum())
        true_positive = int((mask & (true_class == SUCCESS) & (predicted == SUCCESS)).sum())
        success_windows = mask & (true_class == SUCCESS)
        normal_windows = mask & (true_class == WINDOW)
        return {
            "true_success": true_success,
            "predicted_success": predicted_success,
            "success_recall": (true_positive / true_success) if true_success else None,
            "success_recall_denominator": true_success,
            "success_precision": (true_positive / predicted_success) if predicted_success else None,
            "success_precision_denominator": predicted_success,
            "true_success_predicted_window_complete_fraction": (
                float((predicted[success_windows] == WINDOW).mean()) if true_success else None
            ),
            "true_success_mean_success_probability": (
                float(probabilities[success_windows, SUCCESS].mean()) if true_success else None
            ),
            "true_normal_predicted_success_fraction": (
                float((predicted[normal_windows] == SUCCESS).mean()) if int(normal_windows.sum()) else None
            ),
            "physical_failure_windows": int((true_class == FAILURE).sum()),
            "physical_failure_note": "not validated" if int((true_class == FAILURE).sum()) == 0 else "",
        }

    everything = np.ones(n, dtype=bool)
    report: Dict[str, object] = {
        "n": n,
        "classes": list(TERMINATION_CLASSES),
        "all": {**_class_table(everything), **_success_stats(everything)},
        "families": {},
    }
    for family in sorted(set(families)):
        mask = families == family
        report["families"][family] = {**_class_table(mask), **_success_stats(mask)}
    report["identity_residual"] = float(
        report["all"]["total_ce"] - report["all"]["contribution_sum"]
    )
    return report
