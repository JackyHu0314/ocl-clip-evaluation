"""Whole-dataset Top-k accuracy and OCL-compatible macro AP."""

import numpy as np
from sklearn.metrics import average_precision_score


def topk_accuracy(scores, targets, ks=(1, 3, 5)):
    scores, targets = np.asarray(scores), np.asarray(targets)
    if scores.ndim != 2 or targets.shape != (len(scores),) or len(scores) == 0:
        raise ValueError("Expected nonempty scores [N,C] and targets [N]")
    if not np.isfinite(scores).all() or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Scores must be finite and targets must be integer class indices")
    if np.any((targets < 0) | (targets >= scores.shape[1])):
        raise ValueError("Target class out of range")
    ks = tuple(ks)
    if not ks or any(type(k) is not int or not 1 <= k <= scores.shape[1] for k in ks):
        raise ValueError("Every k must be between 1 and the number of candidate classes")
    # Stable tie handling: earlier candidate/category ID wins equal scores.
    ranks = np.argsort(-scores, axis=1, kind="stable")
    return {f"top{k}": float(np.mean(np.any(ranks[:, :k] == targets[:, None], axis=1)))
            for k in ks}


def multilabel_map(scores, targets):
    scores, targets = np.asarray(scores), np.asarray(targets)
    if scores.ndim != 2 or scores.shape != targets.shape or 0 in scores.shape:
        raise ValueError("Expected nonempty matching scores and targets [N,L]")
    if not np.isfinite(scores).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Scores must be finite and targets must be binary")
    positives = targets.sum(axis=0).astype(int)
    # Match OCL's evaluator: no-positive labels have undefined AP and are excluded.
    ap = [float(average_precision_score(targets[:, i], scores[:, i])) if count else None
          for i, count in enumerate(positives)]
    valid = [value for value in ap if value is not None]
    return {
        "mAP": float(np.mean(valid)) if valid else None,
        "per_label_ap": ap,
        "positive_counts": positives.tolist(),
        "evaluated_labels": len(valid),
        "excluded_no_positive_labels": [i for i, value in enumerate(ap) if value is None],
    }
