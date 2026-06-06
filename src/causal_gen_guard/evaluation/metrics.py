'''Metrics for binary behavior anomaly detection evaluation.'''

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
except Exception:  # pragma: no cover - fallback path is tested when sklearn is absent.
    average_precision_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None


def _arrays(y_true: Any, y_score: Any = None, y_pred: Any = None) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = None if y_score is None else np.asarray(y_score, dtype=np.float64).reshape(-1)
    preds = None if y_pred is None else np.asarray(y_pred, dtype=np.int64).reshape(-1)
    return true, scores, preds


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def precision(y_true: Any, y_pred: Any) -> float:
    true, _, preds = _arrays(y_true, y_pred=y_pred)
    if precision_score is not None:
        return float(precision_score(true, preds, zero_division=0))
    tp, fp, _, _ = _binary_counts(true, preds)
    return float(tp / max(tp + fp, 1))


def recall(y_true: Any, y_pred: Any) -> float:
    true, _, preds = _arrays(y_true, y_pred=y_pred)
    if recall_score is not None:
        return float(recall_score(true, preds, zero_division=0))
    tp, _, _, fn = _binary_counts(true, preds)
    return float(tp / max(tp + fn, 1))


def f1(y_true: Any, y_pred: Any) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return float(2.0 * p * r / max(p + r, 1e-12))


def fpr(y_true: Any, y_pred: Any) -> float:
    true, _, preds = _arrays(y_true, y_pred=y_pred)
    _, fp, tn, _ = _binary_counts(true, preds)
    return float(fp / max(fp + tn, 1))


def fnr(y_true: Any, y_pred: Any) -> float:
    true, _, preds = _arrays(y_true, y_pred=y_pred)
    tp, _, _, fn = _binary_counts(true, preds)
    return float(fn / max(tp + fn, 1))


def auroc(y_true: Any, y_score: Any) -> float:
    true, scores, _ = _arrays(y_true, y_score=y_score)
    if len(np.unique(true)) < 2:
        return float('nan')
    if roc_auc_score is not None:
        return float(roc_auc_score(true, scores))
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def auprc(y_true: Any, y_score: Any) -> float:
    true, scores, _ = _arrays(y_true, y_score=y_score)
    if len(np.unique(true)) < 2:
        return float('nan')
    if average_precision_score is not None:
        return float(average_precision_score(true, scores))
    order = np.argsort(-scores)
    sorted_true = true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision_curve = tp / np.maximum(tp + fp, 1)
    recall_curve = tp / max(int(np.sum(true == 1)), 1)
    precision_curve = np.concatenate([[1.0], precision_curve])
    recall_curve = np.concatenate([[0.0], recall_curve])
    return float(np.trapz(precision_curve, recall_curve))


def compute_binary_metrics(y_true: Any, y_score: Any, threshold: float) -> Dict[str, float]:
    true, scores, _ = _arrays(y_true, y_score=y_score)
    preds = (scores > float(threshold)).astype(np.int64)
    return {
        'precision': precision(true, preds),
        'recall': recall(true, preds),
        'f1': f1(true, preds),
        'fpr': fpr(true, preds),
        'fnr': fnr(true, preds),
        'auroc': auroc(true, scores),
        'auprc': auprc(true, scores),
    }
