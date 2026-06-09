"""Shared evaluation metrics for return prediction experiments."""

from typing import Dict, Iterable, List

import numpy as np


def signal(value: float, threshold: float = 0.5) -> str:
    """Convert a return percentage into a Buy/Sell/Hold signal."""
    if value > threshold:
        return "BUY"
    if value < -threshold:
        return "SELL"
    return "HOLD"


def regression_metrics(preds: Iterable[float], labels: Iterable[float]) -> Dict[str, float]:
    """Compute regression and direction metrics for predicted returns."""
    pred_arr = np.asarray(list(preds), dtype=float)
    label_arr = np.asarray(list(labels), dtype=float)
    errors = pred_arr - label_arr
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "directional_accuracy": float(np.mean((pred_arr >= 0) == (label_arr >= 0))),
    }


def signal_accuracy(
    preds: Iterable[float],
    labels: Iterable[float],
    thresholds: Iterable[float] = (0.5, 1.0, 2.0, 3.0),
) -> List[Dict[str, float]]:
    """Compute Buy/Sell/Hold agreement for several thresholds."""
    pred_arr = np.asarray(list(preds), dtype=float)
    label_arr = np.asarray(list(labels), dtype=float)
    rows = []
    for threshold in thresholds:
        correct = sum(
            signal(pred, threshold) == signal(label, threshold)
            for pred, label in zip(pred_arr, label_arr)
        )
        rows.append(
            {
                "threshold": float(threshold),
                "correct": int(correct),
                "total": int(len(pred_arr)),
                "accuracy": float(correct / len(pred_arr)),
            }
        )
    return rows
