import numpy as np
import pandas as pd


def f1_smart(y_true, y_pred) -> tuple[float, float]:
    """
    Smart calculation of F1 score that should be fast.

    Returns `max_f1, best_threshold`.
    """
    args = np.argsort(y_pred)
    tp = y_true.sum()
    fs = (tp - np.cumsum(y_true[args[:-1]])) / np.arange(y_true.shape[0] + tp - 1, tp, -1)
    res_idx = np.argmax(fs)
    max_f1 = 2 * fs[res_idx]
    best_threshold = (y_pred[args[res_idx]] + y_pred[args[res_idx + 1]]) / 2
    return max_f1, best_threshold


def print_confusion_matrix(cm, title="Validation results"):
    cm_labeled = pd.DataFrame(
        {
            "Predicted Negative": {"Actual Negative": f"TN: {cm[0, 0]}", "Actual Positive": f"FN: {cm[1, 0]}"},
            "Predicted Positive": {"Actual Negative": f"FP: {cm[0, 1]}", "Actual Positive": f"TP: {cm[1, 1]}"},
        }
    )
    print("-" * 60)
    print(title)
    print("-" * 60)
    print(cm_labeled)
    print("-" * 60)

    pr = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    rc = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    f1 = 2 * pr * rc / (pr + rc)
    acc = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    print(f"F1 score: {round(f1, 4)}")
    print(f"Accuracy: {round(acc, 4)}")
    print("-" * 60)
