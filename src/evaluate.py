from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def build_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    labels = list(range(num_classes))
    return confusion_matrix(y_true, y_pred, labels=labels)


def plot_confusion_matrix(cm: np.ndarray, output_path: str, title: str = "Confusion Matrix") -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def summarize_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10
) -> Tuple[Dict[str, float], np.ndarray]:
    metrics = compute_metrics(y_true, y_pred)
    cm = build_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    return metrics, cm


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    num_classes: int = 10,
) -> List[Dict]:
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    names = class_names if class_names is not None else [str(i) for i in labels]
    rows = []
    for i, lab in enumerate(labels):
        rows.append(
            {
                "class_index": int(lab),
                "class_name": names[lab] if lab < len(names) else str(lab),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
        )
    return rows


def plot_per_class_f1(per_class: List[Dict], output_path: str, title: str = "Per-class F1") -> None:
    names = [r["class_name"] for r in per_class]
    f1s = [r["f1_score"] for r in per_class]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names))
    ax.bar(x, f1s, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize_metrics_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    num_classes: int = 10,
) -> Tuple[Dict[str, float], np.ndarray, List[Dict]]:
    metrics, cm = summarize_metrics(y_true, y_pred, num_classes=num_classes)
    per_class = per_class_metrics(y_true, y_pred, class_names=class_names, num_classes=num_classes)
    return metrics, cm, per_class
