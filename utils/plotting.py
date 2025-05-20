import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve


def plotsize(x, y):
    sns.set(rc={"figure.figsize": (x, y)})


def plot_pr_curve(true_labels, scores, *, figsize=(8, 6), xlim=(-0.02, 1.02), ylim=(-0.02, 1.02)):
    """
    Plots a Precision-Recall curve with a filled area under the curve.

    Args:
        true_labels (array-like): True binary labels.
        scores (array-like): Scores or probabilities of the positive class.
    """
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    auc_pr = auc(recall, precision)

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color="orange", label=f"PR curve (AUC = {auc_pr:.2f})")
    plt.fill_between(recall, precision, color="orange", alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def plot_pr_curve_subplot(ax, true_labels, scores, title, f1_score_val=None, threshold=None):
    precision, recall, _ = precision_recall_curve(true_labels, scores)
    auc_pr = auc(recall, precision)

    # Plot PR curve
    ax.plot(recall, precision, color="blue", label=f"PR curve (AUC = {auc_pr:.3f})")
    ax.fill_between(recall, precision, color="blue", alpha=0.2)

    # Add labels and grid
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # Modify title to include F1 score if provided
    if f1_score_val is not None and threshold is not None:
        title = f"{title}\nF1 = {f1_score_val:.3f} (thr = {threshold:.3f})"

    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)

    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
