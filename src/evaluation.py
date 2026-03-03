from typing import Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluer_classification_binaire(y_true, y_pred, y_proba=None, pos_label="yes") -> Dict[str, float]:
    """
    Calcule des métriques adaptées au cas binaire déséquilibré.
    Si y_proba est fourni, calcule ROC-AUC.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label)
    }

    if y_proba is not None:
        # y_proba = proba de la classe positive
        metrics["roc_auc"] = roc_auc_score((y_true == pos_label).astype(int), y_proba)

    return {k: float(np.round(v, 4)) for k, v in metrics.items()}


def afficher_matrice_confusion(y_true, y_pred, labels=("no", "yes"), titre="Matrice de confusion"):
    """
    Affiche une matrice de confusion lisible.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(titre)
    plt.show()


def afficher_rapport_classification(y_true, y_pred):
    """
    Affiche le classification_report sklearn.
    """
    print(classification_report(y_true, y_pred))


def tracer_roc(y_true, y_proba, pos_label="yes", titre="Courbe ROC"):
    """
    Trace la courbe ROC à partir des probabilités de la classe positive.
    """
    y_bin = (pd.Series(y_true) == pos_label).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Aléatoire")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(titre)
    ax.legend()
    plt.show()