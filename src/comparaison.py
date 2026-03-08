import numpy as np
import pandas as pd


def tableau_comparatif(resultats: list[dict]) -> pd.DataFrame:
    """
    Construit un tableau comparatif à partir d'une liste de dictionnaires de résultats de modèles
    Chaque dictionnaire doit contenir : nom_modele + métriques
    """
    df = pd.DataFrame(resultats).copy()

    # Réorganiser les colonnes : priorité aux métriques d'évaluation
    ordre = ["modele", "accuracy", "precision", "recall", "f1", "roc_auc"]
    cols = [c for c in ordre if c in df.columns] + [c for c in df.columns if c not in ordre]
    df = df[cols]

    # Tri : priorité au meilleur ROC-AUC, puis F1 (si présents)
    if "roc_auc" in df.columns:
        df = df.sort_values(by=["roc_auc", "f1"], ascending=False, na_position="last")
    elif "f1" in df.columns:
        df = df.sort_values(by="f1", ascending=False, na_position="last")

    return df.reset_index(drop=True)