from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


@dataclass
class ColonnesModele:
    """
    Regroupe les noms de colonnes utilisées dans le modèle
    """
    cible: str
    numeriques: List[str]
    categorielles: List[str]


def construire_preprocesseur(colonnes: ColonnesModele) -> ColumnTransformer:
    """
    Construit un préprocesseur sklearn :
    - numériques : imputation simple + standardisation
    - catégorielles : imputation + one-hot encoding
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, colonnes.numeriques),
            ("cat", cat_pipeline, colonnes.categorielles),
        ]
    )
    return preprocessor


def construire_modele_logistique(
    colonnes: ColonnesModele,
    penalty: str = "l2",
    C: float = 1.0,
    class_weight=None,
    max_iter: int = 1000,
    solver: str = "liblinear"
) -> Pipeline:
    """
    Pipeline complet : prétraitement + régression logistique

    Notes :
    - solver='liblinear' supporte l1 et l2, et est adapté pour les petits datasets
     (pour les grands datasets, 'saga' est plus rapide et supporte aussi l1 et l2)
     mais 'saga' peut être plus instable, donc je choisis 'liblinear' pour ce projet
    - class_weight peut être 'balanced' pour gérer le déséquilibre.
    """
    preprocessor = construire_preprocesseur(colonnes)

    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        solver=solver,
        class_weight=class_weight,
        max_iter=max_iter
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ])
    return pipe


def separer_X_y(df: pd.DataFrame, colonnes: ColonnesModele) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare X et y à partir d'un DataFrame.
    """
    X = df[colonnes.numeriques + colonnes.categorielles].copy()
    y = df[colonnes.cible].copy()
    return X, y