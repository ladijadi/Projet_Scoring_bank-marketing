import pandas as pd

def nettoyer_colonnes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les noms de colonnes pour éviter les caractères ambigus :
    - remplace '.' par '_'
    - supprime espaces inutiles
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(".", "_", regex=False)
    )
    return df


def retirer_duration(df: pd.DataFrame, nom_colonne: str = "duration") -> pd.DataFrame:
    """
    Retire la variable duration si elle existe (data leakage).
    """
    df = df.copy()
    if nom_colonne in df.columns:
        df = df.drop(columns=[nom_colonne])
    return df