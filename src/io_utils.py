from pathlib import Path
import pandas as pd


def charger_dataset(chemin: Path, sep: str = ";") -> pd.DataFrame:
    """
    Charge le dataset Bank Marketing depuis un fichier CSV

    Paramètres
    ----------
    chemin : Path
        Chemin vers le fichier CSV
    sep : str
        Séparateur du CSV (par défaut ';' pour bank-additional-full.csv)

    Retour
    ------
    pd.DataFrame
        Données chargées
    """
    if not Path(chemin).exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    df = pd.read_csv(chemin, sep=sep)
    return df