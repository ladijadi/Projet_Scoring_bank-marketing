import pandas as pd

def detecter_doublons(df: pd.DataFrame) -> int:
    """
    Renvoie le nombre de lignes dupliquées dans le DataFrame
    """
    return int(df.duplicated().sum())

def resume_qualite(df: pd.DataFrame, cible: str = "y") -> pd.DataFrame:
    """
    Affiche un résumé simple de la qualité des données :
    - dimensions
    - valeurs manquantes
    - présence de la modalité 'unknown' dans les variables catégorielles
    - répartition de la cible (si présente)

    Retourne aussi un tableau 'missing' exploitable
    """
    n_lignes, n_colonnes = df.shape
    print(f"Dimensions : {n_lignes} lignes × {n_colonnes} colonnes")

    # Valeurs manquantes
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    missing_table = pd.DataFrame({"nb_manquants": missing, "%_manquants": missing_pct})
    missing_table = missing_table[missing_table["nb_manquants"] > 0]

    if missing_table.empty:
        print("Valeurs manquantes : aucune (NaN) détectée.")
    else:
        print("Valeurs manquantes :")
        print(missing_table.head())  # fonctionne dans notebook

    # Modalité "unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cible in cat_cols:
        cat_cols.remove(cible)

    unknown_info = []
    for col in cat_cols:
        taux_unknown = (df[col].astype(str).str.lower() == "unknown").mean()
        if taux_unknown > 0:
            unknown_info.append((col, round(100 * taux_unknown, 2)))

    if len(unknown_info) == 0:
        print('Modalité "unknown" : absente des variables catégorielles.')
    else:
        unknown_df = pd.DataFrame(unknown_info, columns=["variable", "%_unknown"]).sort_values("%_unknown", ascending=False)
        print('Modalité "unknown" : taux par variable (catégorielles)')
        print(unknown_df)

    # Cible
    if cible in df.columns:
        rep = df[cible].value_counts(dropna=False)
        rep_pct = df[cible].value_counts(normalize=True, dropna=False).round(4)
        print("\nRépartition de la cible :")
        print(pd.DataFrame({"effectif": rep, "proportion": rep_pct}))

    return missing_table