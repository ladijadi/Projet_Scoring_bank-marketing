from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _sauvegarder(fig, save_path: Path | None):
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_cible_repartition(df: pd.DataFrame, cible: str = "y", save_path: Path | None = None):
    """
    Barplot simple de la distribution de la cible.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ordre = df[cible].value_counts().index.tolist()
    sns.countplot(data=df, x=cible, order=ordre, ax=ax)
    ax.set_title("Répartition de la variable cible")
    ax.set_xlabel(cible)
    ax.set_ylabel("Effectif")

    _sauvegarder(fig, save_path)
    plt.show()


def plot_distribution_age(df: pd.DataFrame, cible: str = "y", save_path: Path | None = None):
    """
    Distribution de l'âge selon la cible (2 histogrammes superposés).
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # On force une copie de colonnes nécessaires
    tmp = df[[cible, "age"]].dropna()

    for modalite in sorted(tmp[cible].unique()):
        sns.histplot(
            data=tmp[tmp[cible] == modalite],
            x="age",
            bins=30,
            kde=False,
            stat="density",
            label=str(modalite),
            ax=ax
        )

    ax.set_title("Distribution de l'âge selon la cible")
    ax.set_xlabel("Âge")
    ax.set_ylabel("Densité")
    ax.legend(title=cible)

    _sauvegarder(fig, save_path)
    plt.show()


def plot_heatmap_correlation(df: pd.DataFrame, variables_numeriques: list[str], save_path: Path | None = None):
    """
    Heatmap de corrélation (Pearson) sur les variables numériques.
    """
    corr = df[variables_numeriques].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, cmap="viridis", ax=ax)
    ax.set_title("Corrélation (Pearson) — variables numériques")

    _sauvegarder(fig, save_path)
    plt.show()