# Bank Marketing - Classification Supervisée

## Contexte

Ce projet s’appuie sur le dataset **Bank Marketing** issu du UCI Machine Learning Repository.

Il contient des données relatives à des campagnes de marketing direct menées par une institution bancaire portugaise entre 2008 et 2010.

L’objectif est de prédire si un client souscrit à un dépôt à terme à l’issue d’un contact téléphonique.

Référence :
Moro, S., Cortez, P., & Rita, P. (2014).  
*A Data-Driven Approach to Predict the Success of Bank Telemarketing*.  
Decision Support Systems.

---

## 🎯 Objectif

Construire un modèle de **classification supervisée binaire** permettant d’estimer :

P(Y = 1 | X = x)

où Y représente la souscription à un dépôt bancaire.

Le dataset présente un **déséquilibre de classes (~11 % de souscriptions)**.

---

## 📊 Données

- 41 176 observations (après suppression des doublons)
- 20 variables explicatives
- 1 variable cible
- Aucune valeur manquante (NaN)
- Présence de modalités `"unknown"` conservées comme catégories

La variable `duration` a été exclue afin d’éviter toute fuite d’information (data leakage).

---

## 🔎 Analyse Exploratoire

- Analyse descriptive des variables numériques
- Étude de la distribution de la cible
- Analyse des corrélations (Pearson)
- Identification de multicolinéarité potentielle entre variables macroéconomiques

Constat : l’accuracy seule n’est pas adaptée au déséquilibre du dataset.

---

## ⚙️ Méthodologie

### Pipeline de traitement

- Split stratifié (80 % / 20 %)
- Imputation (SimpleImputer)
- Standardisation (StandardScaler)
- Encodage catégoriel (OneHotEncoder)
- Pipeline scikit-learn reproductible

---

## 🤖 Modèles étudiés

### 1️⃣ Régression Logistique
- Régularisation L2
- Version standard
- Version pondérée (`class_weight="balanced"`)

### 2️⃣ Random Forest
- Modèle de base
- Optimisation légère via GridSearchCV

---

## 📈 Résultats

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Logistique (base) | 0.898 | 0.651 | 0.209 | 0.316 | 0.800 |
| Logistique (pondérée) | 0.830 | 0.359 | 0.646 | 0.462 | 0.800 |
| Random Forest (base) | 0.894 | 0.553 | 0.287 | 0.378 | 0.778 |
| Random Forest (tuned) | 0.901 | 0.645 | 0.261 | 0.372 | 0.814 |

---

## 🧠 Analyse des résultats

- Le meilleur ROC-AUC est obtenu par la Random Forest optimisée.
- Le meilleur rappel de la classe positive est obtenu par la logistique pondérée.
- La logistique offre une meilleure interprétabilité.
- Le choix final dépend de l’objectif opérationnel (détection vs précision globale).

---

## 📌 Compétences mobilisées

- Python (Pandas, NumPy, Scikit-learn)
- Pipeline sklearn
- Validation croisée stratifiée
- Gestion du déséquilibre de classes
- Évaluation multi-métriques (ROC-AUC, F1, Recall)
- Analyse de corrélation et multicolinéarité

---

## 📚 Référence scientifique

Moro, S., Cortez, P., & Rita, P. (2014).  
*A Data-Driven Approach to Predict the Success of Bank Telemarketing.*