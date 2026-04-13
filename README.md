# 📊 Bank Marketing – Customer Subscription Prediction

## 🎯 Objective

Build a supervised classification model to predict whether a client subscribes to a term deposit following a marketing campaign.

The dataset presents a strong **class imbalance (~11% positive class)**, making standard accuracy insufficient.

---

## 📁 Dataset

- 41,176 observations  
- 20 features  
- No missing values  
- "unknown" values kept as meaningful categories  

⚠️ The variable `duration` was removed to avoid data leakage.

---

## ⚙️ Methodology

### Data Pipeline

- Stratified train/test split (80/20)
- Imputation (SimpleImputer)
- Scaling (StandardScaler)
- Categorical encoding (OneHotEncoder)
- Full sklearn pipeline for reproducibility

---

## 🤖 Models

### Logistic Regression
- L2 regularization  
- Standard version  
- Weighted version (`class_weight="balanced"`)

### Random Forest
- Baseline model  
- Light tuning (GridSearchCV)

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Logistic (baseline) | 0.898 | 0.651 | 0.209 | 0.316 | 0.800 |
| Logistic (weighted) | 0.830 | 0.359 | 0.646 | 0.462 | 0.800 |
| Random Forest (baseline) | 0.894 | 0.553 | 0.287 | 0.378 | 0.778 |
| Random Forest (tuned) | 0.901 | 0.645 | 0.261 | 0.372 | 0.814 |

---

## 🧠 Key Insights

- Class imbalance significantly impacts model performance  
- Logistic regression (weighted) improves **recall** → better detection of target clients  
- Random Forest achieves higher **AUC** → better global discrimination  

➡️ **No single best model — choice depends on business objective (detection vs precision)**

---

## 💼 Business Interpretation

This project highlights a common trade-off in marketing analytics:

- High recall → identify more potential subscribers  
- High precision → reduce unnecessary campaign costs  

➡️ Model selection must align with campaign strategy and budget constraints.

## 📊 Visual Highlights

<img width="1872" height="1003" alt="Screenshot 2026-04-13 102230" src="https://github.com/user-attachments/assets/44720b77-2896-48f7-8622-f251028ef68b" />

Model shows good discrimination ability (AUC ~0.81), but performance must be interpreted with class imbalance in mind.

<img width="1901" height="998" alt="Screenshot 2026-04-13 102339" src="https://github.com/user-attachments/assets/36a788d4-e6db-49d2-9e05-cb904c6c0502" />

Key drivers of subscription are consistent across models, highlighting the importance of customer profile, campaign history and macroeconomic context.

---

## 🛠️ Skills Demonstrated

- Data preprocessing & pipelines (scikit-learn)
- Handling imbalanced datasets
- Model comparison & evaluation
- Business-oriented interpretation of results

---

## 📚 Reference

Moro, S., Cortez, P., & Rita, P. (2014).  
*A Data-Driven Approach to Predict the Success of Bank Telemarketing.*
