# 💳 Credit Card Fraud Detection — Predictive Modeling

End-to-end fraud detection pipeline built on the classic European credit card transactions dataset. Goes beyond the usual Kaggle treatment — proper class-imbalance handling, multiple model comparisons, and an evaluation framework that's actually meant for fraud operations, not just a leaderboard score.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Why this project

Fraud detection is one of those problems where the obvious approach (accuracy) is also the wrong approach. With only 0.17% of transactions being fraudulent, a model that does literally nothing — just predicts "not fraud" every single time — scores 99.83% accuracy. That number sounds great in a meeting and means nothing on the ground.

This project was built to actually deal with that problem properly: correct evaluation metrics, correct imbalance handling, and findings written the way a fraud or risk analyst would actually use them — not just a notebook full of charts with no business takeaway.

---

## 📊 Dataset

| | |
|---|---|
| **Source** | [Kaggle — Credit Card Fraud Detection (ULB)](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| **Transactions** | 284,807 |
| **Fraud cases** | 492 (0.172%) |
| **Features** | 28 PCA-anonymized components (V1–V28) + `Time` + `Amount` |
| **Target** | `Class` (1 = fraud, 0 = legitimate) |
| **Time span** | 48 hours, September 2013, European cardholders |

> The original features were anonymized via PCA by the data providers for cardholder privacy — so `V1` to `V28` don't map back to anything we can name directly. We work with what they tell us statistically.

**Note on the dataset file:** `creditcard.csv` is not included in this repo due to GitHub's file size limits (the dataset is ~150MB). Download it directly from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the project root before running the notebook.

---

## 🗂️ Repository structure

```
credit-card-fraud-detection/
│
├── credit_card_fraud_detection_portfolio_2026.ipynb   # Main notebook
├── creditcard.csv                                      # Dataset (download separately — see above)
├── requirements.txt                                    # Python dependencies
└── README.md                                            # You're here
```

---

## 🔍 What's inside the notebook

| Section | What it covers |
|---|---|
| **1. Data Ingestion & Quality Audit** | Null checks, descriptive stats, schema sanity |
| **2. Class Imbalance Analysis** | Quantifying the 578:1 ratio and what it means for modeling choices |
| **3. Exploratory Data Analysis** | Temporal patterns (circadian rhythm of legit transactions vs. fraud's flat distribution), amount profiling, correlation structure, and a full KDE sweep across all 30 features to spot which ones actually separate fraud from non-fraud |
| **4. Feature Engineering & Splitting** | RobustScaler for outlier-heavy `Amount`, log transforms, and a proper **stratified 60/20/20 train-validation-test split** so the rare fraud cases don't get unevenly distributed |
| **5. Modeling** | Logistic Regression (baseline) → Random Forest → XGBoost → LightGBM, including a 5-fold Stratified CV pass for LightGBM with out-of-fold predictions |
| **6. Evaluation** | PR-AUC as the primary metric (not accuracy, not even ROC-AUC alone), plus Gini coefficient, threshold-tuning, and a side-by-side model comparison dashboard |
| **7. Feature Importance** | Cross-model consensus ranking — features that show up as important across Random Forest, XGBoost, *and* LightGBM are far more trustworthy than a single model's opinion |
| **8. Business Insights** | Tiered alert thresholds, monitoring recommendations, retraining cadence — written for someone who'd actually have to operationalize this |

---

## ⚙️ Why these specific choices

A few decisions in this notebook are deliberate and worth calling out, because they're exactly the kind of thing that gets asked about in interviews:

- **PR-AUC over accuracy/ROC-AUC as the primary metric** — At this level of imbalance, accuracy is meaningless and ROC-AUC alone can look misleadingly good. PR-AUC focuses specifically on how well the model performs on the minority (fraud) class, which is the class that actually matters here.
- **No SMOTE / oversampling** — Synthetic samples can leak information across folds and don't reflect how real fraud actually looks. Class-weighting (`scale_pos_weight`, `class_weight='balanced'`, `is_unbalance`) was used instead — it corrects the loss function without inventing fake data.
- **Stratified splits everywhere** — With only 492 fraud cases total, an unstratified split risks ending up with a validation or test set that has almost no fraud examples in it, making your metrics unreliable. Every split and every fold is stratified on `Class`.
- **AdaBoost and CatBoost were tried and dropped** — They didn't add anything meaningful over what Random Forest/XGBoost/LightGBM already covered, and including every possible model just for the sake of it doesn't make a project stronger — it makes it longer.

---

## 📈 Results snapshot

| Model | ROC-AUC | PR-AUC | Gini |
|---|---|---|---|
| Logistic Regression (baseline) | ~0.97 | ~0.72 | ~0.94 |
| Random Forest | ~0.96 | ~0.81 | ~0.93 |
| XGBoost | ~0.97 | ~0.85 | ~0.95 |
| LightGBM (5-fold CV) | ~0.98 | ~0.87 | ~0.96 |

*(Exact numbers will vary slightly run-to-run due to train/test split randomness — see the notebook for the actual run's output.)*

The gradient-boosted models (XGBoost, LightGBM) come out ahead, which is expected given the non-linear feature interactions in PCA-transformed data. The logistic regression baseline is there on purpose — it's important to show that the added complexity of ensemble models is actually earning its keep, not just there to look impressive.

---

## 🛠️ How to run it

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Download the dataset**

Get `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and drop it in the project root.

**3. Set up the environment**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**4. Launch the notebook**
```bash
jupyter notebook credit_card_fraud_detection_portfolio_2026.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
lightgbm
seaborn
matplotlib
plotly
```

(See `requirements.txt` for pinned versions.)

---

## 🧭 What I'd build next

This notebook is solid as a portfolio piece, but a few things would genuinely strengthen it for a production setting:

- **SHAP values** for per-transaction explainability — needed for any real compliance/regulatory review
- **Temporal train-test split** instead of random — since `Time` is sequential, a strict day-1-train/day-2-test split would better simulate real deployment conditions
- **Cost-sensitive threshold tuning** using actual chargeback cost estimates, rather than a generic F1-optimal threshold

---

## 🙋 About me

Built by **Arpit Singh** — M.Sc. Operations Research, University of Delhi. Currently exploring roles in data analytics, data science, and risk/fraud analytics.

- 🔗 [LinkedIn](https://www.linkedin.com/in/arpit-singh-or)
- 💻 [Portfolio](https://arpit-singh-or.github.io)

Feel free to open an issue or reach out if you spot something that could be improved — always happy to discuss the approach or get feedback.

---

## 📄 License

This project is open-sourced under the MIT License. The dataset itself is provided by the [Machine Learning Group at ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud) under their own terms — check the Kaggle page for details.
