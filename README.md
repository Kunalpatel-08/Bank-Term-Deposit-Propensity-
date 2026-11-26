# Term Deposit Propensity Model — Pre-Call Scoring Dashboard

> A leakage-free machine-learning model that ranks bank customers by their likelihood to subscribe to a term deposit **before** the call is made, served through an interactive Streamlit dashboard.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kunalpatel-08/Banking_Term_deposit_classification/blob/main/Propensity_bank_term_deposit_NO_DURATION.ipynb)

**Live dashboard:** `https://<your-app-name>.streamlit.app` *(replace with your Streamlit Cloud URL after deploying)*

---

## 1. Business problem

A bank's call-centre has limited agent time and cannot phone every customer. The goal is to **score each customer before the call** so agents contact the most promising people first, which reduces wasted calls and raises campaign ROI.

## 2. The key design decision: `duration` is excluded

The UCI dataset includes `duration` (length of the last call). It is only known **after** the call ends, so a model that uses it cannot decide *whom to call*.

| | With `duration` (earlier run) | **Without `duration` (this project)** |
|---|---|---|
| AUC-ROC | 0.9166 | **0.7880** |
| Gini | 0.8332 | **0.5760** |
| KS | 0.7038 | **0.4463** |
| Deployable before the call? | No (data leakage) | **Yes** |

The lower numbers are the honest, deployable performance.

## 3. Results (held-out test set, 6,782 customers, 11.7% subscribers)

| Metric | Value |
|---|---|
| AUC-ROC | 0.7880 |
| Gini | 0.5760 |
| KS statistic | 0.4463 (max at decile 2) |
| Top-3 decile capture | 68.5% of all subscribers found in the top 30% of customers |
| Top-decile lift | 4.07× the average rate |
| Brier score (calibrated) | 0.0840 |
| 5-fold CV AUC (SMOTE inside each fold) | 0.7806 ± 0.0082 |

**Two decision thresholds, both chosen on the validation set (the test set is only used to report):**

| Threshold | Rule | Precision | Recall | F1 | False positives | False negatives | Cost* |
|---|---|---|---|---|---|---|---|
| 0.10 | Cost-optimal | 0.2680 | 0.6797 | 0.3845 | 1,472 | 254 | 4,012 |
| 0.21 | F1-optimal | 0.4318 | 0.4994 | 0.4632 | 521 | 397 | 4,491 |

\*Illustrative cost model: a missed subscriber costs 10 units, a wasted call costs 1 unit. Replace these with the bank's real economics in the dashboard.

**Stability:** PSI between validation and test scores is 0.0014 (random split, sanity check only). On a chronological out-of-time test (train on the first 70% of rows, test on the last 30%), PSI is 0.2496 (monitor) and AUC is 0.6752.

## 4. Approach

1. **Split first** (70 / 15 / 15, stratified): 31,665 train, 6,764 validation, 6,782 test. Weight of Evidence, Information Value and balance cut-points use **train rows only**.
2. **Feature engineering** (`features.py`): 15 raw columns become 32 engineered features and 67 encoded columns. A train-only correlation filter (|corr| > 0.90) removes 12, leaving **55 model features**. The same file is used for training and for the app, so there is no train/serve skew.
3. **Model comparison on validation AUC** (SMOTE-trained): Logistic Regression 0.7149, Random Forest 0.7786, LightGBM 0.7861, **XGBoost 0.7908**.
4. **Imbalance strategy test** (XGBoost, validation AUC): SMOTE 0.7908, class weight 0.7976, **plain 0.8001**. The plain model won, so SMOTE is not used in the final model.
5. **Probability calibration:** Platt scaling fitted on validation, `p = sigmoid(0.977 × logit(raw) − 0.050)`. This keeps the ranking unchanged but makes thresholds and cost calculations meaningful.
6. **Explainability:** SHAP summary and per-customer feature contributions. The strongest global drivers are `contact_unknown`, `prev_success_flag`, `financial_stress`, `marital_married` and `age`.
7. **Export:** one `propensity_model.joblib` bundle with round-trip checks.

Top predictors by Information Value (train only): `poutcome` 0.4850, `month` 0.4474, `contact` 0.2882, `housing` 0.1901, `job` 0.1597.

## 5. Dashboard

| Tab | What it does |
|---|---|
| **Score a customer** | Enter a customer's pre-call details and get the subscription probability, a call/skip decision at the chosen threshold and the top 8 drivers of that score |
| **Model performance** | All test metrics, ROC curve, lift by decile, decile table and an interactive threshold/cost explorer |
| **Batch scoring** | Upload a CSV, get a ranked and scored list and download it. Extra columns such as `duration` or `y` are ignored |
| **About** | Plain-language explanation and limits |

## 6. Repository structure

```
├── app.py                    # Streamlit dashboard
├── features.py               # feature engineering shared by notebook and app
├── propensity_model.joblib   # trained model + calibration + thresholds + metrics
├── test_scores.csv           # held-out scores used by the performance charts
├── xgb_model.json            # XGBoost native backup (used if the joblib cannot be opened)
├── model_meta.json           # metadata for the backup
├── requirements.txt          # pinned library versions
├── README.md
└── Propensity_bank_term_deposit_NO_DURATION_v3_joblib.ipynb   # (optional) full training notebook
```

## 7. Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 8. Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
4. Select the repository, branch `main`, main file path `app.py`.
5. Under **Advanced settings**, choose the Python version used to train the model.
6. Click **Deploy**.

## 9. Retrain the model

1. Open the notebook in Colab and run all cells (upload `bank-full.csv` when asked).
2. Section 12 runs three checks (train/serve parity, calibration, round-trip) and creates `app_artifacts.zip`.
3. Replace `propensity_model.joblib`, `test_scores.csv`, `xgb_model.json`, `model_meta.json` and `requirements.txt` in the repo **together**. A joblib file must be opened with the same `xgboost` and `scikit-learn` versions that created it.

## 10. Limitations

- **Old, single-bank data:** a Portuguese bank, 2008–2010. Month and channel effects are campaign-specific and may not transfer to another bank.
- **Time shift:** the out-of-time AUC (0.6752) is well below the random-split test AUC (0.7880), and the subscription rate rises from 5.8% to 25.4% between the early and late rows. Treat the random-split figure as an upper bound for future performance.
- **`campaign` is approximately pre-call:** in the UCI file it counts contacts including the last one. In production it should be redefined as "contacts before this call".
- **`poutcome` is a concentrated signal:** its high IV comes from a small group of previously converted customers.
- **Cost figures are illustrative.** Set real values before using the cost-optimal threshold.

## 11. Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · imbalanced-learn · SHAP · Streamlit · Altair · joblib

## 12. Data source and citation

UCI Machine Learning Repository, *Bank Marketing* dataset (`bank-full.csv`, 45,211 rows).

S. Moro, P. Cortez and P. Rita. *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 2014.

## 13. Author

**Kunal** — aspiring Data Analyst / Data Scientist · [GitHub](https://github.com/Kunalpatel-08)
