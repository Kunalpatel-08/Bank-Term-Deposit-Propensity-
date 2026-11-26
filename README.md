# Term Deposit Propensity — Streamlit dashboard

Pre-call propensity scoring on the UCI Bank Marketing data (`duration` excluded to avoid leakage).

## Repo layout
```
app.py                    # Streamlit dashboard
features.py               # feature engineering shared with the notebook
propensity_model.joblib   # from notebook Section 12 (NOT included here — generate it)
test_scores.csv           # from notebook Section 12
xgb_model.json            # from notebook Section 12 (fallback)
model_meta.json           # from notebook Section 12 (fallback)
requirements.txt          # from notebook Section 12 (pinned versions)
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push this folder to GitHub, then on https://share.streamlit.io choose **Create app**, pick the repo,
branch `main`, main file `app.py`, and deploy.
