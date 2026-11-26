"""features.py -- feature engineering shared by the notebook AND the Streamlit app.

The notebook writes this file (%%writefile), so training and serving use identical code.
All categorical levels are fixed here, so scoring a SINGLE customer gives the same
columns as training (plain get_dummies on one row would silently drop the dummy).
"""
import numpy as np
import pandas as pd

RAW_COLS = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
            'contact', 'day', 'month', 'campaign', 'pdays', 'previous', 'poutcome']

JOBS      = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
             'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
MARITAL   = ['divorced', 'married', 'single']
EDUCATION = ['unknown', 'primary', 'secondary', 'tertiary']
CONTACT   = ['cellular', 'telephone', 'unknown']
POUTCOME  = ['failure', 'other', 'success', 'unknown']
MONTHS    = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

AGE_LABELS = ['<25', '25-35', '35-45', '45-55', '55-65', '65+']
BAL_LABELS = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
QUARTERS   = ['Q1', 'Q2', 'Q3', 'Q4']

MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTHS)}
EDU_MAP   = {k: i for i, k in enumerate(EDUCATION)}

_CATS = {'job': JOBS, 'marital': MARITAL, 'contact': CONTACT, 'poutcome': POUTCOME,
         'age_group': AGE_LABELS, 'balance_bucket': BAL_LABELS, 'quarter': QUARTERS}


def feature_engineering_nodur(df, bal_cuts):
    """df: raw bank-marketing columns (no duration). bal_cuts: 4 balance quantile cut-points
    (20/40/60/80th percentile) learned on the TRAIN rows only."""
    d = df.copy()
    edges = [-np.inf] + [float(c) for c in bal_cuts] + [np.inf]

    # Age / balance
    d['age_group'] = pd.cut(d['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=AGE_LABELS)
    d['balance_log'] = np.log1p(np.clip(d['balance'], 0, None))
    d['has_negative_bal'] = (d['balance'] < 0).astype(int)
    d['balance_bucket'] = pd.cut(d['balance'], bins=edges, labels=BAL_LABELS, include_lowest=True)

    # Campaign fatigue (pre-call observable)
    d['campaign_log'] = np.log1p(d['campaign'])
    d['is_first_contact'] = (d['campaign'] == 1).astype(int)
    d['many_contacts'] = (d['campaign'] > 5).astype(int)

    # Prior-contact recency & engagement
    d['was_contacted_prev'] = (d['pdays'] != -1).astype(int)
    d['days_since_contact'] = d['pdays'].replace(-1, 999)
    d['days_since_contact_log'] = np.log1p(d['days_since_contact'])
    d['has_previous'] = (d['previous'] > 0).astype(int)
    d['previous_log'] = np.log1p(d['previous'])
    d['recency_score'] = np.where(d['pdays'] == -1, 0,
                         np.where(d['pdays'] <= 30, 3,
                         np.where(d['pdays'] <= 90, 2,
                         np.where(d['pdays'] <= 180, 1, 0))))
    d['engagement_index'] = d['previous'] * (d['recency_score'] + 1)

    # Seasonality
    d['month_num'] = d['month'].map(MONTH_MAP)
    d['month_sin'] = np.sin(2 * np.pi * d['month_num'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month_num'] / 12)
    d['is_end_of_year'] = d['month'].isin(['oct', 'nov', 'dec']).astype(int)
    d['quarter'] = d['month_num'].apply(
        lambda m: 'Q1' if m <= 3 else 'Q2' if m <= 6 else 'Q3' if m <= 9 else 'Q4')
    d['is_end_of_month'] = (d['day'] >= 25).astype(int)
    d['is_start_of_month'] = (d['day'] <= 5).astype(int)

    # Financial profile
    d['financial_stress'] = ((d['default'] == 'yes').astype(int) +
                             (d['loan'] == 'yes').astype(int) +
                             (d['housing'] == 'yes').astype(int))
    d['edu_score'] = d['education'].map(EDU_MAP).fillna(0)
    d['wealth_readiness'] = d['balance_log'] * d['edu_score'] / (d['financial_stress'] + 1)

    # Interactions / flags
    d['age_x_balance'] = d['age'] * d['balance_log']
    d['retired_flag'] = (d['job'] == 'retired').astype(int)
    d['student_flag'] = (d['job'] == 'student').astype(int)
    d['management_flag'] = (d['job'] == 'management').astype(int)
    d['prev_success_flag'] = (d['poutcome'] == 'success').astype(int)
    d['prev_failure_flag'] = (d['poutcome'] == 'failure').astype(int)
    d['age_stress'] = d['age'] * d['financial_stress']
    d['cellular_prev_success'] = (d['contact'] == 'cellular').astype(int) * d['prev_success_flag']
    return d


def encode_features_nodur(df_fe):
    d = df_fe.copy()
    for col in ['default', 'housing', 'loan']:
        d[col] = d[col].map({'yes': 1, 'no': 0})
    d['education'] = d['education'].map(EDU_MAP)

    # fixed category lists -> identical dummy columns for 1 row or 45,000 rows
    for col, cats in _CATS.items():
        d[col] = pd.Categorical(d[col].astype(str), categories=cats)
    d = pd.get_dummies(d, columns=list(_CATS), drop_first=True)

    d = d.drop(columns=[c for c in ['y', 'day', 'month', 'month_num'] if c in d.columns])
    bool_cols = d.select_dtypes('bool').columns
    d[bool_cols] = d[bool_cols].astype(int)
    return d


def build_model_matrix(raw, bal_cuts, feature_cols):
    """Raw customer rows -> model matrix with exactly the training columns (used by the app)."""
    fe = feature_engineering_nodur(raw[RAW_COLS], bal_cuts)
    enc = encode_features_nodur(fe)
    return enc.reindex(columns=feature_cols, fill_value=0).astype(float)
