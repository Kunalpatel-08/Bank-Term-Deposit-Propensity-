"""Term Deposit Propensity — interactive Streamlit dashboard.

Loads `propensity_model.joblib` (written by Section 12 of the notebook) and scores customers
using only PRE-CALL information (the leaky `duration` column is never used).

Files this app expects next to it:
    propensity_model.joblib   model + calibration + thresholds + metrics
    features.py               the SAME feature-engineering code used in training
    test_scores.csv           held-out test scores (for the performance charts)
    xgb_model.json            optional fallback if the joblib cannot be opened
    model_meta.json           optional, only used together with the JSON fallback
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from features import (RAW_COLS, JOBS, MARITAL, EDUCATION, CONTACT, POUTCOME, MONTHS,
                      build_model_matrix)

HERE = Path(__file__).parent
GREEN, RED, BLUE = "#27AE60", "#C0392B", "#2980B9"

st.set_page_config(page_title="Term Deposit Propensity", page_icon="🏦", layout="wide")


# ───────────────────────────── loading ─────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_bundle():
    """Return (bundle, note). Tries the joblib first, then the XGBoost JSON backup."""
    try:
        return joblib.load(HERE / "propensity_model.joblib"), None
    except Exception as err:  # e.g. library-version mismatch
        try:
            meta = json.load(open(HERE / "model_meta.json"))
            model = xgb.XGBClassifier()
            model.load_model(str(HERE / "xgb_model.json"))
            return {"model": model, **meta}, (
                f"The joblib file could not be opened ({type(err).__name__}); "
                "using the XGBoost JSON backup instead. Pin the library versions in requirements.txt.")
        except Exception as err2:
            st.error(f"Could not load the model: {err}\nBackup also failed: {err2}")
            st.stop()


@st.cache_data
def load_test_scores():
    df = pd.read_csv(HERE / "test_scores.csv")
    return df["y_true"].to_numpy(dtype=int), df["proba"].to_numpy(dtype=float)


bundle, load_note = load_bundle()
MODEL, METRICS = bundle["model"], bundle["metrics"]
y_test, p_test = load_test_scores()


# ───────────────────────────── scoring ─────────────────────────────
def calibrate(p, a, b):
    """Platt scaling: sigmoid(a * logit(p) + b) — identical to the notebook."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(a * np.log(p / (1 - p)) + b)))


def score(raw_df):
    """Raw customer rows -> (calibrated probability array, model matrix)."""
    X = build_model_matrix(raw_df, bundle["bal_cuts"], bundle["features"])
    raw = MODEL.predict_proba(X)[:, 1]
    return calibrate(raw, bundle["platt_a"], bundle["platt_b"]), X


def drivers(X_row):
    """Per-feature push on the raw score (log-odds) for ONE customer, via XGBoost's built-in contributions."""
    contrib = MODEL.get_booster().predict(xgb.DMatrix(X_row), pred_contribs=True)[0][:-1]
    out = pd.DataFrame({"feature": X_row.columns, "push": contrib, "value": X_row.iloc[0].to_numpy()})
    out = out[out["push"].abs() > 1e-9]
    return out.reindex(out["push"].abs().sort_values(ascending=False).index).head(8)


# ───────────────────────── test-set analytics ─────────────────────────
@st.cache_data
def roc_points(y, p, max_pts=400):
    order = np.argsort(-p)
    ys = y[order]
    tpr = np.cumsum(ys) / ys.sum()
    fpr = np.cumsum(1 - ys) / (len(ys) - ys.sum())
    keep = np.unique(np.linspace(0, len(ys) - 1, max_pts).astype(int))
    return pd.DataFrame({"False positive rate": np.r_[0, fpr[keep]], "True positive rate": np.r_[0, tpr[keep]]})


@st.cache_data
def decile_table(y, p):
    order = np.argsort(-p)
    ys, base = y[order], y.mean()
    rows, cum = [], 0
    for i, chunk in enumerate(np.array_split(ys, 10), start=1):
        cum += chunk.sum()
        rows.append({"Decile": i, "Customers": len(chunk), "Subscribers": int(chunk.sum()),
                     "Conversion rate": chunk.mean(), "Lift": chunk.mean() / base,
                     "Cumulative capture": cum / ys.sum()})
    return pd.DataFrame(rows)


def confusion_at(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1, called=tp + fp)


# ───────────────────────────── sidebar ─────────────────────────────
with st.sidebar:
    st.header("⚙️ Decision rule")
    mode = st.radio("Call a customer when the score is above…",
                    [f"Cost-optimal ({bundle['threshold_cost']:.2f})",
                     f"F1-optimal ({bundle['threshold_f1']:.2f})", "Custom"], index=0)
    if mode.startswith("Cost"):
        THRESHOLD = bundle["threshold_cost"]
    elif mode.startswith("F1"):
        THRESHOLD = bundle["threshold_f1"]
    else:
        THRESHOLD = st.slider("Custom threshold", 0.02, 0.90, 0.20, 0.01)
    st.metric("Active threshold", f"{THRESHOLD:.2f}")
    st.caption(f"Model: **{bundle['model_name']}** (Platt-calibrated)  \n"
               f"Trained on: {bundle['trained_on']}  \n"
               f"Features used: {METRICS['n_features']}")
    if load_note:
        st.warning(load_note)

st.title("🏦 Term Deposit Propensity — pre-call scoring")
st.caption("Who should an agent call first? Uses only information known **before** the call "
           "(the post-call `duration` field is deliberately excluded to avoid data leakage).")

tab_score, tab_perf, tab_batch, tab_about = st.tabs(
    ["🎯 Score a customer", "📊 Model performance", "📁 Batch scoring", "ℹ️ About"])

# ───────────────────────── tab 1: single customer ─────────────────────────
with tab_score:
    st.subheader("Customer details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 100, 40)
        job = st.selectbox("Job", JOBS, index=JOBS.index("management"))
        marital = st.selectbox("Marital status", MARITAL, index=1)
        education = st.selectbox("Education", EDUCATION, index=EDUCATION.index("secondary"))
        default = st.selectbox("Credit in default?", ["no", "yes"])
    with c2:
        balance = st.number_input("Average yearly balance (€)", -10000, 150000, 1000, step=100)
        housing = st.selectbox("Housing loan?", ["yes", "no"])
        loan = st.selectbox("Personal loan?", ["no", "yes"])
        contact = st.selectbox("Contact channel", CONTACT, index=0)
        month = st.selectbox("Month of contact", MONTHS, index=MONTHS.index("may"))
    with c3:
        day = st.number_input("Day of month", 1, 31, 15)
        campaign = st.number_input("Contacts in this campaign (incl. this one)", 1, 60, 1)
        pdays = st.number_input("Days since previous campaign contact (-1 = never)", -1, 900, -1)
        previous = st.number_input("Contacts before this campaign", 0, 60, 0)
        poutcome = st.selectbox("Outcome of previous campaign", POUTCOME, index=POUTCOME.index("unknown"))

    if previous == 0 and (pdays != -1 or poutcome != "unknown"):
        st.warning("Previous contacts = 0, but days-since-contact / previous outcome are filled in. "
                   "For a never-contacted customer use pdays = -1 and outcome = unknown.")

    row = pd.DataFrame([dict(age=age, job=job, marital=marital, education=education, default=default,
                             balance=balance, housing=housing, loan=loan, contact=contact, day=day,
                             month=month, campaign=campaign, pdays=pdays, previous=previous,
                             poutcome=poutcome)])[RAW_COLS]
    prob, X_row = score(row)
    prob = float(prob[0])
    base_rate = METRICS["event_rate_test"]
    better_than = float((p_test < prob).mean())

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Subscription probability", f"{prob * 100:.2f}%")
    m2.metric("vs. average customer", f"{prob / base_rate:.2f}×", help=f"Average subscription rate: {base_rate * 100:.2f}%")
    m3.metric("Ranks above", f"{better_than * 100:.1f}% of test customers")
    if prob >= THRESHOLD:
        st.success(f"✅ **Call this customer** — score {prob:.4f} ≥ threshold {THRESHOLD:.2f}.")
    else:
        st.info(f"⏸️ **Low priority** — score {prob:.4f} < threshold {THRESHOLD:.2f}.")

    st.markdown("**What is pushing this score up or down?**")
    d = drivers(X_row)
    if d.empty:
        st.caption("No feature contributions to show.")
    else:
        d["direction"] = np.where(d["push"] >= 0, "raises score", "lowers score")
        chart = (alt.Chart(d).mark_bar().encode(
            x=alt.X("push:Q", title="Push on raw model score (log-odds)"),
            y=alt.Y("feature:N", sort=alt.EncodingSortField("push", op="sum", order="descending"), title=None),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises score", "lowers score"],
                                                            range=[GREEN, RED]), legend=alt.Legend(title=None)),
            tooltip=["feature", alt.Tooltip("value:Q", format=".3f"), alt.Tooltip("push:Q", format="+.3f")],
        ).properties(height=280))
        st.altair_chart(chart, width="stretch")
        st.caption("Bars show the 8 strongest drivers for this customer (XGBoost feature contributions). "
                   "Feature names are the engineered model inputs, e.g. `poutcome_success` = previous campaign succeeded.")

# ───────────────────────── tab 2: performance ─────────────────────────
with tab_perf:
    st.subheader("Held-out test performance")
    st.caption(f"{METRICS['n_test']:,} test customers · natural event rate {METRICS['event_rate_test'] * 100:.2f}% · "
               "never used for training or threshold tuning")
    a = st.columns(5)
    a[0].metric("AUC-ROC", f"{METRICS['auc']:.4f}")
    a[1].metric("Gini", f"{METRICS['gini']:.4f}")
    a[2].metric("KS", f"{METRICS['ks']:.4f}")
    a[3].metric("PR-AUC", f"{METRICS['pr_auc']:.4f}")
    a[4].metric("Brier score", f"{METRICS['brier']:.4f}")
    b = st.columns(5)
    b[0].metric(f"F1 @ {bundle['threshold_f1']:.2f}", f"{METRICS['f1_at_f1_thr']:.4f}")
    b[1].metric("Precision", f"{METRICS['precision_at_f1_thr']:.4f}")
    b[2].metric("Recall", f"{METRICS['recall_at_f1_thr']:.4f}")
    b[3].metric("Top-3 decile capture", f"{METRICS['top3_decile_capture_pct']:.1f}%")
    b[4].metric("Top-decile lift", f"{METRICS['top_decile_lift']:.2f}×")
    st.caption(f"Stability: out-of-time AUC **{METRICS['auc_oot']:.4f}** · out-of-time PSI **{METRICS['psi_oot']:.4f}** "
               f"· random-split PSI {METRICS['psi_random']:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**ROC curve**")
        roc = roc_points(y_test, p_test)
        line = alt.Chart(roc).mark_line(color=BLUE).encode(x="False positive rate:Q", y="True positive rate:Q")
        diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        st.altair_chart((line + diag).properties(height=300), width="stretch")
    dec = decile_table(y_test, p_test)
    with right:
        st.markdown("**Lift by decile** (1 = highest scores)")
        bars = alt.Chart(dec).mark_bar(color=BLUE).encode(
            x=alt.X("Decile:O", title="Decile"), y=alt.Y("Lift:Q"),
            tooltip=["Decile", alt.Tooltip("Lift:Q", format=".2f"), alt.Tooltip("Conversion rate:Q", format=".2%")])
        base = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color=RED, strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart((bars + base).properties(height=300), width="stretch")

    st.markdown("**Decile table**")
    st.dataframe(dec.style.format({"Conversion rate": "{:.2%}", "Lift": "{:.2f}", "Cumulative capture": "{:.1%}"}),
                 hide_index=True, width="stretch")

    st.divider()
    st.subheader("Threshold & cost explorer")
    k1, k2, k3 = st.columns(3)
    cost_fn = k1.number_input("Cost of missing a subscriber (FN)", 1.0, 1000.0, float(bundle["cost_fn"]), step=1.0)
    cost_fp = k2.number_input("Cost of a wasted call (FP)", 0.1, 1000.0, float(bundle["cost_fp"]), step=0.5)
    t_try = k3.slider("Threshold to inspect", 0.02, 0.90, float(round(THRESHOLD, 2)), 0.01)

    grid = np.round(np.arange(0.02, 0.91, 0.01), 2)
    sweep = pd.DataFrame([{"Threshold": t, **confusion_at(y_test, p_test, t)} for t in grid])
    sweep["Total cost"] = sweep["fn"] * cost_fn + sweep["fp"] * cost_fp
    best = sweep.loc[sweep["Total cost"].idxmin()]
    cur = confusion_at(y_test, p_test, t_try)
    r = st.columns(6)
    r[0].metric("Customers called", f"{cur['called']:,}")
    r[1].metric("Subscribers reached", f"{cur['tp']:,}")
    r[2].metric("Precision", f"{cur['precision']:.4f}")
    r[3].metric("Recall", f"{cur['recall']:.4f}")
    r[4].metric("F1", f"{cur['f1']:.4f}")
    r[5].metric("Total cost", f"{cur['fn'] * cost_fn + cur['fp'] * cost_fp:,.0f}")
    cost_line = alt.Chart(sweep).mark_line(color=RED).encode(x="Threshold:Q", y="Total cost:Q")
    pts = alt.Chart(pd.DataFrame({"Threshold": [float(best["Threshold"])], "Total cost": [float(best["Total cost"])]})
                    ).mark_point(size=120, filled=True, color="black").encode(x="Threshold:Q", y="Total cost:Q")
    now = alt.Chart(pd.DataFrame({"Threshold": [t_try]})).mark_rule(color=BLUE, strokeDash=[4, 4]).encode(x="Threshold:Q")
    st.altair_chart((cost_line + pts + now).properties(height=280), width="stretch")
    st.caption(f"Black dot = lowest total cost on the test set with these costs: threshold **{best['Threshold']:.2f}** "
               f"(theory for calibrated scores: FP ÷ (FP + FN) = {cost_fp / (cost_fp + cost_fn):.3f}).")

# ───────────────────────── tab 3: batch ─────────────────────────
with tab_batch:
    st.subheader("Score a customer list")
    st.write("Upload a CSV with these columns (extra columns such as `duration` or `y` are ignored):")
    st.code(", ".join(RAW_COLS), language="text")
    sample = pd.DataFrame([
        [58, "management", "married", "tertiary", "no", 2143, "yes", "no", "unknown", 5, "may", 1, -1, 0, "unknown"],
        [33, "technician", "single", "secondary", "no", 450, "no", "no", "cellular", 20, "aug", 2, 90, 2, "success"],
        [29, "student", "single", "tertiary", "no", 3100, "no", "no", "cellular", 12, "mar", 1, -1, 0, "unknown"],
    ], columns=RAW_COLS)
    st.download_button("⬇️ Download a 3-row template", sample.to_csv(index=False), "template_customers.csv", "text/csv")

    up = st.file_uploader("Customer CSV", type=["csv"])
    if up is not None:
        try:
            df_in = pd.read_csv(up, sep=None, engine="python")  # auto-detects , or ;
            df_in.columns = [c.strip().strip('"') for c in df_in.columns]
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            st.stop()
        missing_cols = [c for c in RAW_COLS if c not in df_in.columns]
        if missing_cols:
            st.error(f"Missing required column(s): {', '.join(missing_cols)}")
        else:
            work = df_in[RAW_COLS].copy()
            for c in ["age", "balance", "day", "campaign", "pdays", "previous"]:
                work[c] = pd.to_numeric(work[c], errors="coerce")
            levels = {"job": JOBS, "marital": MARITAL, "education": EDUCATION, "default": ["no", "yes"],
                      "housing": ["no", "yes"], "loan": ["no", "yes"], "contact": CONTACT,
                      "month": MONTHS, "poutcome": POUTCOME}
            valid = work.notna().all(axis=1)
            for col, allowed in levels.items():
                valid &= work[col].isin(allowed)
            n_bad = int((~valid).sum())
            if n_bad:
                st.warning(f"{n_bad:,} of {len(work):,} rows have missing values or unknown category labels "
                           "and were skipped (they are shown with an empty score).")
            if valid.sum() == 0:
                st.error("No valid rows to score.")
            else:
                probs, _ = score(work[valid])
                out = df_in.copy()
                out["propensity"] = np.nan
                out.loc[valid, "propensity"] = np.round(probs, 4)
                out["call_decision"] = np.where(out["propensity"].isna(), "invalid row",
                                         np.where(out["propensity"] >= THRESHOLD, "CALL", "skip"))
                out["priority_rank"] = out["propensity"].rank(ascending=False, method="first")
                out = out.sort_values("priority_rank")
                s1, s2, s3 = st.columns(3)
                s1.metric("Rows scored", f"{int(valid.sum()):,}")
                s2.metric(f"To call (≥ {THRESHOLD:.2f})", f"{int((out['call_decision'] == 'CALL').sum()):,}")
                s3.metric("Expected subscribers", f"{float(np.nansum(out['propensity'])):.1f}")
                st.dataframe(out.head(500), hide_index=True, width="stretch")
                if len(out) > 500:
                    st.caption("Showing the top 500 rows; the download contains all rows.")
                st.download_button("⬇️ Download scored list", out.to_csv(index=False),
                                   "scored_customers.csv", "text/csv")

# ───────────────────────── tab 4: about ─────────────────────────
with tab_about:
    st.markdown(f"""
### What this tool does
Gives every customer a **probability of subscribing to a term deposit** *before* anyone picks up the phone,
so call-centre time goes to the people most likely to say yes.

### Why there is no `duration` feature
Call duration is only known **after** the call ends. A model that uses it looks brilliant on paper
(AUC 0.9166 in the earlier, leaky version) but cannot be used to decide *whom to call*. This model uses only pre-call fields.

### How the score is produced
1. Your inputs are turned into the same engineered features used in training (`features.py`).
2. An XGBoost model gives a raw score.
3. Platt calibration (`sigmoid({bundle['platt_a']:.3f} × logit(score) + {bundle['platt_b']:.3f})`) turns it into a
   realistic probability, so thresholds and cost calculations make sense.
4. The customer is flagged **CALL** when the probability is at or above the threshold in the sidebar.

### Honest limits
- Data: UCI Bank Marketing (Portuguese bank, 2008–2010). Patterns such as month effects are campaign-specific.
- Out-of-time AUC is **{METRICS['auc_oot']:.4f}** vs **{METRICS['auc']:.4f}** on the random test split — expect some drop on future data.
- Probabilities are most trustworthy for ranking and for the average rate; treat single-customer values as estimates.
""")
