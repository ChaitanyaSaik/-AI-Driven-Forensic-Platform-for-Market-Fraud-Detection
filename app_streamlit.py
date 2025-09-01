# app_streamlit_final.py
"""
Hackathon-ready Streamlit app implementing:
- Model Loading & Scoring (IsolationForest + XGBoost)
- Basic KPIs & Alerts Table
- Visualization: risk-over-time (Plotly)
- Evaluation: ROC, PR, Confusion Matrix (if labels exist)
- Explainability: SHAP summary/beeswarm (safe, limited sample)
- News Scanner: headline keyword triggers
- National Security framing + demo script downloads
Also contains a Presentation Guide describing which diagrams to show and what to say.
"""

import os
import io
import joblib
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report, average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report

# optional imports
try:
    import xgboost as xgb
    from xgboost import DMatrix, Booster, XGBClassifier
except Exception:
    xgb = None
try:
    import shap
except Exception:
    shap = None

st.set_page_config(page_title="AI Fraud Detection", layout="wide", page_icon="🛡️")

st.title("🛡️ AI-Driven Forensic Platform for Market Fraud Detection")


# ---------------------- Helpers ----------------------
def safe_load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def ensure_datetime_symbol(df):
    # rename timestamp -> datetime if present
    if "timestamp" in df.columns and "datetime" not in df.columns:
        df = df.rename(columns={"timestamp": "datetime"})
    # try parse datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # no datetime: create one (incremental) so visualizations don't crash
        df["datetime"] = pd.date_range(end=datetime.now(), periods=len(df), freq="T")
    # ensure symbol column exists
    if "symbol" not in df.columns:
        df["symbol"] = "N/A"
    return df

def default_feature_list():
    # 10-feature default used in training historically for this project.
    return ["open","high","low","close","volume",
            "price_gap_pct","true_range","rolling_volatility","volume_z","circuit_breaker"]

def feature_engineer(df):
    df = df.copy()
    if "datetime" in df.columns:
        df = df.sort_values(["symbol","datetime"], kind="mergesort")
    for c in ["open","high","low","close","volume"]:
        if c not in df.columns:
            df[c] = np.nan
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["price_gap_pct"] = ((df["close"] - df["prev_close"]) / df["prev_close"]).replace([np.inf, -np.inf], np.nan)
    df["true_range"] = (df["high"] - df["low"]).abs()
    df["ret"] = df.groupby("symbol")["close"].pct_change()
    df["rolling_volatility"] = df.groupby("symbol")["ret"].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
    df["vol_mean"] = df.groupby("symbol")["volume"].rolling(20, min_periods=5).mean().reset_index(level=0, drop=True)
    df["vol_std"] = df.groupby("symbol")["volume"].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)
    df["volume_z"] = (df["volume"] - df["vol_mean"]) / (df["vol_std"] + 1e-9)
    df["circuit_breaker"] = ((df["price_gap_pct"].abs() > 0.05) | (df["rolling_volatility"] > 0.04)).astype(int)
    # cleanup helper cols
    df = df.drop(columns=["prev_close","ret","vol_mean","vol_std"], errors="ignore")
    feat_cols = default_feature_list()
    # if some of default features missing, return intersection
    feat_cols = [c for c in feat_cols if c in df.columns]
    return df, feat_cols

def try_load(job_path):
    if os.path.exists(job_path):
        try:
            return joblib.load(job_path)
        except Exception as e:
            st.warning(f"Failed to load {job_path}: {e}")
    return None

def try_load_xgb_model(path="xgboost_model.json"):
    if xgb is None:
        return None, None
    if os.path.exists(path):
        # try XGBClassifier first
        try:
            clf = XGBClassifier()
            clf.load_model(path)
            return clf, "classifier"
        except Exception:
            pass
        try:
            booster = Booster()
            booster.load_model(path)
            return booster, "booster"
        except Exception as e:
            st.warning(f"Failed to load xgboost model: {e}")
    return None, None

def align_feature_matrix(df, model_feat_list):
    """Return X aligned to model_feat_list; fills missing features with 0.0"""
    X = pd.DataFrame(index=df.index)
    for f in model_feat_list:
        if f in df.columns:
            X[f] = df[f].astype(float)
        else:
            X[f] = 0.0
    return X

def compute_metrics_if_possible(df, label_col="fraud_flag", prob_col="sup_prob", thresh=0.6):
    if label_col not in df.columns or df[label_col].isna().all():
        return None

    y = df[label_col].astype(int).values
    p = df[prob_col].values

    # ROC
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)

    # PR (use average_precision_score instead of auc(rec, prec))
    rec, prec, _ = precision_recall_curve(y, p)
    pr_auc = average_precision_score(y, p)

    # Confusion Matrix
    yhat = (p >= thresh).astype(int)
    cm = confusion_matrix(y, yhat)
    report = classification_report(y, yhat, output_dict=True, zero_division=0)

    return {
        "roc": (fpr, tpr, roc_auc),
        "pr": (rec, prec, pr_auc),
        "cm": cm,
        "report": report,
    }

def news_keywords_hits(df, col_names=None):
    # pick headline-like columns
    if col_names is None:
        candidates = [c for c in df.columns if c.lower() in ("headline","news_headline","news","title","news_title","headline_text")]
    else:
        candidates = [c for c in df.columns if c in col_names]
    if not candidates:
        return pd.DataFrame()
    # keywords
    keywords = ["pledge","default","raid","sebi","regulatory","scam","fraud","whistleblower","insider","short seller","audit","restatement","pledged shares"]
    alerts = []
    for cn in candidates:
        for i,row in df[[cn]].dropna().iterrows():
            text = str(row[cn]).lower()
            hits = [k for k in keywords if k in text]
            if hits:
                alerts.append({"index": int(i), "headline_col": cn, "headline": row[cn], "keywords": ", ".join(hits)})
    return pd.DataFrame(alerts)

# ---------------------- UI: Sidebar ----------------------
st.sidebar.title("Controls & Data")
uploaded = st.sidebar.file_uploader("Upload CSV (alerts.csv)", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo dataset if upload missing", value=True)
sup_w = st.sidebar.slider("Supervised weight (XGBoost)", 0.0, 1.0, 0.6, 0.05)
unsup_w = 1.0 - sup_w
thresh = st.sidebar.slider("Risk threshold", 0.01, 0.95, 0.6, 0.01)
topn = st.sidebar.slider("Top N alerts to show", 5, 50, 20, 5)
st.sidebar.markdown("---")
st.sidebar.markdown("**Presentation guide** (open at end): use the bottom panel to follow demo script")

# ---------------------- Load Data ----------------------
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.sidebar.success(f"Uploaded {uploaded.name} ({df_raw.shape[0]} rows)")
else:
    if os.path.exists("alerts.csv"):
        df_raw = pd.read_csv("alerts.csv")
        st.sidebar.info("Loaded alerts.csv from local folder")
    elif use_demo:
        # create tiny demo
        now = pd.Timestamp.now()
        rows = []
        for i in range(120):
            t = now - pd.Timedelta(minutes=i*5)
            o = 100 + np.random.randn()
            c = o * (1 + np.random.randn()*0.001)
            h = max(o,c) + abs(np.random.randn()*0.5)
            l = min(o,c) - abs(np.random.randn()*0.5)
            v = int(abs(1e4 + np.random.randn()*2e3))
            rows.append([t, "DEMO", o, h, l, c, v, 0, np.nan])
        df_raw = pd.DataFrame(rows, columns=["datetime","symbol","open","high","low","close","volume","fraud_flag","headline"])
        st.sidebar.info("Using built-in demo dataset")
    else:
        st.error("No data provided. Upload alerts.csv or enable demo dataset.")
        st.stop()

# normalize columns
df_raw = ensure_datetime_symbol(df_raw)

# drop exact duplicate columns if any
df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()

# feature engineering
df_fe, feat_gen = feature_engineer(df_raw)

# ---------------------- Load models and feature list ----------------------
scaler = try_load("feature_scaler.joblib")  # StandardScaler fitted earlier
iforest = try_load("isolation_forest_model.joblib")  # trained isolation forest
xgb_model, xgb_kind = try_load_xgb_model("xgboost_model.json")

# feature list detection: features.joblib -> scaler.feature_names_in_ -> iforest.feature_names_in_ -> default features -> numeric columns
features_job = try_load("features.joblib")
if features_job is not None:
    model_feat_list = list(features_job)
elif scaler is not None and hasattr(scaler, "feature_names_in_"):
    model_feat_list = list(scaler.feature_names_in_)
elif iforest is not None and hasattr(iforest, "feature_names_in_"):
    model_feat_list = list(iforest.feature_names_in_)
else:
    # prefer the engineered defaults if available, else numeric columns excluding labels
    default = default_feature_list()
    model_feat_list = [c for c in default if c in df_fe.columns]
    if not model_feat_list:
        model_feat_list = [c for c in df_fe.select_dtypes(include=[np.number]).columns if c not in ("fraud_flag",)]

st.sidebar.write(f"Model features used: {len(model_feat_list)} features")

# Align to model features for display and scoring
X_for_model = align_feature_matrix(df_fe, model_feat_list)

# ---------------------- Scoring (Model Loading & Scoring) ----------------------
def score_pipeline(X_aligned, scaler, iforest, xgb_model, xgb_kind, if_weight, sup_weight):
    # Scale
    if scaler is not None:
        try:
            Xs = scaler.transform(X_aligned)
        except Exception:
            Xs = scaler.fit_transform(X_aligned)
    else:
        # fit a local StandardScaler
        scaler_local = StandardScaler().fit(X_aligned)
        Xs = scaler_local.transform(X_aligned)

    # IsolationForest scoring (unsupervised)
    if iforest is not None:
        try:
            isos = -iforest.score_samples(Xs)
        except Exception:
            # try fit if needed (safe for demo only)
            try:
                iforest.fit(Xs)
                isos = -iforest.score_samples(Xs)
                st.warning("IsolationForest was not fitted for these features; a temporary refit was performed (for demo only).")
            except Exception as e:
                st.error(f"IsolationForest scoring failed: {e}")
                isos = np.zeros(Xs.shape[0])
    else:
        isos = np.zeros(Xs.shape[0])

    # normalize
    isos = (isos - np.nanmin(isos)) / (np.nanmax(isos) - np.nanmin(isos) + 1e-9)

    # Supervised (XGBoost)
    if xgb_model is not None and xgb is not None and xgb_kind=="classifier":
        try:
            sup = xgb_model.predict_proba(Xs)[:,1]
        except Exception:
            # if mismatch, attempt DMatrix prediction for booster or fallback
            try:
                if xgb_kind=="booster":
                    dmat = DMatrix(Xs)
                    sup = xgb_model.predict(dmat)
                else:
                    sup = isos  # fallback
            except Exception:
                sup = isos
    else:
        sup = isos  # fallback to unsupervised if no supervised model

    sup = (sup - np.nanmin(sup)) / (np.nanmax(sup) - np.nanmin(sup) + 1e-9)

    risk = (if_weight * isos) + (sup_weight * sup)
    return isos, sup, risk, Xs

with st.spinner("Scoring..."):
    iso_scores, xgb_probs, risk_scores, Xs_used = score_pipeline(X_for_model.values, scaler, iforest, xgb_model, xgb_kind, unsup_w, sup_w)
    # attach to df_fe copy
    scored = df_fe.copy().reset_index(drop=True)
    scored["unsup_score"] = iso_scores
    scored["sup_prob"] = xgb_probs
    scored["risk_score"] = risk_scores
    scored["is_alert"] = (scored["risk_score"] >= thresh).astype(int)

# ---------------------- Basic KPIs & Alerts Table ----------------------
st.header("Topline — KPIs & Alerts")
k1,k2,k3 = st.columns(3)
k1.metric("Total observations", f"{len(scored):,}")
k2.metric("High-risk flags", f"{int(scored['is_alert'].sum()):,}")
k3.metric("Avg risk score", f"{scored['risk_score'].mean():.4f}")

# prepare columns for display safely (avoid duplicate column names)
display_df = scored.loc[:, ~scored.columns.duplicated()].copy()

st.subheader("🚨 High-Risk Alerts (Top rows)")
top_alerts = display_df.sort_values("risk_score", ascending=False).head(topn)
cols_to_show = ["datetime","symbol","close","volume","price_gap_pct","rolling_volatility","volume_z","unsup_score","sup_prob","risk_score"]
cols_to_show = [c for c in cols_to_show if c in display_df.columns]
st.dataframe(top_alerts[cols_to_show], use_container_width=True)

# ---------------------- Visualization: Risk over time ----------------------
st.subheader("📈 Risk over Time")
try:
    fig_risk = px.line(display_df.sort_values("datetime"), x="datetime", y="risk_score", color="symbol", title="Risk score over time")
    st.plotly_chart(fig_risk, use_container_width=True)
except Exception as e:
    st.warning(f"Risk plot failed: {e}")

# ---------------------- Evaluation metrics (if labels present) ----------------------
st.subheader("🧪 Evaluation Metrics (if ground truth exists)")
metrics = compute_metrics_if_possible(display_df, label_col="fraud_flag", prob_col="sup_prob", thresh=thresh)
if metrics is None:
    st.info("No `fraud_flag` labels detected in the dataset. If you have ground truth 0/1 labels, add a `fraud_flag` column to enable evaluation metrics.")
else:
    fpr,tpr,roc_auc = metrics["roc"]
    rec,prec,pr_auc = metrics["pr"]
    # ROC
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC={roc_auc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig_roc, use_container_width=True)
    # PR
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"PR AUC={pr_auc:.3f}"))
    fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, use_container_width=True)
    # Confusion matrix & table
    st.write("Confusion matrix (threshold = {:.2f})".format(thresh))
    cm = metrics["cm"]
    cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
    st.dataframe(cm_df, use_container_width=True)
    st.write("Classification report")
    st.dataframe(pd.DataFrame(metrics["report"]).T, use_container_width=True)

# ---------------------- Explainability (SHAP) ----------------------
st.subheader("🧠 Explainability (SHAP)")
if shap is None or xgb is None or xgb_model is None:
    st.info("SHAP explainability requires `shap` and a trained XGBoost classifier. Install shap and ensure xgboost model is present to enable SHAP plots.")
else:
    try:
        # get a small sample for speed
        sample_n = min(100, len(display_df))
        sample = display_df[model_feat_list].replace([np.inf,-np.inf], np.nan).fillna(0.0).sample(sample_n, random_state=42)
        # make sure Xs order matches model_feat_list
        # use the scaled features Xs_used if available and matched, else scale sample freshly
        if 'Xs_used' in locals() and Xs_used.shape[0] == len(display_df):
            # locate indices sampled
            sample_idx = sample.index
            Xs_sample = Xs_used[sample_idx, :]
        else:
            # scale with scaler
            try:
                Xs_sample = scaler.transform(sample.values)
            except Exception:
                Xs_sample = StandardScaler().fit_transform(sample.values)
        explainer = shap.Explainer(xgb_model) if hasattr(shap, "Explainer") else shap.TreeExplainer(xgb_model)
        shap_vals = explainer(Xs_sample)
        st.pyplot(shap.plots.beeswarm(shap_vals, show=False).figure, bbox_inches='tight')
    except Exception as e:
        st.warning(f"SHAP generation failed: {e}")

# ---------------------- News Scanner ----------------------
st.subheader("📰 News Headline Scanner")
# find likely headline columns
headline_options = [c for c in display_df.columns if c.lower() in ("headline","news_headline","news","title","news_title","headline_text")]
selected_headline_col = st.selectbox("Choose headline column (if present)", options=["(none)"] + headline_options)
if selected_headline_col == "(none)":
    pasted = st.text_area("Or paste headlines (one per line) to scan for risky keywords:")
    news_df = pd.DataFrame({"headline":[s for s in pasted.splitlines() if s.strip()]}) if pasted.strip() else pd.DataFrame()
else:
    news_df = display_df[[selected_headline_col]].dropna().rename(columns={selected_headline_col:"headline"})
scan_df = news_keywords_hits(news_df, col_names=["headline"]) if not news_df.empty else pd.DataFrame()
if scan_df.empty:
    st.write("No keyword triggers found (or no headlines provided).")
else:
    st.success("Keyword-triggered items found")
    st.dataframe(scan_df, use_container_width=True)

# ---------------------- National Security Framing and Exports ----------------------
with st.expander("🛡️ National security framing & demo guidance (Open for judges)"):
    st.markdown("""
    **Why this matters:** financial market manipulation and coordinated attacks against key firms can degrade confidence,
    cause liquidity shocks, and propagate into the real economy. This tool provides early-warning signals, explainability,
    and rapid triage for regulatory/market infrastructure teams.
    """)
    st.markdown("**Demo script (60–90s)**")
    st.markdown("""
    1. Show KPIs (Total observations, High-risk flags, Avg risk).  
    2. Open Risk-over-time — point to a spike.  
    3. Show the top alert row and explain: `price_gap_pct`, `volume_z`, and `unsup_score` contributions.  
    4. Paste a provocative headline (e.g., 'SEBI issues notice...') and show news-trigger alert.  
    5. Open SHAP plot to explain feature contributions for the supervised signal.  
    6. Mention national security angle: systemic risk & retail trust.
    """)

# Downloads
st.write("### ⬇️ Exports")
buf = io.StringIO()
display_df.to_csv(buf, index=False)
st.download_button("Download scored CSV", buf.getvalue(), file_name="scored_alerts.csv")

demo_script_txt = """Demo script:
1. KPIs -> 2. Risk-over-time -> 3. Top alert rationale -> 4. News trigger -> 5. SHAP explainability -> 6. National security framing
"""
st.download_button("Download demo script (text)", demo_script_txt, file_name="demo_script.txt")

st.success("Ready — use the Presentation Guide above when you demo this to judges.")
