import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Earthquake–Tsunami ML Prototype", layout="wide")
st.sidebar.title("Prototype Controls")
st.sidebar.markdown("Upload data → Explore → Train → Predict")

st.title("🌊 Earthquake–Tsunami ML Prototype")
st.caption("Upload a CSV, explore EDA, train a model, and make predictions interactively.")

# -----------------------------------------------------------------------------
# Session state defaults
# -----------------------------------------------------------------------------
if "run_id" not in st.session_state:
    st.session_state.run_id = 0
if "trained" not in st.session_state:
    st.session_state.trained = False
if "t_star" not in st.session_state:
    st.session_state.t_star = 0.50

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def best_threshold(y_true, y_proba, mode="f1"):
    """Return threshold t* that optimizes a chosen criterion on validation data."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if mode == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        j = tpr - fpr  # maximize TPR - FPR
        # roc_curve returns thresholds aligned with points; guard edges
        idx = int(np.argmax(j))
        return float(np.clip(thr[idx], 0.01, 0.99))

    # grid search for F1 / Accuracy (stable & simple)
    grid = np.linspace(0.05, 0.95, 181)
    scores = []
    if mode == "accuracy":
        scorer = lambda yt, yp: accuracy_score(yt, yp)
    else:  # "f1"
        scorer = lambda yt, yp: f1_score(yt, yp)

    for t in grid:
        yp = (y_proba >= t).astype(int)
        scores.append(scorer(y_true, yp))
    return float(grid[int(np.argmax(scores))])

def metric_summary(y_true, y_pred, y_proba):
    acc  = accuracy_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_proba)
    return acc, rec, prec, f1, auc

# -----------------------------------------------------------------------------
# 1) Upload data
# -----------------------------------------------------------------------------
uploaded = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin. For your dataset, use the Global Earthquake–Tsunami CSV.")
    st.stop()

@st.cache_data
def load_df(file):
    return pd.read_csv(file)

df = load_df(uploaded)
st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------------------------------------------------------
# 2) Modeling setup
# -----------------------------------------------------------------------------
with st.expander("⚙️ Modeling Setup", expanded=True):
    cols = df.columns.tolist()
    target = st.selectbox("Select target (binary 0/1)", options=cols, index=cols.index("tsunami") if "tsunami" in cols else 0)
    feature_cols = st.multiselect(
        "Select feature columns",
        options=[c for c in cols if c != target],
        default=[c for c in ["magnitude","depth","latitude","longitude","sig"] if c in cols]
    )

    # Validate target is binary
    if df[target].nunique() != 2:
        st.error(f"Target `{target}` must be binary (two unique values). Found: {df[target].unique()}")
        st.stop()

    # Split numeric / categorical
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    st.markdown(f"**Numeric features:** {num_cols if num_cols else '—'}")
    st.markdown(f"**Categorical features:** {cat_cols if cat_cols else '—'}")

# -----------------------------------------------------------------------------
# 3) EDA (light but informative)
# -----------------------------------------------------------------------------
st.header("🔎 EDA")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Shape")
    st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
with c2:
    st.subheader("Missing")
    miss = df.isna().sum().sum()
    st.write(f"Any missing? {'Yes' if miss > 0 else 'No'}")
with c3:
    st.subheader("Class balance")
    bal = df[target].value_counts().rename(index={0:"Non-tsunami",1:"Tsunami"}).reset_index()
    bal.columns = ["Class","Count"]
    fig_bal = px.bar(bal, x="Class", y="Count", text="Count", title="Class Balance").update_traces(textposition="outside")
    st.plotly_chart(fig_bal, use_container_width=True, key=f"class_balance_{st.session_state.run_id}")

st.subheader("Describe (numeric)")
st.dataframe(df.select_dtypes(include=[np.number]).describe().T, use_container_width=True)

dist_cols = [c for c in ["magnitude","depth","sig"] if c in df.columns]
if dist_cols:
    st.subheader("Distributions by class")
    cols_d = st.columns(len(dist_cols))
    for i, c in enumerate(dist_cols):
        fig = px.histogram(df, x=c, color=target, nbins=35, marginal="box", opacity=0.7, title=f"{c} distribution")
        cols_d[i].plotly_chart(fig, use_container_width=True, key=f"dist_{c}_{st.session_state.run_id}")

num_for_corr = df.select_dtypes(include=[np.number])
if num_for_corr.shape[1] >= 2:
    st.subheader("Correlation heatmap (numeric)")
    corr = num_for_corr.corr(method="pearson")
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson correlation")
    st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_{st.session_state.run_id}")
    st.caption("Note: `nst–year` correlation reflects changing instrumentation/reporting over time → metadata, not tsunami physics.")

if {"latitude","longitude"}.issubset(df.columns):
    st.subheader("Epicenters (global map)")
    fig_geo = px.scatter_geo(df, lat="latitude", lon="longitude", color=target,
                             title="Epicenter locations (color = tsunami class)",
                             projection="natural earth", opacity=0.7)
    st.plotly_chart(fig_geo, use_container_width=True, key=f"geo_{st.session_state.run_id}")

# -----------------------------------------------------------------------------
# 4) Model Training (with best-threshold selection)
# -----------------------------------------------------------------------------
st.header("🧠 Model Training")
test_size   = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state= st.sidebar.number_input("Random state", value=42, step=1)

# Choose algorithm + operating point rule
algo = st.selectbox("Choose model", ["Logistic Regression","Random Forest","Neural Network (MLP)"])
criterion = st.selectbox("Operating point (choose best threshold by)", ["Best F1","Best Accuracy","Best Youden (TPR−FPR)"], index=0)

# Build data
X = df[feature_cols].copy()
y = df[target].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=random_state, stratify=y)

# Preprocess
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ],
    remainder="drop"
)

# Models with sensible defaults
if algo == "Logistic Regression":
    clf = LogisticRegression(max_iter=400, class_weight="balanced")
elif algo == "Random Forest":
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
else:
    clf = MLPClassifier(
        hidden_layer_sizes=(32,16),
        activation="relu",
        alpha=1e-4,
        early_stopping=True,
        max_iter=1000,
        random_state=random_state
    )

pipe = Pipeline([("pre", pre), ("clf", clf)])

if st.button("Train model", type="primary", key="train_button"):
    st.session_state.run_id += 1
    run_id = st.session_state.run_id

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    mode_map = {"Best F1":"f1", "Best Accuracy":"accuracy", "Best Youden (TPR−FPR)":"youden"}
    t_star = best_threshold(y_test, y_proba, mode=mode_map[criterion])

    y_pred = (y_proba >= t_star).astype(int)
    acc, rec, prec, f1, auc = metric_summary(y_test, y_pred, y_proba)

    # Save for prediction form
    st.session_state.trained   = True
    st.session_state.pipe      = pipe
    st.session_state.features  = feature_cols
    st.session_state.num_cols  = num_cols
    st.session_state.cat_cols  = cat_cols
    st.session_state.t_star    = float(t_star)

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Recall (1)", f"{rec:.3f}")
    c3.metric("Precision (1)", f"{prec:.3f}")
    c4.metric("F1 (1)", f"{f1:.3f}")
    c5.metric("ROC–AUC", f"{auc:.3f}")
    st.caption(f"Operating threshold selected by **{criterion}**: t* = **{t_star:.2f}**")

    # Confusion Matrix @ t*
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = np.array([[tn, fp],[fn, tp]])
    fig_cm = px.imshow(cm, text_auto=True, x=["Pred 0","Pred 1"], y=["True 0","True 1"],
                       color_continuous_scale="Blues", title=f"Confusion Matrix (threshold = {t_star:.2f})")
    st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{run_id}")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR (Recall)")
    st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_{run_id}")

    # PR
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines", name="PR"))
    fig_pr.update_layout(title="Precision–Recall Curve (class = 1)", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, use_container_width=True, key=f"pr_{run_id}")

# -----------------------------------------------------------------------------
# 5) Prediction form (uses trained pipeline + chosen threshold)
# -----------------------------------------------------------------------------
if st.session_state.get("trained", False):
    st.header("🔮 Make a Prediction (single record)")
    with st.form("predict_form"):
        inputs = {}
        for c in st.session_state.features:
            if c in st.session_state.num_cols:
                default = float(df[c].median()) if np.isfinite(df[c].median()) else 0.0
                inputs[c] = st.number_input(f"{c}", value=default)
            else:
                opts = sorted([str(v) for v in df[c].dropna().unique().tolist()][:50])
                default_opt = 0 if opts else None
                inputs[c] = st.selectbox(f"{c}", options=opts, index=default_opt)
        submit = st.form_submit_button("Predict tsunami probability")

    if submit:
        x_row = pd.DataFrame([inputs])
        p = st.session_state.pipe.predict_proba(x_row)[0, 1]
        yhat = int(p >= st.session_state.t_star)
        st.success(
            f"Predicted probability (class=1 tsunami) = **{p:.3f}** "
            f"→ Predicted class @ t*={st.session_state.t_star:.2f}: **{yhat}**"
        )
        st.caption("The same operating threshold t* (chosen on validation) is used for production-style predictions.")
else:
    st.info("Train a model first to enable the prediction form.")
