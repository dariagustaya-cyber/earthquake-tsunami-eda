import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.neural_network import MLPClassifier

# =============================================================================
# Page & session
# =============================================================================
st.set_page_config(page_title="Earthquake–Tsunami ML Prototype", layout="wide")
st.sidebar.title("Prototype Controls")
st.sidebar.markdown("Upload → EDA → Train → Predict")

if "run_id" not in st.session_state:
    st.session_state.run_id = 0
if "trained" not in st.session_state:
    st.session_state.trained = False
if "t_star" not in st.session_state:
    st.session_state.t_star = 0.50

st.title("Earthquake–Tsunami ML Prototype")
st.caption("Upload a CSV, explore EDA, train a neural net, and make single-record predictions.")

# =============================================================================
# Helpers
# =============================================================================
def best_threshold(y_true, y_proba, mode="accuracy"):
    """
    Return threshold t* that optimizes a chosen criterion on validation data.
    mode ∈ {'accuracy', 'f1', 'youden'}  (we use 'accuracy' by default)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if mode == "youden":
        fpr, tpr, thr = roc_curve(y_true, y_proba)
        j = tpr - fpr
        idx = int(np.argmax(j))
        return float(np.clip(thr[idx], 0.01, 0.99))

    grid = np.linspace(0.05, 0.95, 181)
    if mode == "f1":
        scorer = lambda yt, yp: f1_score(yt, yp)
    else:  # accuracy (default)
        scorer = lambda yt, yp: accuracy_score(yt, yp)

    scores = []
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

# =============================================================================
# 1) Upload data
# =============================================================================
uploaded = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded is None:
    st.info("Upload the Global Earthquake–Tsunami CSV to begin.")
    st.stop()

@st.cache_data
def load_df(file):
    return pd.read_csv(file)

df = load_df(uploaded)
st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
st.dataframe(df.head(), use_container_width=True)

# =============================================================================
# 2) Fixed modeling setup (no user feature choices)
# =============================================================================
TARGET = "tsunami"
if TARGET not in df.columns:
    st.error("Column 'tsunami' not found in your file. Please upload the expected dataset.")
    st.stop()
if df[TARGET].nunique() != 2:
    st.error("Target 'tsunami' must be binary (0/1).")
    st.stop()

# Geophysically meaningful features only
CANDIDATE_FEATURES = ["magnitude", "depth", "latitude", "longitude", "sig"]
FEATURES = [c for c in CANDIDATE_FEATURES if c in df.columns]
if len(FEATURES) < 3:
    st.error("Not enough core features found. Need at least 3 among: magnitude, depth, latitude, longitude, sig.")
    st.stop()

# Locked split for stability (no UI changes)
TEST_SIZE = 0.20
RANDOM_STATE = 42
st.sidebar.write(f"**Test size:** {TEST_SIZE:.2f} (locked)")
st.sidebar.write(f"**Random state:** {RANDOM_STATE} (locked)")

# =============================================================================
# 3) EDA (compact & informative)
# =============================================================================
st.header("EDA")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Shape")
    st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
with c2:
    st.subheader("Missing")
    st.write("Any missing?", "Yes" if df.isna().sum().sum() > 0 else "No")
with c3:
    st.subheader("Class balance")
    bal = df[TARGET].value_counts().rename(index={0: "Non-tsunami", 1: "Tsunami"}).reset_index()
    bal.columns = ["Class", "Count"]
    fig_bal = px.bar(bal, x="Class", y="Count", text="Count", title="Class Balance").update_traces(textposition="outside")
    st.plotly_chart(fig_bal, use_container_width=True, key=f"class_balance_{st.session_state.run_id}")

st.subheader("Describe (numeric)")
st.dataframe(df.select_dtypes(include=[np.number]).describe().T, use_container_width=True)

dist_cols = [c for c in ["magnitude", "depth", "sig"] if c in df.columns]
if dist_cols:
    st.subheader("Distributions by class")
    cols_d = st.columns(len(dist_cols))
    for i, c in enumerate(dist_cols):
        fig = px.histogram(df, x=c, color=TARGET, nbins=35, marginal="box", opacity=0.75, title=f"{c} distribution")
        cols_d[i].plotly_chart(fig, use_container_width=True, key=f"dist_{c}_{st.session_state.run_id}")

num_for_corr = df.select_dtypes(include=[np.number])
if num_for_corr.shape[1] >= 2:
    st.subheader("Correlation heatmap (numeric)")
    corr = num_for_corr.corr(method="pearson")
    fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Pearson correlation")
    st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_{st.session_state.run_id}")
    st.caption("Note: any strong `nst–year` correlation reflects reporting changes, not tsunami physics (we exclude them).")

if {"latitude", "longitude"}.issubset(df.columns):
    st.subheader("Epicenters (global map)")
    fig_geo = px.scatter_geo(
        df, lat="latitude", lon="longitude", color=TARGET,
        title="Epicenter locations (color = tsunami class)",
        projection="natural earth", opacity=0.7
    )
    st.plotly_chart(fig_geo, use_container_width=True, key=f"geo_{st.session_state.run_id}")

# =============================================================================
# 4) Train ONLY Neural Network (MLP) — auto best threshold (Accuracy)
# =============================================================================
st.header("Train Neural Network (MLP)")

# Build matrices
X = df[FEATURES].copy()
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Preprocess + model pipeline
pre = ColumnTransformer(
    transformers=[("num", StandardScaler(), FEATURES)],
    remainder="drop"
)

# Tuned MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    alpha=1e-4,               # mild L2
    early_stopping=True,      # internal validation for generalization
    n_iter_no_change=20,
    max_iter=1200,
    random_state=RANDOM_STATE
)

pipe = Pipeline([("pre", pre), ("clf", mlp)])

if st.button("Train MLP", type="primary", key="train_button"):
    st.session_state.run_id += 1
    run_id = st.session_state.run_id

    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Auto-select best threshold by Accuracy
    t_star = best_threshold(y_test, y_proba, mode="accuracy")
    y_pred = (y_proba >= t_star).astype(int)

    # Metrics @ t*
    acc, rec, prec, f1, auc = metric_summary(y_test, y_pred, y_proba)

    # Save for prediction & download
    st.session_state.trained  = True
    st.session_state.pipe     = pipe
    st.session_state.features = FEATURES
    st.session_state.X_test   = X_test
    st.session_state.y_test   = y_test.values
    st.session_state.y_proba  = y_proba
    st.session_state.y_pred   = y_pred
    st.session_state.t_star   = float(t_star)

    # Metric tiles
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{acc:.3f}")
    c2.metric("Recall (1)", f"{rec:.3f}")
    c3.metric("Precision (1)", f"{prec:.3f}")
    c4.metric("F1 (1)", f"{f1:.3f}")
    c5.metric("ROC–AUC", f"{auc:.3f}")
    st.caption(f"Operating threshold selected automatically for maximum Accuracy: t* = {t_star:.2f}")

    # Confusion Matrix @ t*
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = np.array([[tn, fp], [fn, tp]])
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
        color_continuous_scale="Blues",
        title=f"Confusion Matrix (threshold = {t_star:.2f})"
    )
    st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{run_id}")

    # ROC (threshold-independent)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR (Recall)")
    st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_{run_id}")

    # PR (class-1 focus)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines", name="PR"))
    fig_pr.update_layout(title="Precision–Recall Curve (class = 1)", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, use_container_width=True, key=f"pr_{run_id}")

    # Download results (CSV of test set with proba & preds @ t*)
    results = X_test.copy()
    results["y_true"]    = y_test.values
    results["proba_1"]   = y_proba
    results["pred_1@t*"] = y_pred
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download test predictions (CSV)",
        data=csv_bytes,
        file_name="mlp_test_predictions.csv",
        mime="text/csv",
        key=f"dl_results_{run_id}"
    )

# =============================================================================
# 5) Single-record prediction (uses trained pipeline @ stored t*)
# =============================================================================
if st.session_state.get("trained", False):
    st.header("Make a Prediction (single record)")
    with st.form("predict_form"):
        inputs = {}
        for c in st.session_state.features:
            default = float(df[c].median()) if np.isfinite(df[c].median()) else 0.0
            inputs[c] = st.number_input(f"{c}", value=default)
        submit = st.form_submit_button("Predict tsunami probability")

    if submit:
        x_row = pd.DataFrame([inputs])
        p = st.session_state.pipe.predict_proba(x_row)[0, 1]
        yhat = int(p >= st.session_state.t_star)
        st.success(
            f"Predicted probability (class=1 tsunami) = {p:.3f} "
            f"→ Predicted class @ t*={st.session_state.t_star:.2f}: {yhat}"
        )
        st.caption("The same validation-chosen threshold t* is used for predictions.")
else:
    st.info("Train the model to enable the prediction form.")
