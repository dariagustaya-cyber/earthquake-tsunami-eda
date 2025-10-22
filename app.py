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

st.title("Earthquake–Tsunami ML Prototype")
st.caption("Upload a CSV, explore EDA, train a neural net, and make single-record predictions.")

# =============================================================================
# Helpers
# =============================================================================
def metric_summary(y_true, y_pred, y_proba):
    acc  = accuracy_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
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
# 2) Fixed modeling setup (broad feature set like before)
# =============================================================================
TARGET = "tsunami"
if TARGET not in df.columns:
    st.error("Column 'tsunami' not found in your file. Please upload the expected dataset.")
    st.stop()
if df[TARGET].nunique() != 2:
    st.error("Target 'tsunami' must be binary (0/1).")
    st.stop()

# This mirrors the wider set you used earlier (kept only if present).
CANDIDATE_FEATURES = [
    "magnitude", "depth", "latitude", "longitude", "sig",
    "mmi", "cdi", "nst", "dmin", "gap",
    "year", "Year", "Month", "month"
]
# keep order but include only columns that exist
seen = set()
FEATURES = []
for c in CANDIDATE_FEATURES:
    if c in df.columns and c not in seen:
        FEATURES.append(c); seen.add(c)

if len(FEATURES) < 4:
    st.error("Not enough features found. At least 4 of the expected columns are needed.")
    st.stop()

# Locked split for stability (same as before)
TEST_SIZE = 0.20
RANDOM_STATE = 42
st.sidebar.write(f"**Test size:** {TEST_SIZE:.2f} (locked)")
st.sidebar.write(f"**Random state:** {RANDOM_STATE} (locked)")

# =============================================================================
# 3) EDA (compact)
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
    st.caption("Note: metadata correlations (e.g., nst–year) can be strong; we include them here to mirror the earlier setup.")

if {"latitude", "longitude"}.issubset(df.columns):
    st.subheader("Epicenters (global map)")
    fig_geo = px.scatter_geo(
        df, lat="latitude", lon="longitude", color=TARGET,
        title="Epicenter locations (color = tsunami class)",
        projection="natural earth", opacity=0.7
    )
    st.plotly_chart(fig_geo, use_container_width=True, key=f"geo_{st.session_state.run_id}")

# =============================================================================
# 4) Train Neural Network (MLP) — same “old way”: predict() with default 0.5
# =============================================================================
st.header("Train Neural Network (MLP) — classic setup")

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

# MLP similar to before; early stopping on internal validation
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    early_stopping=True,
    max_iter=1000,
    random_state=RANDOM_STATE
)
pipe = Pipeline([("pre", pre), ("clf", mlp)])

if st.button("Train MLP", type="primary", key="train_button"):
    st.session_state.run_id += 1
    run_id = st.session_state.run_id

    pipe.fit(X_train, y_train)

    # classic predict at 0.5
    y_pred  = pipe.predict(X_test)
    # proba for ROC/PR only
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    # Metrics
    acc, rec, prec, f1, auc = metric_summary(y_test, y_pred, y_proba)

    # Save for prediction & download
    st.session_state.trained  = True
    st.session_state.pipe     = pipe
    st.session_state.features = FEATURES
    st.session_state.X_test   = X_test
    st.session_state.y_test   = y_test.values
    st.session_state.y_pred   = y_pred
    st.session_state.y_proba  = y_proba

    # Metric tiles
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{acc:.3f}")
    c2.metric("Recall (1)", f"{rec:.3f}")
    c3.metric("Precision (1)", f"{prec:.3f}")
    c4.metric("F1 (1)", f"{f1:.3f}")
    c5.metric("ROC–AUC", f"{auc:.3f}" if not np.isnan(auc) else "—")

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = np.array([[tn, fp], [fn, tp]])
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
        color_continuous_scale="Blues",
        title="Confusion Matrix (threshold = 0.50)"
    )
    st.plotly_chart(fig_cm, use_container_width=True, key=f"cm_{run_id}")

    # ROC/PR if we have probabilities
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR (Recall)")
        st.plotly_chart(fig_roc, use_container_width=True, key=f"roc_{run_id}")

        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines", name="PR"))
        fig_pr.update_layout(title="Precision–Recall Curve (class = 1)", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, use_container_width=True, key=f"pr_{run_id}")

    # Download results (CSV of test set with predicted class & probability if available)
    results = X_test.copy()
    results["y_true"]  = y_test.values
    results["y_pred"]  = y_pred
    if y_proba is not None:
        results["proba_1"] = y_proba
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download test predictions (CSV)",
        data=csv_bytes,
        file_name="mlp_test_predictions.csv",
        mime="text/csv",
        key=f"dl_results_{run_id}"
    )

# =============================================================================
# 5) Single-record prediction (same pipeline, default 0.5)
# =============================================================================
if st.session_state.get("trained", False):
    st.header("Make a Prediction (single record)")
    with st.form("predict_form"):
        inputs = {}
        for c in st.session_state.features:
            # these columns are numeric in this dataset
            default = float(df[c].median()) if np.isfinite(df[c].median()) else 0.0
            inputs[c] = st.number_input(f"{c}", value=default)
        submit = st.form_submit_button("Predict tsunami probability")

    if submit:
        x_row = pd.DataFrame([inputs])
        proba = st.session_state.pipe.predict_proba(x_row)[:, 1][0]
        yhat  = int(proba >= 0.5)
        st.success(
            f"Predicted probability (class=1 tsunami) = {proba:.3f} "
            f"→ Predicted class (threshold 0.5): {yhat}"
        )
else:
    st.info("Train the model to enable the prediction form.")
