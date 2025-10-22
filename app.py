import io
import json
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
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

st.set_page_config(page_title="Earthquake–Tsunami ML Prototype", layout="wide")

# ---------------------------
# Sidebar: app controls
# ---------------------------
st.sidebar.title("Prototype Controls")
st.sidebar.markdown("Upload data → Explore → Train → Predict")

st.title("🌊 Earthquake–Tsunami ML Prototype")
st.caption("Upload a CSV, explore EDA, train a model, and make predictions interactively.")

# ---------------------------
# 1) Upload data
# ---------------------------
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

# ---------------------------
# 2) Choose target and features
# ---------------------------
with st.expander("⚙️ Modeling Setup", expanded=True):
    cols = df.columns.tolist()
    target = st.selectbox("Select target (binary 0/1)", options=cols, index=cols.index("tsunami") if "tsunami" in cols else 0)
    feature_cols = st.multiselect("Select feature columns", options=[c for c in cols if c != target],
                                  default=[c for c in ["magnitude","depth","latitude","longitude","sig"] if c in cols])

    # Validate target is binary
    if df[target].nunique() != 2:
        st.error(f"Target {target} must be binary (two unique values). Found: {df[target].unique()}")
        st.stop()

    # Split numeric / categorical
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    st.markdown(f"**Numeric features:** {num_cols if num_cols else '—'}")
    st.markdown(f"**Categorical features:** {cat_cols if cat_cols else '—'}")

# ---------------------------
# 3) EDA
# ---------------------------
st.header("🔎 EDA")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Shape")
    st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")
with col2:
    st.subheader("Missing")
    miss = df.isna().sum()
    st.write(f"Any missing? {'Yes' if miss.sum() > 0 else 'No'}")
with col3:
    st.subheader("Class balance")
    bal = df[target].value_counts().rename(index={0: "Non-tsunami", 1: "Tsunami"}).reset_index()
    bal.columns = ["Class","Count"]
    st.plotly_chart(
        px.bar(bal, x="Class", y="Count", text="Count", title="Class Balance").update_traces(textposition="outside"),
        use_container_width=True, key="class_balance"
    )

st.subheader("Describe (numeric)")
st.dataframe(df.select_dtypes(include=[np.number]).describe().T, use_container_width=True)

# Distributions for a few key vars (if present)
with col3:
    st.subheader("Class balance")
    bal = df[target].value_counts().rename(index={0: "Non-tsunami", 1: "Tsunami"}).reset_index()
    bal.columns = ["Class", "Count"]
    fig_bal = px.bar(
        bal,
        x="Class",
        y="Count",
        text="Count",
        title="Class Balance"
    ).update_traces(textposition="outside")
    st.plotly_chart(fig_bal, use_container_width=True, key=f"class_balance_{st.session_state.get('run_id', 0)}")
    st.caption("Note: nst–year correlation reflects changing instrumentation/reporting over time → metadata, not tsunami physics.")

# Map (if lat/lon present)
if {"latitude","longitude"}.issubset(df.columns):
    st.subheader("Epicenters (global map)")
    fig_geo = px.scatter_geo(df, lat="latitude", lon="longitude", color=target,
                             title="Epicenter locations (color = tsunami class)",
                             projection="natural earth", opacity=0.7)
    st.plotly_chart(fig_geo, use_container_width=True, key="geo_map")

# ---------------------------
# 4) Model Training
# ---------------------------
st.header("🧠 Model Training")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
X = df[feature_cols].copy()
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# Preprocess
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ],
    remainder="drop"
)

# Choose algorithm
algo = st.selectbox("Choose model", ["Logistic Regression", "Random Forest", "Neural Network (MLP)"])
if algo == "Logistic Regression":
    clf = LogisticRegression(max_iter=200)
elif algo == "Random Forest":
    clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
else:
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", max_iter=1000, early_stopping=True)

pipe = Pipeline([("pre", pre), ("clf", clf)])

# Train
if st.button("Train model", type="primary"):
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    rocA = roc_auc_score(y_test, y_proba)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Recall (1)", f"{rec:.3f}")
    c3.metric("Precision (1)", f"{prec:.3f}")
    c4.metric("F1 (1)", f"{f1:.3f}")
    c5.metric("ROC–AUC", f"{rocA:.3f}")

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = np.array([[tn, fp], [fn, tp]])
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Pred 0", "Pred 1"], y=["True 0", "True 1"],
        color_continuous_scale="Blues",
        title="Confusion Matrix (threshold = 0.5)"
    )
    st.plotly_chart(fig_cm, use_container_width=True, key="cm_final")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={rocA:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR (Recall)")
    st.plotly_chart(fig_roc, use_container_width=True, key="roc_curve")

    # PR Curve
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=rec_curve, y=prec_curve, mode="lines", name="PR"))
    fig_pr.update_layout(title="Precision–Recall Curve (class = 1)", xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig_pr, use_container_width=True, key="pr_curve")

    # Save to session for prediction form
    st.session_state.pipe = pipe
    st.session_state.trained = True

# ---------------------------
# 5) Prediction form
# ---------------------------
if st.session_state.get("trained", False):
    st.header("🔮 Make a Prediction (single record)")
    with st.form("predict_form"):
        inputs = {}
        for c in feature_cols:
            if c in num_cols:
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
        yhat = int(p >= 0.5)
        st.success(
f"Predicted probability (class=1 tsunami) = {p:.3f} "
            f"→ Predicted class (threshold 0.5): **{yhat}**"
        )
else:
    st.info("Train a model first to enable the prediction form.")
