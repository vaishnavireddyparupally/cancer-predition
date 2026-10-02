"""
Breast Cancer Prediction - Streamlit App
========================================
Run:  streamlit run app.py
Needs model_artifacts.joblib in the same folder (create it with: python train_model.py)

Educational project only - NOT a medical device.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🩺", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load("model_artifacts.joblib")


art = load_artifacts()
models = art["models"]
metrics = art["metrics"]
features = art["feature_names"]
ranges = art["feature_ranges"]
class_names = art["class_names"]

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox(
    "Choose a model", list(models.keys()),
    index=list(models.keys()).index(art["best_model_name"]))
model = models[model_name]
st.sidebar.markdown("**Test accuracy**")
for n, m in metrics.items():
    star = " ⭐" if n == art["best_model_name"] else ""
    st.sidebar.write(f"- {n}: {m['accuracy']:.1%}{star}")
st.sidebar.info("Educational project. Not for real medical decisions.")

st.title("🩺 Breast Cancer Prediction")
st.write("Predict whether a tumour is **benign** or **malignant** from cell-nucleus "
         "measurements (Breast Cancer Wisconsin dataset, 569 patients).")

tab_predict, tab_perf, tab_data = st.tabs(["🔮 Predict", "📊 Model performance", "🗂 Data"])

# ---------------- Tab 1: Predict ----------------
with tab_predict:
    st.subheader("Enter the measurements")
    use_example = st.radio("Start from", ["Average patient", "Typical malignant", "Typical benign"],
                           horizontal=True)
    sample = art["sample_data"]
    if use_example == "Typical malignant":
        defaults = sample[sample["target"] == 0][features].median()
    elif use_example == "Typical benign":
        defaults = sample[sample["target"] == 1][features].median()
    else:
        defaults = pd.Series({f: ranges[f][2] for f in features})

    values = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        lo, hi, _ = ranges[f]
        step = float((hi - lo) / 200)
        with cols[i % 3]:
            values[f] = st.slider(f.replace("mean ", "Mean ").title(), float(lo), float(hi),
                                  float(np.clip(defaults[f], lo, hi)), step=step,
                                  key=f"{use_example}-{f}")

    if st.button("Predict", type="primary"):
        X_new = pd.DataFrame([values])[features]
        pred = int(model.predict(X_new)[0])
        proba = model.predict_proba(X_new)[0]
        label = class_names[pred]
        (st.success if pred == 1 else st.error)(
            f"Prediction: **{label}**  (confidence {proba[pred]:.1%})")
        c1, c2 = st.columns(2)
        c1.metric("Malignant probability", f"{proba[0]:.1%}")
        c2.metric("Benign probability", f"{proba[1]:.1%}")
        st.progress(float(proba[1]), text="Benign probability")

# ---------------- Tab 2: Performance ----------------
with tab_perf:
    st.subheader("Model comparison (held-out test set)")
    table = pd.DataFrame(metrics).T[["cv_accuracy", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    table.columns = ["CV accuracy", "Test accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8d5"),
                 use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Confusion matrix: {model_name}**")
        cm = np.array(metrics[model_name]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Malignant", "Benign"])
        ax.set_yticks([0, 1], ["Malignant", "Benign"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=14)
        st.pyplot(fig)
    with right:
        st.markdown("**Feature importance (Random Forest)**")
        imp = pd.Series(art["feature_importance"]).sort_values()
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.barh([i.replace("mean ", "") for i in imp.index], imp.values, color="#0e7c86")
        ax.set_xlabel("Importance")
        st.pyplot(fig)

# ---------------- Tab 3: Data ----------------
with tab_data:
    st.subheader("Sample of the dataset")
    show = art["sample_data"].copy()
    show["diagnosis"] = show["target"].map(class_names)
    st.dataframe(show.drop(columns="target").head(50), use_container_width=True)
    st.markdown("**Class balance in the sample**")
    st.bar_chart(show["diagnosis"].value_counts())
