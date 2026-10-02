"""
Breast Cancer Prediction - Model Training
=========================================
Dataset : Breast Cancer Wisconsin (Diagnostic) - bundled with scikit-learn,
          so no download or CSV file is needed.
Target  : 0 = Malignant, 1 = Benign
Inputs  : the 10 "mean" measurements of the cell nuclei (radius, texture, ...)

Trains 3 models, compares them with 5-fold cross-validation, evaluates on a
held-out test set and saves everything the Streamlit app needs.

Run:  python train_model.py
Output: model_artifacts.joblib
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# 1. Load data -----------------------------------------------------------
raw = load_breast_cancer(as_frame=True)
df = raw.frame
mean_cols = [c for c in df.columns if c.startswith("mean ")]
X = df[mean_cols]
y = df["target"]                       # 0 = malignant, 1 = benign
print(f"Dataset: {X.shape[0]} patients, {X.shape[1]} features")
print("Class counts:", y.value_counts().rename({0: 'malignant', 1: 'benign'}).to_dict())

# 2. Train / test split --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# 3. Models (scaling lives inside the pipeline, so no data leakage) -------
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
metrics = {}
for name, model in models.items():
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics[name] = {
        "cv_accuracy": float(cv_acc.mean()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    print(f"{name:<20} CV acc {cv_acc.mean():.3f} | test acc {metrics[name]['accuracy']:.3f}"
          f" | ROC-AUC {metrics[name]['roc_auc']:.3f}")

best_name = max(metrics, key=lambda n: metrics[n]["cv_accuracy"])
print(f"\nBest model (by cross-validation): {best_name}")

# 4. Feature importance of the tree model ---------------------------------
importance = pd.Series(models["Random Forest"].feature_importances_,
                       index=mean_cols).sort_values(ascending=False)

# 5. Save everything the app needs ----------------------------------------
artifacts = {
    "models": models,
    "best_model_name": best_name,
    "metrics": metrics,
    "feature_names": mean_cols,
    "feature_ranges": {c: (float(X[c].min()), float(X[c].max()), float(X[c].mean()))
                       for c in mean_cols},
    "feature_importance": importance.to_dict(),
    "class_names": {0: "Malignant", 1: "Benign"},
    "sample_data": df[mean_cols + ["target"]].sample(150, random_state=RANDOM_STATE),
}
joblib.dump(artifacts, "model_artifacts.joblib")
print("Saved model_artifacts.joblib")
