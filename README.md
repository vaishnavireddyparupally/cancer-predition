# Breast Cancer Prediction (Machine Learning + Streamlit)

Predicts whether a tumour is **benign or malignant** from 10 cell-nucleus measurements
using the Breast Cancer Wisconsin dataset (569 patients, bundled with scikit-learn, so
no CSV download is needed).

> Educational project only. Not a medical device.

## Files
| File | Purpose |
|---|---|
| `train_model.py` | Loads data, trains 3 models, cross-validates, saves `model_artifacts.joblib` |
| `app.py` | Streamlit app (Predict, Model performance, Data tabs) |
| `requirements.txt` | Python dependencies |
| `Breast_Cancer_ML_Colab.ipynb` | One-click Google Colab notebook (EDA, training, launches the app) |

## Run in VS Code
```bash
# 1. open the folder in VS Code, then open the terminal (Ctrl + `)
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt

python train_model.py            # creates model_artifacts.joblib
streamlit run app.py             # opens http://localhost:8501
```

## Run in Google Colab
1. Go to https://colab.research.google.com and choose **File > Upload notebook**.
2. Upload `Breast_Cancer_ML_Colab.ipynb`.
3. **Runtime > Run all**. The last cell prints a link ending in `loca.lt`.
4. Open it. If asked for a password, use the IP address printed just above the link.

## Models
Logistic Regression, Random Forest and Gradient Boosting, compared with 5-fold
cross-validation and a stratified 80/20 test split. Scaling is done inside a pipeline
so no test data leaks into training.

## Ideas to extend
- Use all 30 features, or add hyperparameter tuning with `GridSearchCV`
- Add SHAP explanations to the Predict tab
- Deploy free on Streamlit Community Cloud (push to GitHub, connect the repo)
