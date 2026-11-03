# Predicting Acute Kidney Injury in Acute Myocardial Infarction Patients

## 📌 Overview
This project uses **MIMIC-IV** clinical data to predict **acute kidney injury (AKI) within 7 days** in patients with **acute myocardial infarction (AMI)**, using clinical information collected around the first **12 hours of ICU admission**. The model is trained and internally validated using the **MIMIC-IV** database, with external validation performed on an independent dataset from the **eICU** database.

**Live Web Application:** [kidney-injury-prediction.streamlit.app](https://kidney-injury-prediction.streamlit.app/)


## 🛠️ Pipeline

1. **`01_create-dataset.py`** — Build the AMI patient cohort, apply exclusion criteria, and assign AKI labels based on KDIGO stages.
2. **`02_extract-features.py`** — Extract demographic, vital sign, laboratory, urine output, respiratory, medication, and cardiac features.
3. **`03_preprocess-data.py`** — Split MIMIC-IV data into training and internal test sets (80/20), handle missing values and outliers, encode categorical variables, and prepare eICU data.
4. **`04_train.py`** — Train Logistic Regression, Random Forest, XGBoost, and LightGBM using **5-fold cross-validation** and Optuna hyperparameter tuning.
5. **`05_evaluate.py`** — Evaluate the best model (XGBoost) on both MIMIC and eICU dataset using AUROC, AUPRC, Brier Score, PPV, and sensitivity; generate ROC, confusion matrix, calibration, decision curve, and SHAP plots.

## 🚀 Usage

The MIMIC-IV and eICU Collaborative Research Database datasets are available through [PhysioNet](https://physionet.org/), subject to credentialed access and data use agreements.
- **MIMIC-IV:** https://physionet.org/content/mimiciv/
- **eICU-CRD:** https://physionet.org/content/eicu-crd/

Place the required MIMIC-IV, derived, and KDIGO data files in the paths expected by the scripts. Also prepare the eICU feature dataset at `data/raw/AKI_patients_with_features_eICU.csv`.

Run the following scripts:
```bash
python src/01_create-dataset.py
python src/02_extract-features.py
python src/03_preprocess-data.py
python src/04_train.py
python src/05_evaluate.py
```

**Outputs:** Preprocessed datasets are saved in `data/preprocessed/`, trained models and cross-validation results in `models/`, and evaluation metrics and figures in `figures/`.


## 📁 Demo Repository Structure
Put the required files following this structure to properly run the code.
```text
Kidney Injury Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── 01_create-dataset.py
│   ├── 02_extract-features.py
│   ├── 03_preprocess-data.py
│   ├── 04_train.py
│   └── 05_evaluate.py
│
├── data/
│   ├── mimiciv/
│   │   ├── icu/
│   │   │   └── icustays.csv
│   │   └── hosp/
│   │       ├── diagnoses_icd.csv
│   │       └── patients.csv
│   │
│   ├── derived/
│   │   └── organfailure/
│   │       └── kdigo_stages.csv
│   │
│   ├── mimic-derived/
│   │   ├── demographics/
│   │   │   └── icustay_detail.csv
│   │   │
│   │   ├── firstday/
│   │   │   ├── first_day_weight.csv
│   │   │   └── first_day_height.csv
│   │   │
│   │   ├── measurement/
│   │   │   ├── vitalsign.csv
│   │   │   ├── chemistry.csv
│   │   │   ├── complete_blood_count.csv
│   │   │   ├── cardiac_marker.csv
│   │   │   ├── coagulation.csv
│   │   │   ├── enzyme.csv
│   │   │   ├── urine_output.csv
│   │   │   ├── ventilator_setting.csv
│   │   │   ├── blood_differential.csv
│   │   │   └── rhythm.csv
│   │   │
│   │   ├── treatment/
│   │   │   └── ventilation.csv
│   │   │
│   │   └── medication/
│   │       └── nsaid.csv
│   │
│   ├── preprocessed/
│   │   └── xgboost_preprocessor.joblib
│   │
│   └── raw/
│       └── mock_AKI_patients.csv
│
└── models/
    ├── best_params.json
    └── xgboost_features.json
    └── xgboost_model.joblib
```
