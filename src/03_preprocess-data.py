#!/usr/bin/env python3

from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLS = ["subject_id", "hadm_id", "stay_id"]

CATEGORICAL_COLS = [
    "gender", "first_hosp_stay",
    "non_invasive_vent_flag", "use_aspirin", "is_afib", "is_high_grade_block"
]

TARGET = "aki_label"

NUMERICAL_COLS = [
    "admission_age", "weight", "height", "heart_rate_max", "sys_bp_min",
    "mean_bp_min", "dias_bp_min", "resp_rate_max", "temperature_max",
    "glucose_vitalsign_max", "creatinine_min", "bun_max", "bicarbonate_min",
    "potassium_chemistry_max", "aniongap_max", "albumin_median",
    "glucose_chemistry_max", "hemoglobin_min", "hematocrit_min", "wbc_max",
    "platelet_median", "rdw_median", "troponin_t_max", "ck_mb_max",
    "ntprobnp_max", "inr_median", "pt_median", "ptt_median",
    "fibrinogen_median", "ck_cpk_max", "ld_ldh_max", "alt_max", "ast_max",
    "bilirubin_total_max", "alp_median", "amylase_median",
    "urineoutput_12h_sum", "peep_max", "fio2_max", "plateau_pressure_max",
    "tidal_volume_observed_max", "bands_max"
]


def load_and_split(mimic_path, eicu_path):
    mimic = pd.read_csv(mimic_path)
    eicu = pd.read_csv(eicu_path)

    mimic = mimic.loc[:, ~mimic.columns.str.startswith("Unnamed")]
    eicu = eicu.loc[:, ~eicu.columns.str.startswith("Unnamed")]

    required_cols = ID_COLS + CATEGORICAL_COLS + NUMERICAL_COLS + [TARGET]

    missing_mimic = [c for c in required_cols if c not in mimic.columns]
    missing_eicu = [c for c in required_cols if c not in eicu.columns]

    if missing_mimic:
        raise ValueError(f"MIMIC missing columns: {missing_mimic}")
    if missing_eicu:
        raise ValueError(f"eICU missing columns: {missing_eicu}")

    mimic = mimic[required_cols].copy()
    eicu = eicu[required_cols].copy()

    train_df, test_df = train_test_split(
        mimic,
        test_size=0.2,
        random_state=42,
        stratify=mimic[TARGET]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), eicu.reset_index(drop=True)


def prepare_common(train_df, test_df, eicu_df):
    feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS

    missing_rate = train_df[feature_cols].isna().mean()
    drop_cols = missing_rate[missing_rate > 0.5].index.tolist()

    categorical_cols = [c for c in CATEGORICAL_COLS if c not in drop_cols]
    numerical_cols = [c for c in NUMERICAL_COLS if c not in drop_cols]

    # Cap numerical features using the 1st and 99th percentiles of the training set.
    for col in numerical_cols:
        lower = train_df[col].quantile(0.01)
        upper = train_df[col].quantile(0.99)

        train_df[col] = train_df[col].clip(lower, upper)
        test_df[col] = test_df[col].clip(lower, upper)
        eicu_df[col] = eicu_df[col].clip(lower, upper)

    print(f"Dropped >50% missing: {drop_cols}")

    return train_df, test_df, eicu_df, categorical_cols, numerical_cols


def preprocess_sklearn(train_df, test_df, eicu_df, categorical_cols, numerical_cols, scale=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer([
        ("num", Pipeline(num_steps), numerical_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical_cols)
    ])

    feature_cols = numerical_cols + categorical_cols

    X_train = preprocessor.fit_transform(train_df[feature_cols])
    X_test = preprocessor.transform(test_df[feature_cols])
    X_eicu = preprocessor.transform(eicu_df[feature_cols])

    names = [
        x.replace("num__", "").replace("cat__", "")
        for x in preprocessor.get_feature_names_out()
    ]

    X_train = pd.DataFrame(X_train, columns=names)
    X_test = pd.DataFrame(X_test, columns=names)
    X_eicu = pd.DataFrame(X_eicu, columns=names)

    return X_train, X_test, X_eicu, preprocessor


def preprocess_native(train_df, test_df, eicu_df, categorical_cols, numerical_cols):
    feature_cols = numerical_cols + categorical_cols

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    X_eicu = eicu_df[feature_cols].copy()

    for col in categorical_cols:
        levels = sorted(X_train[col].dropna().astype(str).unique())

        for df in [X_train, X_test, X_eicu]:
            values = df[col].where(df[col].isna(), df[col].astype(str))
            df[col] = pd.Categorical(values, categories=levels)

    # Keep numerical missing values as np.nan.
    for col in numerical_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
        X_eicu[col] = pd.to_numeric(X_eicu[col], errors="coerce")

    return X_train, X_test, X_eicu


def add_ids_target(X, source_df):
    return pd.concat([
        source_df[ID_COLS].reset_index(drop=True),
        X.reset_index(drop=True),
        source_df[[TARGET]].reset_index(drop=True)
    ], axis=1)


def preprocess(mimic_path, eicu_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, eicu_df = load_and_split(mimic_path, eicu_path)

    train_df, test_df, eicu_df, categorical_cols, numerical_cols = prepare_common(
        train_df, test_df, eicu_df
    )

    # Logistic Regression
    lr_train, lr_test, lr_eicu, lr_preprocessor = preprocess_sklearn(
        train_df, test_df, eicu_df,
        categorical_cols, numerical_cols,
        scale=True
    )

    add_ids_target(lr_train, train_df).to_csv(output_dir / "lr_train.csv", index=False)
    add_ids_target(lr_test, test_df).to_csv(output_dir / "lr_test_internal.csv", index=False)
    add_ids_target(lr_eicu, eicu_df).to_csv(output_dir / "lr_test_eicu.csv", index=False)
    joblib.dump(lr_preprocessor, output_dir / "lr_preprocessor.joblib")

    # Random Forest
    rf_train, rf_test, rf_eicu, rf_preprocessor = preprocess_sklearn(
        train_df, test_df, eicu_df,
        categorical_cols, numerical_cols,
        scale=False
    )

    add_ids_target(rf_train, train_df).to_csv(output_dir / "rf_train.csv", index=False)
    add_ids_target(rf_test, test_df).to_csv(output_dir / "rf_test_internal.csv", index=False)
    add_ids_target(rf_eicu, eicu_df).to_csv(output_dir / "rf_test_eicu.csv", index=False)
    joblib.dump(rf_preprocessor, output_dir / "rf_preprocessor.joblib")

    # XGBoost / LightGBM
    native_train, native_test, native_eicu = preprocess_native(
        train_df, test_df, eicu_df,
        categorical_cols, numerical_cols
    )
    native_preprocessor = {
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "category_levels": {
            col: native_train[col].cat.categories.tolist()
            for col in categorical_cols
        },
    }

    model_datasets = {
        "xgboost": (native_train, native_test, native_eicu),
        "lightgbm": (native_train, native_test, native_eicu),
    }

    for model_name, (model_train, model_test, model_eicu) in model_datasets.items():
        add_ids_target(model_train, train_df).to_csv(output_dir / f"{model_name}_train.csv", index=False)
        add_ids_target(model_test, test_df).to_csv(output_dir / f"{model_name}_test_internal.csv", index=False)
        add_ids_target(model_eicu, eicu_df).to_csv(output_dir / f"{model_name}_test_eicu.csv", index=False)
        joblib.dump(native_preprocessor, output_dir / f"{model_name}_preprocessor.joblib")

    print(f"Train: {len(train_df)}")
    print(f"Internal test: {len(test_df)}")
    print(f"eICU test: {len(eicu_df)}")
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mimic-path",
        type=Path,
        default=Path("data/raw/AKI_patients_with_features.csv")
    )
    parser.add_argument(
        "--eicu-path",
        type=Path,
        default=Path("data/raw/AKI_patients_with_features_eICU.csv")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/preprocessed")
    )

    args = parser.parse_args()
    preprocess(args.mimic_path, args.eicu_path, args.output_dir)