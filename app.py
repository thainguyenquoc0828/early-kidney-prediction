#!/usr/bin/env python3
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="AKI Risk Prediction – XGBoost", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
PREPROCESSED_DIR = BASE_DIR / "data" / "preprocessed"
MODEL_FILE = MODELS_DIR / "xgboost_model.joblib"
THRESHOLD_FILE = MODELS_DIR / "best_params.json"
PREPROCESSOR_CANDIDATES = [
    MODELS_DIR / "xgboost_preprocessor.joblib",
    PREPROCESSED_DIR / "xgboost_preprocessor.joblib",
]

# Units assume original MIMIC-IV units and no post-extraction unit conversion.
UNITS = {
    "admission_age": "years", "weight": "kg", "height": "cm",
    "heart_rate_max": "beats/min",
    "sys_bp_min": "mmHg", "mean_bp_min": "mmHg", "dias_bp_min": "mmHg",
    "resp_rate_max": "breaths/min", "temperature_max": "°C",
    "glucose_vitalsign_max": "mg/dL", "creatinine_min": "mg/dL",
    "bun_max": "mg/dL", "bicarbonate_min": "mEq/L",
    "potassium_chemistry_max": "mEq/L", "aniongap_max": "mEq/L",
    "albumin_median": "g/dL", "glucose_chemistry_max": "mg/dL",
    "hemoglobin_min": "g/dL", "hematocrit_min": "%",
    "wbc_max": "K/µL", "platelet_median": "K/µL", "rdw_median": "%",
    "troponin_t_max": "ng/mL", "ck_mb_max": "ng/mL",
    "ntprobnp_max": "pg/mL", "inr_median": "unitless",
    "pt_median": "s", "ptt_median": "s",
    "fibrinogen_median": "mg/dL", "ck_cpk_max": "U/L",
    "ld_ldh_max": "U/L", "alt_max": "U/L", "ast_max": "U/L",
    "bilirubin_total_max": "mg/dL", "alp_median": "U/L",
    "amylase_median": "U/L", "urineoutput_12h_sum": "mL/12 h",
    "peep_max": "cmH₂O", "fio2_max": "%",
    "plateau_pressure_max": "cmH₂O",
    "tidal_volume_observed_max": "mL",
    "bands_max": "%",
}

# MIMIC-IV sample row from the supplied source file; empty fields remain missing.
SAMPLE_CSV = """subject_id,hadm_id,stay_id,gender,admission_age,aki_label,first_hosp_stay,weight,height,heart_rate_max,sys_bp_min,mean_bp_min,dias_bp_min,resp_rate_max,temperature_max,glucose_vitalsign_max,creatinine_min,bun_max,bicarbonate_min,potassium_chemistry_max,aniongap_max,albumin_median,glucose_chemistry_max,hemoglobin_min,hematocrit_min,wbc_max,platelet_median,rdw_median,troponin_t_max,ck_mb_max,ntprobnp_max,inr_median,pt_median,ptt_median,fibrinogen_median,ck_cpk_max,ld_ldh_max,alt_max,ast_max,bilirubin_total_max,alp_median,amylase_median,urineoutput_12h_sum,peep_max,fio2_max,plateau_pressure_max,tidal_volume_observed_max,non_invasive_vent_flag,use_aspirin,bands_max,is_afib,is_high_grade_block
10002495,24982426,36753294,M,81,1,TRUE,64.1,170,115,72,47,40,26,37.06,331,1.4,35,14,4.1,23,,331,14,40.6,36.8,180.5,12.4,1.75,96,,1.3,14.5,49.6,,,310,,,,,,2955,,,,,0,1,10,0,0
"""
SAMPLE_ROW = next(csv.DictReader(io.StringIO(SAMPLE_CSV)))
MISSING_OPTION = "(Missing data)"


def locate_preprocessor() -> Path:
    for path in PREPROCESSOR_CANDIDATES:
        if path.is_file():
            return path
    checked = "\n".join(f"- {p}" for p in PREPROCESSOR_CANDIDATES)
    raise FileNotFoundError(f"Preprocessor not found. Checked:\n{checked}")


@st.cache_resource(show_spinner="Loading XGBoost model...")
def load_artifacts() -> dict:
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    model = joblib.load(MODEL_FILE)
    config = joblib.load(locate_preprocessor())
    if not isinstance(config, dict):
        raise TypeError("Preprocessor must contain a metadata dictionary.")

    required = {"categorical_cols", "numerical_cols", "category_levels"}
    missing = required - set(config)
    if missing:
        raise KeyError(f"Missing metadata keys: {sorted(missing)}")

    cat_cols = list(config["categorical_cols"])
    num_cols = list(config["numerical_cols"])
    provided = cat_cols + num_cols
    if len(provided) != len(set(provided)):
        raise ValueError("Duplicate columns found across numerical and categorical features.")

    model_features = model.get_booster().feature_names
    if not model_features:
        model_features = list(getattr(model, "feature_names_in_", []))
    if not model_features:
        raise ValueError("Model has no feature names.")
    if set(model_features) != set(provided):
        raise ValueError(
            "Model and preprocessor features do not match. "
            f"Model features missing from metadata: {sorted(set(model_features) - set(provided))}; "
            f"Extra metadata features: {sorted(set(provided) - set(model_features))}"
        )
    if "aki_label" in model_features:
        raise ValueError("aki_label appears among model features: target leakage detected. Retrain the model.")

    threshold = 0.5
    threshold_source = "Default 0.5"
    if THRESHOLD_FILE.is_file():
        with THRESHOLD_FILE.open("r", encoding="utf-8") as f:
            params = json.load(f)
        loaded = params.get("xgboost", {}).get("threshold")
        if loaded is not None:
            loaded = float(loaded)
            if np.isfinite(loaded) and 0 <= loaded <= 1:
                threshold = loaded
                threshold_source = "Out-of-fold threshold from best_params.json"

    levels = {}
    for col in cat_cols:
        if col not in config["category_levels"]:
            raise KeyError(f"Missing category_levels for {col}")
        levels[col] = [str(v) for v in config["category_levels"][col]]

    return {
        "model": model,
        "features": list(model_features),
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "category_levels": levels,
        "threshold": threshold,
        "threshold_source": threshold_source,
    }


def normalize_category(value: str) -> str:
    """Match common boolean encodings, without changing actual model categories."""
    s = str(value).strip().lower()
    if s in {"true", "t", "yes", "1", "1.0"}:
        return "1"
    if s in {"false", "f", "no", "0", "0.0"}:
        return "0"
    return s


def sample_category(column: str, categories: list[str]) -> str:
    raw = SAMPLE_ROW.get(column, "")
    if not raw:
        return MISSING_OPTION
    if raw in categories:
        return raw
    for category in categories:
        if normalize_category(raw) == normalize_category(category):
            return category
    return MISSING_OPTION  # Never create an unseen category.


def build_input(raw: dict, artifacts: dict) -> pd.DataFrame:
    """Match training feature ordering and native Pandas categorical encoding."""
    x_input = pd.DataFrame(index=[0])
    for col in artifacts["features"]:
        value = raw.get(col)
        if col in artifacts["numerical_cols"]:
            x_input[col] = pd.to_numeric(
                pd.Series([value]), errors="coerce"
            ).astype(float)
        else:
            values = [pd.NA] if value is None or pd.isna(value) else [str(value)]
            x_input[col] = pd.Categorical(
                values, categories=artifacts["category_levels"][col]
            )
    return x_input.loc[:, artifacts["features"]]


def parse_numeric(text_value: str, feature: str) -> float:
    s = text_value.strip()
    if not s:
        return np.nan
    try:
        result = float(s)
    except ValueError as exc:
        raise ValueError(f"{feature}: '{text_value}' is not a valid number.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{feature}: enter a finite number or leave the field blank.")
    return result


def predict_patient(x_input: pd.DataFrame, artifacts: dict) -> float:
    proba = artifacts["model"].predict_proba(x_input)
    result = float(proba[0, 1])
    if not np.isfinite(result):
        raise ValueError("Model returned an invalid probability.")
    return result


def compute_contributions(x_input: pd.DataFrame, artifacts: dict):
    """TreeSHAP from XGBoost, expressed in raw log-odds margin units."""
    booster = artifacts["model"].get_booster()
    dm = xgb.DMatrix(
        x_input, enable_categorical=True, feature_names=artifacts["features"]
    )
    contribs = np.asarray(booster.predict(dm, pred_contribs=True))
    if (contribs.ndim != 2 or contribs.shape[0] != 1 or
            contribs.shape[1] != len(artifacts["features"]) + 1):
        raise RuntimeError(f"Unexpected contribution shape: {contribs.shape}")
    result = pd.DataFrame({
        "Feature": artifacts["features"],
        "SHAP value (log-odds)": contribs[0, :-1],
    })
    result["Absolute contribution"] = result["SHAP value (log-odds)"].abs()
    result = result.sort_values(
        "Absolute contribution", ascending=False
    ).reset_index(drop=True)
    return result, float(contribs[0, -1])


def reset_sample(artifacts: dict) -> None:
    for col in artifacts["numerical_cols"]:
        st.session_state[f"num_{col}"] = SAMPLE_ROW.get(col, "")
    for col in artifacts["categorical_cols"]:
        st.session_state[f"cat_{col}"] = sample_category(
            col, artifacts["category_levels"][col]
        )
    st.session_state.pop("prediction_result", None)


st.title("AKI Risk Prediction")

try:
    artifacts = load_artifacts()
except Exception as exc:
    st.error(f"Failed to load model artifacts: {exc}")
    st.info(
        "xgboost_model.joblib and xgboost_preprocessor.joblib must come from the same "
        "training pipeline."
    )
    st.stop()

if "sample_initialized" not in st.session_state:
    reset_sample(artifacts)
    st.session_state["sample_initialized"] = True

with st.sidebar:
    st.header("Patient Information")
    st.button("Predict on sample data", on_click=reset_sample, args=(artifacts,))

    with st.form("aki_form"):
        raw_text = {}
        raw_category = {}
        # Both expanders COLLAPSED on initial page load.
        with st.expander(
            f"Numerical Features ({len(artifacts['numerical_cols'])})", expanded=False
        ):
            for col in artifacts["numerical_cols"]:
                unit = UNITS.get(col)
                label = f"{col} ({unit})" if unit else col
                raw_text[col] = st.text_input(
                    label,
                    placeholder="Enter a number or leave blank",
                    key=f"num_{col}",
                )

        with st.expander(
            f"Categorical Features ({len(artifacts['categorical_cols'])})",
            expanded=False,
        ):
            for col in artifacts["categorical_cols"]:
                options = [MISSING_OPTION] + artifacts["category_levels"][col]
                raw_category[col] = st.selectbox(
                    col, options=options, key=f"cat_{col}"
                )

        submit = st.form_submit_button(
            "Predict AKI", type="primary", use_container_width=True
        )

if submit:
    try:
        raw = {
            col: parse_numeric(value, col)
            for col, value in raw_text.items()
        }
        raw.update({
            col: None if value == MISSING_OPTION else value
            for col, value in raw_category.items()
        })
        x_input = build_input(raw, artifacts)
        missing_count = int(x_input.isna().sum().sum())
        total = len(artifacts["features"])
        if missing_count == total:
            st.error("All features are missing. Please enter patient data before predicting.")
            st.stop()
        probability = predict_patient(x_input, artifacts)
        try:
            contributions, base_margin = compute_contributions(x_input, artifacts)
            shap_error = None
        except Exception as exc:
            contributions, base_margin = None, None
            shap_error = str(exc)
        st.session_state["prediction_result"] = {
            "probability": probability,
            "threshold": artifacts["threshold"],
            "missing_count": missing_count,
            "total": total,
            "contributions": contributions,
            "base_margin": base_margin,
            "shap_error": shap_error,
        }
    except Exception as exc:
        st.session_state.pop("prediction_result", None)
        st.error(f"Prediction failed: {exc}")

result = st.session_state.get("prediction_result")
if result:
    st.subheader("Prediction Results")
    a, b, c = st.columns(3)
    a.metric("Predicted AKI Probability", f"{result['probability'] * 100:.2f}%")
    b.metric("Threshold", f"{result['threshold']:.4f}")
    c.metric("Missing Features", f"{result['missing_count']}/{result['total']}")

    if result["missing_count"] > result["total"] / 2:
        st.warning("More than half of the features are missing; interpret this prediction with caution.")
    if result["probability"] >= result["threshold"]:
        st.warning("High AKI risk")
    else:
        st.info("Low AKI risk")

    st.caption(
        "This is a statistical prediction and does not replace clinical diagnosis."
    )
    st.subheader("Local Explanation Using SHAP")
    if result["contributions"] is not None:
        st.bar_chart(
            result["contributions"].head(20).set_index("Feature")
            ["SHAP value (log-odds)"]
        )
    else:
        st.warning(f"Unable to compute SHAP contributions: {result['shap_error']}")
else:
    st.info("Expand the sidebar sections to edit patient data, then click 'Predict AKI'.")
