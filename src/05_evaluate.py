
#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve


TARGET = "aki_label"
ID_COLS = ["subject_id", "hadm_id", "stay_id"]
SEED = 42


def load_external_data(input_dir, models_dir):
    config = joblib.load(
        input_dir / "xgboost_preprocessor.joblib"
    )

    dtype = {
        col: "string"
        for col in config["categorical_cols"]
    }

    df = pd.read_csv(
        input_dir / "xgboost_test_eicu.csv",
        dtype=dtype
    )
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    y = df[TARGET].astype(int)
    X = df.drop(columns=ID_COLS + [TARGET])

    # Restore categorical dtypes
    for col in config["categorical_cols"]:
        levels = [
            str(v)
            for v in config["category_levels"][col]
        ]

        X[col] = pd.Categorical(
            X[col].astype("string"),
            categories=levels
        )

    for col in config["numerical_cols"]:
        X[col] = pd.to_numeric(
            X[col], errors="coerce"
        )

    # Exact feature order used during training
    with open(
        models_dir / "xgboost_features.json",
        encoding="utf-8"
    ) as f:
        features = json.load(f)

    missing = [c for c in features if c not in X.columns]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    return X[features].copy(), y


def calculate_metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)

    return {
        "Model": "XGBoost",
        "AUROC": roc_auc_score(y, prob),
        "AUPRC": average_precision_score(y, prob),
        "Brier Score": brier_score_loss(y, prob),
        "PPV": precision_score(
            y, pred, zero_division=0
        ),
        "Sensitivity": recall_score(
            y, pred, zero_division=0
        ),
    }


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# 1. ROC Curve
def plot_roc(y, prob, output_dir):
    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr, tpr,
        linewidth=2,
        label=f"XGBoost (AUC = {auc:.3f})"
    )
    ax.plot(
        [0, 1], [0, 1],
        "--", color="gray",
        label="Random"
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve - External eICU"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    save_figure(fig, output_dir / "roc_curve.png")


# 2. Confusion Matrix
def plot_confusion(y, prob, threshold, output_dir):
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No AKI", "AKI"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(
        f"Confusion Matrix - External eICU\n"
        f"Threshold = {threshold:.3f}"
    )

    save_figure(fig, output_dir / "confusion_matrix.png")


# 3. Calibration Plot
def plot_calibration(y, prob, output_dir):
    fraction_pos, mean_pred = calibration_curve(
        y, prob,
        n_bins=10,
        strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        [0, 1], [0, 1],
        "--", color="gray",
        label="Perfect calibration"
    )

    ax.plot(
        mean_pred,
        fraction_pos,
        marker="o",
        linewidth=2,
        label="XGBoost"
    )

    ax.set(
        xlabel="Mean Predicted Probability",
        ylabel="Observed AKI Frequency",
        title="Calibration Plot - External eICU"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.2)

    save_figure(fig, output_dir / "calibration_plot.png")


# 4. Decision Curve Analysis
def plot_decision_curve(y, prob, output_dir):
    thresholds = np.linspace(0.01, 0.99, 99)

    n = len(y)
    prevalence = np.mean(y)

    nb_model = []
    nb_all = []

    for pt in thresholds:
        pred = prob >= pt

        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))

        net_benefit = (
            tp / n - fp / n * pt / (1 - pt)
        )

        treat_all = (
            prevalence
            - (1 - prevalence) * pt / (1 - pt)
        )

        nb_model.append(net_benefit)
        nb_all.append(treat_all)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        thresholds, nb_model,
        linewidth=2,
        label="XGBoost"
    )

    ax.plot(
        thresholds, nb_all,
        linestyle="--",
        label="Treat All"
    )

    ax.axhline(
        0,
        color="gray",
        linestyle=":",
        label="Treat None"
    )

    ax.set(
        xlabel="Threshold Probability",
        ylabel="Net Benefit",
        title="Decision Curve Analysis - External eICU"
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 0.5)
    ax.legend()
    ax.grid(alpha=0.2)

    save_figure(fig, output_dir / "decision_curve.png")


# 5 & 6. SHAP Mean + Beeswarm
def plot_shap(model, X, output_dir, max_samples=1000, top_n=20):
    # Sample external patients for SHAP visualization
    if len(X) > max_samples:
        X_sample = X.sample(
            n=max_samples,
            random_state=SEED
        )
    else:
        X_sample = X.copy()

    # Native XGBoost SHAP contributions
    dmatrix = xgb.DMatrix(
        X_sample,
        enable_categorical=True
    )

    contributions = model.get_booster().predict(
        dmatrix,
        pred_contribs=True
    )

    # Last column is the expected value (bias)
    shap_values = contributions[:, :-1]
    base_values = contributions[:, -1]

    if shap_values.shape[1] != X_sample.shape[1]:
        raise ValueError("SHAP feature dimension mismatch")

    # Prepare data for beeswarm:
    # categorical features are represented by category codes
    X_display = X_sample.copy()

    for col in X_display.select_dtypes(include="category").columns:
        codes = X_display[col].cat.codes.astype(float)
        X_display[col] = codes.replace(-1, np.nan)

    # Create SHAP Explanation
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_display.to_numpy(dtype=float),
        feature_names=X_display.columns.tolist()
    )

    # Mean absolute SHAP importance
    importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values(
        "mean_abs_shap",
        ascending=False
    )

    importance.to_csv(
        output_dir / "shap_feature_importance.csv",
        index=False
    )

    # 5. SHAP mean absolute values plot
    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        top["feature"],
        top["mean_abs_shap"],
        color="steelblue"
    )

    ax.set(
        xlabel="Mean |SHAP value| (log-odds)",
        title="SHAP Feature Importance - External eICU"
    )
    ax.grid(axis="x", alpha=0.2)

    save_figure(fig, output_dir / "shap_mean.png")

    # 6. SHAP beeswarm plot
    shap.plots.beeswarm(
        explanation,
        max_display=top_n,
        show=False
    )

    fig = plt.gcf()
    fig.set_size_inches(10, 8)

    plt.title("SHAP Beeswarm - External eICU")
    plt.tight_layout()

    fig.savefig(
        output_dir / "shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(
        f"SHAP calculated on {len(X_sample)} external patients"
    )


def main(input_dir, models_dir, figures_dir, shap_samples):
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load optimized threshold
    with open(
        models_dir / "best_params.json",
        encoding="utf-8"
    ) as f:
        best_params = json.load(f)

    threshold = float(
        best_params["xgboost"]["threshold"]
    )

    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError(f"Invalid threshold: {threshold}")

    # Load trained model and external data
    model = joblib.load(
        models_dir / "xgboost_model.joblib"
    )

    X, y = load_external_data(
        input_dir, models_dir
    )

    if y.nunique() != 2:
        raise ValueError(
            "External test must contain both AKI classes"
        )

    prob = model.predict_proba(X)[:, 1]

    # Calculate 5 metrics
    metrics = calculate_metrics(y, prob, threshold)
    results = pd.DataFrame([metrics])

    print("\n=== XGBOOST EXTERNAL TEST RESULTS ===")
    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    results.to_csv(
        figures_dir / "xgboost_external_metrics.csv",
        index=False
    )

    # Evaluation figures
    plot_roc(y, prob, figures_dir)
    plot_confusion(y, prob, threshold, figures_dir)
    plot_calibration(y, prob, figures_dir)
    plot_decision_curve(y, prob, figures_dir)

    # SHAP plots
    plot_shap(
        model,
        X,
        figures_dir,
        max_samples=shap_samples,
        top_n=20
    )

    print(f"\nAll results saved to: {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    root = Path(".")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "data/preprocessed"
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
        default=root / "models"
    )

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=root / "figures"
    )

    parser.add_argument(
        "--shap-samples",
        type=int,
        default=1000,
        help="Maximum number of external patients for SHAP plots"
    )

    args = parser.parse_args()

    main(
        args.input_dir,
        args.models_dir,
        args.figures_dir,
        args.shap_samples
    )