#!/usr/bin/env python3

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import optuna

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


ID_COLS = ["subject_id", "hadm_id", "stay_id"]
TARGET = "aki_label"
SEED = 42
N_FOLDS = 5
THRESHOLD = 0.4

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(input_dir, model_name, selected_json=None):
    path = input_dir / f"{model_name}_train.csv"

    native = model_name in {"xgboost", "lightgbm"}
    config = None

    if native:
        config = joblib.load(
            input_dir / f"{model_name}_preprocessor.joblib"
        )
        dtype = {c: "string" for c in config["categorical_cols"]}
    else:
        dtype = None

    df = pd.read_csv(path, dtype=dtype)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    X = df.drop(columns=ID_COLS + [TARGET])
    y = df[TARGET].astype(int)

    if native:
        for col in config["categorical_cols"]:
            levels = [str(v) for v in config["category_levels"][col]]
            X[col] = pd.Categorical(
                X[col].astype("string"),
                categories=levels,
            )

        for col in config["numerical_cols"]:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    else:
        if selected_json is not None:
            with open(selected_json, encoding="utf-8") as f:
                features = json.load(f)["final_selected_features"]
            X = X[features].copy()

        if X.isna().any().any():
            raise ValueError(f"{model_name}: Missing values detected")
    return X, y


def create_model(name, trial):
    if name == "lr":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100, log=True),
            class_weight=trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
            solver="lbfgs",
            max_iter=3000,
            random_state=SEED,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators", 200, 600, step=100
            ),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int(
                "min_samples_split", 2, 20
            ),
            min_samples_leaf=trial.suggest_int(
                "min_samples_leaf", 1, 10
            ),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            class_weight=trial.suggest_categorical(
                "class_weight", [None, "balanced"]
            ),
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators", 200, 600, step=100
            ),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            subsample=trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            min_child_weight=trial.suggest_int(
                "min_child_weight", 1, 10
            ),
            reg_lambda=trial.suggest_float(
                "reg_lambda", 1e-3, 10, log=True
            ),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            enable_categorical=True,
            random_state=SEED,
            n_jobs=-1,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators", 200, 600, step=100
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_child_samples=trial.suggest_int(
                "min_child_samples", 10, 50
            ),
            subsample=trial.suggest_float(
                "subsample", 0.6, 1.0
            ),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            reg_lambda=trial.suggest_float(
                "reg_lambda", 1e-3, 10, log=True
            ),
            random_state=SEED,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def calculate_metrics(y_true, y_prob):
    y_pred = (y_prob >= THRESHOLD).astype(int)

    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Brier Score": brier_score_loss(y_true, y_prob),
        "PPV": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Sensitivity": recall_score(
            y_true, y_pred, zero_division=0
        ),
    }


def cross_validate(model, X, y, cv):
    results = []

    for train_idx, valid_idx in cv.split(X, y):
        fitted = clone(model)

        fitted.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        y_prob = fitted.predict_proba(
            X.iloc[valid_idx]
        )[:, 1]

        metrics = calculate_metrics(
            y.iloc[valid_idx],
            y_prob,
        )

        results.append(metrics)

    return pd.DataFrame(results)


def tune_model(name, X, y, cv, n_trials):
    def objective(trial):
        model = create_model(name, trial)
        scores = cross_validate(model, X, y, cv)
        return scores["AUROC"].mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    return study


def train_models(input_dir, output_dir, n_trials, selected_json):
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    model_names = ["lr", "rf", "xgboost", "lightgbm"]

    summary = []
    best_params = {}

    for name in model_names:
        print(f"\n{'=' * 50}")
        print(f"Training {name.upper()}")
        print(f"{'=' * 50}")

        X, y = load_data(
            input_dir, name, selected_json
        )

        if y.value_counts().min() < N_FOLDS:
            raise ValueError(
                f"{name}: each class needs at least {N_FOLDS} samples"
            )

        # 1. Hyperparameter tuning
        study = tune_model(
            name, X, y, cv, n_trials
        )

        best_params[name] = study.best_params

        print(f"Best AUROC: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        # 2. Create model with best hyperparameters
        # FixedTrial reuses the same search space.
        best_trial = optuna.trial.FixedTrial(
            study.best_params
        )

        best_model = create_model(name, best_trial)

        # 3. Evaluate best model with 5-fold CV
        cv_results = cross_validate(
            best_model, X, y, cv
        )

        row = {
            "Model": name.upper(),
        }

        for metric in [
            "AUROC",
            "AUPRC",
            "Brier Score",
            "PPV",
            "Sensitivity",
        ]:
            mean = cv_results[metric].mean()
            std = cv_results[metric].std(ddof=1)

            row[metric] = f"{mean:.4f} ± {std:.4f}"

        summary.append(row)

        # 4. Fit final model on full training data
        best_model.fit(X, y)

        # 5. Save model and feature names
        joblib.dump(
            best_model,
            output_dir / f"{name}_model.joblib",
        )

        with open(
            output_dir / f"{name}_features.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                X.columns.tolist(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved {name}_model.joblib")

    # Save best hyperparameters
    with open(
        output_dir / "best_params.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            best_params,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save one CV summary table
    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        output_dir / "cv_summary.csv",
        index=False,
    )

    print("\n=== 5-FOLD CV SUMMARY ===")
    print(summary_df.to_string(index=False))

    print(f"\nSaved all results to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/preprocessed"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--selected-features",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    train_models(
        args.input_dir,
        args.output_dir,
        args.n_trials,
        args.selected_features,
    )
