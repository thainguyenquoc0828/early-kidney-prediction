#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd


ICU_COLS = [
    "subject_id", "hadm_id", "stay_id", "intime", "outtime",
    "los", "first_careunit", "last_careunit"
]

DIAG_COLS = ["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"]
PATIENT_COLS = ["subject_id", "gender", "anchor_age", "anchor_year"]

EXCL_ICD9_LIST = [
    "40301", "40311", "40391", "40402", "40403", "40412", "40413", "40492", "40493",
    "5855", "5856", "V451", "V4511", "V4512", "V560", "V561", "V562", "V5631",
    "V5632", "V568"
]

EXCL_ICD10_LIST = [
    "I120", "I1311", "I132", "N185", "N186", "Z49", "Z490", "Z4901", "Z4902",
    "Z493", "Z4931", "Z4932", "Z992"
]


def build_cohort(mimic_root, kdigo_path):
    print("Reading data...")

    df_icu = pd.read_csv(mimic_root / "icu/icustays.csv", usecols=ICU_COLS)
    df_diag = pd.read_csv(mimic_root / "hosp/diagnoses_icd.csv", usecols=DIAG_COLS)
    df_pat = pd.read_csv(mimic_root / "hosp/patients.csv", usecols=PATIENT_COLS)
    df_kdigo = pd.read_csv(kdigo_path)

    df_diag["icd_code"] = df_diag["icd_code"].astype(str).str.strip()
    df_icu["intime"] = pd.to_datetime(df_icu["intime"])
    df_icu["outtime"] = pd.to_datetime(df_icu["outtime"])
    df_kdigo["charttime"] = pd.to_datetime(df_kdigo["charttime"])

    # Filter AMI
    is_ami = (
        ((df_diag["icd_version"] == 9) & df_diag["icd_code"].str.startswith("410")) |
        ((df_diag["icd_version"] == 10) & df_diag["icd_code"].str.startswith("I21"))
    )

    df_ami_diag = (
        df_diag.loc[
            is_ami & (df_diag["seq_num"] == 1),
            ["subject_id", "hadm_id"]
        ]
        .drop_duplicates()
    )

    # Merge ICU + patient information
    cohort = df_icu.merge(df_pat, on="subject_id", how="inner")

    # Keep only the first ICU stay within each admission
    cohort = cohort.sort_values(["subject_id", "hadm_id", "intime"])
    cohort = cohort.drop_duplicates(subset="hadm_id", keep="first")

    # Calculate age at ICU admission
    cohort["admission_age"] = (
        cohort["anchor_age"] +
        (cohort["intime"].dt.year - cohort["anchor_year"])
    )

    # Filter age 18-90 and ICU stay >= 12h
    cohort = cohort[
        cohort["admission_age"].between(18, 90) &
        (cohort["los"] >= 0.5)
    ].copy()

    # Keep only AMI patients
    cohort = cohort.merge(df_ami_diag, on=["subject_id", "hadm_id"], how="inner")

    print(f"Number of AMI patients after age and ICU >= 12h filtering: {len(cohort)}")

    # Exclude patients with a history of severe kidney disease
    renal_mask = (
        ((df_diag["icd_version"] == 9) & df_diag["icd_code"].isin(EXCL_ICD9_LIST)) |
        ((df_diag["icd_version"] == 10) & df_diag["icd_code"].isin(EXCL_ICD10_LIST))
    )

    renal_hadm_ids = df_diag.loc[renal_mask, "hadm_id"].unique()
    cohort = cohort[~cohort["hadm_id"].isin(renal_hadm_ids)].copy()

    # Keep only the first admission for each patient
    cohort = cohort.sort_values(["subject_id", "stay_id"])
    cohort = cohort.drop_duplicates(subset="subject_id", keep="first")

    print(f"Number of patients after excluding severe kidney disease: {len(cohort)}")

    # Prediction starts after 12h in ICU
    cohort["prediction_start_time"] = cohort["intime"] + pd.Timedelta(hours=12)

    # Exclude patients who developed AKI within the first 12h
    merged = cohort[["stay_id", "prediction_start_time"]].merge(
        df_kdigo[["stay_id", "charttime", "aki_stage_smoothed"]],
        on="stay_id",
        how="inner"
    )

    pre_aki_mask = (
        (merged["charttime"] <= merged["prediction_start_time"]) &
        (merged["aki_stage_smoothed"] > 0)
    )

    pre_aki_stays = merged.loc[pre_aki_mask, "stay_id"].unique()
    cohort = cohort[~cohort["stay_id"].isin(pre_aki_stays)].copy()

    print(f"Number of patients excluded due to AKI within the first 12h: {len(pre_aki_stays)}")
    print(f"Final number of patients in the cohort: {len(cohort)}")

    return cohort, df_kdigo


def assign_aki_label(cohort, df_kdigo):
    cohort = cohort.copy()
    cohort["prediction_end_time"] = (
        cohort["prediction_start_time"] + pd.Timedelta(days=7)
    )

    # Keep only patients with KDIGO data
    df_kdigo = df_kdigo[
        df_kdigo["stay_id"].isin(cohort["stay_id"])
    ].copy()

    valid_stays = df_kdigo["stay_id"].unique()
    cohort = cohort[cohort["stay_id"].isin(valid_stays)].copy()

    print(f"Number of patients with KDIGO data: {len(cohort)}")

    merged = cohort[
        ["stay_id", "prediction_start_time", "prediction_end_time"]
    ].merge(
        df_kdigo[["stay_id", "charttime", "aki_stage_smoothed"]],
        on="stay_id",
        how="inner"
    )

    # Keep only KDIGO records within the window from 12h to the next 7 days
    in_window = (
        (merged["charttime"] >= merged["prediction_start_time"]) &
        (merged["charttime"] <= merged["prediction_end_time"])
    )

    max_aki = (
        merged.loc[in_window]
        .groupby("stay_id", as_index=False)["aki_stage_smoothed"]
        .max()
        .rename(columns={"aki_stage_smoothed": "max_aki_stage_7d"})
    )

    cohort = cohort.merge(max_aki, on="stay_id", how="left")
    cohort["max_aki_stage_7d"] = cohort["max_aki_stage_7d"].fillna(0)
    cohort["aki_label"] = (cohort["max_aki_stage_7d"] >= 1).astype(int)

    print("-" * 40)
    print("LABEL DISTRIBUTION")
    print(cohort["aki_label"].value_counts().sort_index())
    print(f"AKI rate: {cohort['aki_label'].mean():.2%}")
    print("-" * 40)

    return cohort


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=Path("data/mimiciv")
    )
    parser.add_argument(
        "--kdigo-path",
        type=Path,
        default=Path("data/derived/organfailure/kdigo_stages.csv")
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/raw/AKI_patients.csv")
    )
    args = parser.parse_args()

    cohort, df_kdigo = build_cohort(args.mimic_root, args.kdigo_path)
    result = assign_aki_label(cohort, df_kdigo)

    result = result.drop(
        columns=["anchor_age", "anchor_year", "max_aki_stage_7d"],
        errors="ignore"
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_path, index=False)

    print(f"Saved to: {args.output_path}")
    print(f"Shape: {result.shape}")