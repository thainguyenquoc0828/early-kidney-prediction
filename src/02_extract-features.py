#!/usr/bin/env python3

from pathlib import Path
import argparse
import pandas as pd


def extract_demographic_features(aki_df, derived_root):
    detail = pd.read_csv(
        derived_root / "demographics/icustay_detail.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "race", "first_hosp_stay"]
    )
    aki_df = aki_df.merge(detail, on=["subject_id", "hadm_id", "stay_id"], how="left")

    weight = pd.read_csv(
        derived_root / "firstday/first_day_weight.csv",
        usecols=["subject_id", "stay_id", "weight_admit"]
    ).rename(columns={"weight_admit": "weight"})
    aki_df = aki_df.merge(weight, on=["subject_id", "stay_id"], how="left")

    height = pd.read_csv(
        derived_root / "firstday/first_day_height.csv",
        usecols=["subject_id", "stay_id", "height"]
    )
    aki_df = aki_df.merge(height, on=["subject_id", "stay_id"], how="left")

    return aki_df


def extract_vital_features(aki_df, derived_root):
    vital = pd.read_csv(
        derived_root / "measurement/vitalsign.csv",
        usecols=[
            "stay_id", "charttime", "heart_rate", "sbp", "dbp", "mbp",
            "sbp_ni", "dbp_ni", "mbp_ni", "resp_rate", "temperature", "glucose"
        ]
    )
    vital["charttime"] = pd.to_datetime(vital["charttime"])

    x = aki_df[["stay_id", "intime"]].merge(vital, on="stay_id", how="inner")
    x = x[
        (x["charttime"] >= x["intime"]) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12))
    ].copy()

    x["sys_bp"] = x["sbp"].combine_first(x["sbp_ni"])
    x["dias_bp"] = x["dbp"].combine_first(x["dbp_ni"])
    x["mean_bp"] = x["mbp"].combine_first(x["mbp_ni"])

    feat = x.groupby("stay_id").agg({
        "heart_rate": "max",
        "sys_bp": "min",
        "mean_bp": "min",
        "dias_bp": "min",
        "resp_rate": "max",
        "temperature": "max",
        "glucose": "max"
    }).reset_index().rename(columns={
        "heart_rate": "heart_rate_max",
        "sys_bp": "sys_bp_min",
        "mean_bp": "mean_bp_min",
        "dias_bp": "dias_bp_min",
        "resp_rate": "resp_rate_max",
        "temperature": "temperature_max",
        "glucose": "glucose_vitalsign_max"
    })

    return aki_df.merge(feat, on="stay_id", how="left")


def extract_lab_features(aki_df, derived_root):
    # Chemistry
    chem = pd.read_csv(
        derived_root / "measurement/chemistry.csv",
        usecols=[
            "hadm_id", "charttime", "creatinine", "bun", "bicarbonate",
            "potassium", "aniongap", "albumin", "glucose"
        ]
    )
    chem["charttime"] = pd.to_datetime(chem["charttime"])

    x = aki_df[["stay_id", "hadm_id", "intime"]].merge(chem, on="hadm_id", how="inner")
    x = x[
        (x["charttime"] >= x["intime"] - pd.Timedelta(hours=6)) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12))
    ]

    feat = x.groupby("stay_id").agg({
        "creatinine": "min",
        "bun": "max",
        "bicarbonate": "min",
        "potassium": "max",
        "aniongap": "max",
        "albumin": "median",
        "glucose": "max"
    }).reset_index().rename(columns={
        "creatinine": "creatinine_min",
        "bun": "bun_max",
        "bicarbonate": "bicarbonate_min",
        "potassium": "potassium_chemistry_max",
        "aniongap": "aniongap_max",
        "albumin": "albumin_median",
        "glucose": "glucose_chemistry_max"
    })
    aki_df = aki_df.merge(feat, on="stay_id", how="left")

    # Complete blood count
    cbc = pd.read_csv(
        derived_root / "measurement/complete_blood_count.csv",
        usecols=["hadm_id", "charttime", "hemoglobin", "hematocrit", "wbc", "platelet", "rdw"]
    )
    cbc["charttime"] = pd.to_datetime(cbc["charttime"])

    x = aki_df[["stay_id", "hadm_id", "intime"]].merge(cbc, on="hadm_id", how="inner")
    x = x[
        (x["charttime"] >= x["intime"] - pd.Timedelta(hours=6)) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12))
    ]

    feat = x.groupby("stay_id").agg({
        "hemoglobin": "min",
        "hematocrit": "min",
        "wbc": "max",
        "platelet": "median",
        "rdw": "median"
    }).reset_index().rename(columns={
        "hemoglobin": "hemoglobin_min",
        "hematocrit": "hematocrit_min",
        "wbc": "wbc_max",
        "platelet": "platelet_median",
        "rdw": "rdw_median"
    })
    aki_df = aki_df.merge(feat, on="stay_id", how="left")

    # Cardiac markers
    cardiac = pd.read_csv(
        derived_root / "measurement/cardiac_marker.csv",
        usecols=["hadm_id", "charttime", "troponin_t", "ck_mb", "ntprobnp"]
    )
    cardiac["charttime"] = pd.to_datetime(cardiac["charttime"])

    x = aki_df[["stay_id", "hadm_id", "intime"]].merge(cardiac, on="hadm_id", how="inner")
    x = x[
        (x["charttime"] >= x["intime"] - pd.Timedelta(hours=6)) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12))
    ]

    feat = x.groupby("stay_id").agg({
        "troponin_t": "max",
        "ck_mb": "max",
        "ntprobnp": "max"
    }).reset_index().rename(columns={
        "troponin_t": "troponin_t_max",
        "ck_mb": "ck_mb_max",
        "ntprobnp": "ntprobnp_max"
    })
    aki_df = aki_df.merge(feat, on="stay_id", how="left")

    # Coagulation
    coag = pd.read_csv(
        derived_root / "measurement/coagulation.csv",
        usecols=["hadm_id", "charttime", "inr", "pt", "ptt", "fibrinogen"]
    )
    coag["charttime"] = pd.to_datetime(coag["charttime"])

    x = aki_df[["stay_id", "hadm_id", "intime"]].merge(coag, on="hadm_id", how="inner")
    x = x[
        (x["charttime"] >= x["intime"] - pd.Timedelta(hours=6)) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12))
    ]

    feat = x.groupby("stay_id").agg({
        "inr": "median",
        "pt": "median",
        "ptt": "median",
        "fibrinogen": "median"
    }).reset_index().rename(columns={
        "inr": "inr_median",
        "pt": "pt_median",
        "ptt": "ptt_median",
        "fibrinogen": "fibrinogen_median"
    })
    aki_df = aki_df.merge(feat, on="stay_id", how="left")

    # Enzymes
    enzyme = pd.read_csv(
        derived_root / "measurement/enzyme.csv",
        usecols=[
            "hadm_id", "charttime", "ck_cpk", "ld_ldh", "alt", "ast",
            "bilirubin_total", "alp", "amylase"
        ]
    )
    enzyme["charttime"] = pd.to_datetime(enzyme["charttime"])

    x = aki_df[["stay_id", "hadm_id", "intime"]].merge(enzyme, on="hadm_id", how="inner")
    x = x[
        (x["charttime"] >= x["intime"] - pd.Timedelta(hours=6)) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12))
    ]

    feat = x.groupby("stay_id").agg({
        "ck_cpk": "max",
        "ld_ldh": "max",
        "alt": "max",
        "ast": "max",
        "bilirubin_total": "max",
        "alp": "median",
        "amylase": "median"
    }).reset_index().rename(columns={
        "ck_cpk": "ck_cpk_max",
        "ld_ldh": "ld_ldh_max",
        "alt": "alt_max",
        "ast": "ast_max",
        "bilirubin_total": "bilirubin_total_max",
        "alp": "alp_median",
        "amylase": "amylase_median"
    })
    aki_df = aki_df.merge(feat, on="stay_id", how="left")

    return aki_df


def extract_renal_features(aki_df, derived_root):
    uo = pd.read_csv(
        derived_root / "measurement/urine_output.csv",
        usecols=["stay_id", "charttime", "urineoutput"]
    )
    uo["charttime"] = pd.to_datetime(uo["charttime"])

    x = aki_df[["stay_id", "intime"]].merge(uo, on="stay_id", how="inner")
    x = x[
        (x["charttime"] >= x["intime"]) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12)) &
        (x["urineoutput"] >= 0)
    ]

    feat = (
        x.groupby("stay_id", as_index=False)["urineoutput"]
        .sum()
        .rename(columns={"urineoutput": "urineoutput_12h_sum"})
    )

    return aki_df.merge(feat, on="stay_id", how="left")


def extract_respiratory_features(aki_df, derived_root):
    # Ventilator settings
    vent = pd.read_csv(
        derived_root / "measurement/ventilator_setting.csv",
        usecols=["stay_id", "charttime", "peep", "fio2", "plateau_pressure", "tidal_volume_observed"]
    )
    vent["charttime"] = pd.to_datetime(vent["charttime"])

    x = aki_df[["stay_id", "intime"]].merge(vent, on="stay_id", how="inner")
    x = x[
        (x["charttime"] >= x["intime"]) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12))
    ]

    feat = x.groupby("stay_id").agg({
        "peep": "max",
        "fio2": "max",
        "plateau_pressure": "max",
        "tidal_volume_observed": "max"
    }).reset_index().rename(columns={
        "peep": "peep_max",
        "fio2": "fio2_max",
        "plateau_pressure": "plateau_pressure_max",
        "tidal_volume_observed": "tidal_volume_observed_max"
    })
    aki_df = aki_df.merge(feat, on="stay_id", how="left")

    # Keep only non-invasive ventilation because invasive_vent_flag is not used in the final output
    ventilation = pd.read_csv(
        derived_root / "treatment/ventilation.csv",
        usecols=["stay_id", "starttime", "endtime", "ventilation_status"]
    )
    ventilation["starttime"] = pd.to_datetime(ventilation["starttime"])
    ventilation["endtime"] = pd.to_datetime(ventilation["endtime"])

    x = aki_df[["stay_id", "intime"]].merge(ventilation, on="stay_id", how="inner")
    x = x[
        (x["starttime"] <= x["intime"] + pd.Timedelta(hours=12)) &
        (x["endtime"] >= x["intime"])
    ].copy()

    x["non_invasive_vent_flag"] = (x["ventilation_status"] == "NonInvasiveVent").astype(int)
    feat = x.groupby("stay_id", as_index=False)["non_invasive_vent_flag"].max()

    aki_df = aki_df.merge(feat, on="stay_id", how="left")
    aki_df["non_invasive_vent_flag"] = aki_df["non_invasive_vent_flag"].fillna(0).astype(int)

    return aki_df


def extract_medication_features(aki_df, derived_root):
    # Keep only aspirin because the vasopressor, ACEI, antibiotic, and invasive line features
    # are not used in the final output.
    nsaid = pd.read_csv(
        derived_root / "medication/nsaid.csv",
        usecols=["hadm_id", "starttime", "stoptime", "nsaid"]
    )
    nsaid["starttime"] = pd.to_datetime(nsaid["starttime"])
    nsaid["stoptime"] = pd.to_datetime(nsaid["stoptime"])

    x = aki_df[["stay_id", "hadm_id", "intime"]].merge(nsaid, on="hadm_id", how="inner")
    x = x[
        (x["starttime"] <= x["intime"] + pd.Timedelta(hours=12)) &
        (x["stoptime"] >= x["intime"])
    ].copy()

    x["use_aspirin"] = x["nsaid"].astype(str).str.lower().str.contains("aspirin").astype(int)
    feat = x.groupby("stay_id", as_index=False)["use_aspirin"].max()

    aki_df = aki_df.merge(feat, on="stay_id", how="left")
    aki_df["use_aspirin"] = aki_df["use_aspirin"].fillna(0).astype(int)

    return aki_df


def extract_inflammation_features(aki_df, derived_root):
    # Keep only bands_max; nlr_max and infection flags are not used in the final output.
    diff = pd.read_csv(
        derived_root / "measurement/blood_differential.csv",
        usecols=["hadm_id", "charttime", "bands"]
    )
    diff["charttime"] = pd.to_datetime(diff["charttime"])

    x = aki_df[["stay_id", "hadm_id", "intime"]].merge(diff, on="hadm_id", how="inner")
    x = x[
        (x["charttime"] >= x["intime"] - pd.Timedelta(hours=6)) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12))
    ]

    feat = x.groupby("stay_id", as_index=False)["bands"].max()
    feat = feat.rename(columns={"bands": "bands_max"})

    return aki_df.merge(feat, on="stay_id", how="left")


def extract_rhythm_features(aki_df, derived_root):
    rhythm = pd.read_csv(
        derived_root / "measurement/rhythm.csv",
        usecols=["subject_id", "charttime", "heart_rhythm"]
    )
    rhythm["charttime"] = pd.to_datetime(rhythm["charttime"])

    x = aki_df[["subject_id", "stay_id", "intime"]].merge(
        rhythm,
        on="subject_id",
        how="inner"
    )
    x = x[
        (x["charttime"] >= x["intime"] - pd.Timedelta(hours=2)) &
        (x["charttime"] <= x["intime"] + pd.Timedelta(hours=12))
    ].copy()

    r = x["heart_rhythm"].str.lower().fillna("")
    x["is_afib"] = r.str.contains("af |atrial fib|a flut|atrial flutter", regex=True).astype(int)
    x["is_high_grade_block"] = r.str.contains(
        r"3rd av|complete heart block|2nd av m2|mobitz 2",
        regex=True
    ).astype(int)

    feat = x.groupby("stay_id").agg({
        "is_afib": "max",
        "is_high_grade_block": "max"
    }).reset_index()

    aki_df = aki_df.merge(feat, on="stay_id", how="left")
    aki_df[["is_afib", "is_high_grade_block"]] = (
        aki_df[["is_afib", "is_high_grade_block"]].fillna(0).astype(int)
    )

    return aki_df


def extract_features(aki_path, derived_root):
    aki_df = pd.read_csv(aki_path)
    aki_df = aki_df.loc[:, ~aki_df.columns.str.startswith("Unnamed")]
    aki_df["intime"] = pd.to_datetime(aki_df["intime"])

    aki_df = extract_demographic_features(aki_df, derived_root)
    aki_df = extract_vital_features(aki_df, derived_root)
    aki_df = extract_lab_features(aki_df, derived_root)
    aki_df = extract_renal_features(aki_df, derived_root)
    aki_df = extract_respiratory_features(aki_df, derived_root)
    aki_df = extract_medication_features(aki_df, derived_root)
    aki_df = extract_inflammation_features(aki_df, derived_root)
    aki_df = extract_rhythm_features(aki_df, derived_root)

    # Only drop columns already present in AKI_patients.csv that are not used as features.
    drop_cols = [
        "los",
        "intime",
        "outtime",
        "prediction_start_time",
        "prediction_end_time",
        "last_careunit",
        "race"
    ]
    aki_df = aki_df.drop(columns=drop_cols, errors="ignore")

    print(f"Final shape: {aki_df.shape}")
    print(f'AKI prevalence: {aki_df["aki_label"].mean():.2%}')

    return aki_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract grouped first-12h features for the AMI-AKI cohort."
    )
    parser.add_argument(
        "--aki-path",
        type=Path,
        default=Path("data/raw/AKI_patients.csv")
    )
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=Path("mimic-derived")
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/raw/AKI_patients_with_features.csv")
    )
    args = parser.parse_args()

    result = extract_features(args.aki_path, args.derived_root)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_path, index=False)

    print(f"Saved to: {args.output_path}")
