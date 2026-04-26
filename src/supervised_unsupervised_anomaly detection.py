# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:26:01 2026

@author: edwin
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score



# LOAD LOCAL DATASETS



med_file_path = Path(
    r"C:\Users\edwin\Desktop\Project-Data Science Finals\cms\med.csv"
)

oig_file_path = Path(
    r"C:\Users\edwin\Desktop\Project-Data Science Finals\cms\oig.csv"
)

cms_df = pd.read_csv(med_file_path, low_memory=False)
leie_df = pd.read_csv(oig_file_path, low_memory=False)

print("\nRaw CMS shape:", cms_df.shape)
print("Raw OIG shape:", leie_df.shape)



# STANDARDIZE CMS COLUMNS

cms_df.columns = (
    cms_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

rename_map = {
    "rndrng_npi": "npi",
    "rndrng_prvdr_last_org_name": "last_name",
    "rndrng_prvdr_first_name": "first_name",
    "rndrng_prvdr_city": "city",
    "rndrng_prvdr_state_abrvtn": "state",
    "rndrng_prvdr_type": "provider_type",
    "place_of_srvc": "place_of_service",
    "hcpcs_cd": "hcpcs",
    "hcpcs_desc": "hcpcs_description",
    "tot_benes": "total_beneficiaries",
    "tot_srvcs": "total_services",
    "avg_sbmtd_chrg": "avg_submitted_charge",
    "avg_mdcr_alowd_amt": "avg_medicare_allowed",
    "avg_mdcr_pymt_amt": "avg_medicare_payment",
    "avg_mdcr_stdzd_amt": "avg_medicare_standardized"
}

cms_df = cms_df.rename(columns={k: v for k, v in rename_map.items() if k in cms_df.columns})



# SELECT RELEVANT CMS COLUMNS


needed_cols = [
    "npi", "last_name", "first_name", "city", "state", "provider_type",
    "place_of_service", "hcpcs", "hcpcs_description",
    "total_beneficiaries", "total_services",
    "avg_submitted_charge", "avg_medicare_allowed",
    "avg_medicare_payment", "avg_medicare_standardized"
]

cms_df = cms_df[[col for col in needed_cols if col in cms_df.columns]].copy()

required_cols = [
    "npi", "state", "hcpcs", "total_services", "total_beneficiaries",
    "avg_submitted_charge", "avg_medicare_allowed", "avg_medicare_payment"
]

missing_required = [col for col in required_cols if col not in cms_df.columns]

if missing_required:
    raise KeyError(f"Missing required CMS columns: {missing_required}")



# CLEAN NUMERIC VARIABLES


numeric_cols = [
    "total_beneficiaries",
    "total_services",
    "avg_submitted_charge",
    "avg_medicare_allowed",
    "avg_medicare_payment",
    "avg_medicare_standardized"
]

for col in numeric_cols:
    if col in cms_df.columns:
        cms_df[col] = (
            cms_df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )
        cms_df[col] = pd.to_numeric(cms_df[col], errors="coerce")

cms_df["npi"] = pd.to_numeric(cms_df["npi"], errors="coerce").astype("Int64")

cms_df = cms_df.dropna(subset=["npi", "state", "hcpcs"])
cms_df = cms_df[cms_df["total_services"] > 0]
cms_df.replace([np.inf, -np.inf], np.nan, inplace=True)

for col in ["state", "provider_type", "hcpcs"]:
    if col in cms_df.columns:
        cms_df[col] = cms_df[col].astype("category")

print("\nCleaned CMS shape:", cms_df.shape)



#  BASIC BILLING FEATURES

cms_df["services_per_beneficiary"] = (
    cms_df["total_services"] / cms_df["total_beneficiaries"].replace(0, np.nan)
)

cms_df["charge_to_payment_ratio"] = (
    cms_df["avg_submitted_charge"] / cms_df["avg_medicare_payment"].replace(0, np.nan)
)

cms_df["charge_to_allowed_ratio"] = (
    cms_df["avg_submitted_charge"] / cms_df["avg_medicare_allowed"].replace(0, np.nan)
)

cms_df["payment_to_allowed_ratio"] = (
    cms_df["avg_medicare_payment"] / cms_df["avg_medicare_allowed"].replace(0, np.nan)
)

cms_df["estimated_total_payment"] = (
    cms_df["total_services"] * cms_df["avg_medicare_payment"]
)

cms_df["estimated_total_charge"] = (
    cms_df["total_services"] * cms_df["avg_submitted_charge"]
)

cms_df.replace([np.inf, -np.inf], np.nan, inplace=True)



# H1: BILLING / DISBURSEMENT ANOMALIES


cms_df["payment_per_beneficiary"] = (
    cms_df["estimated_total_payment"] / cms_df["total_beneficiaries"].replace(0, np.nan)
)

cms_df["charge_per_beneficiary"] = (
    cms_df["estimated_total_charge"] / cms_df["total_beneficiaries"].replace(0, np.nan)
)

hcpcs_billing_bench = (
    cms_df.groupby("hcpcs", observed=True)
    .agg(
        hcpcs_mean_charge_to_payment=("charge_to_payment_ratio", "mean"),
        hcpcs_std_charge_to_payment=("charge_to_payment_ratio", "std"),
        hcpcs_mean_payment_per_bene=("payment_per_beneficiary", "mean"),
        hcpcs_std_payment_per_bene=("payment_per_beneficiary", "std")
    )
    .reset_index()
)

cms_df = cms_df.merge(hcpcs_billing_bench, on="hcpcs", how="left")

cms_df["billing_disbursement_z"] = (
    (cms_df["charge_to_payment_ratio"] - cms_df["hcpcs_mean_charge_to_payment"])
    / cms_df["hcpcs_std_charge_to_payment"].replace(0, np.nan)
)

cms_df["payment_per_bene_z"] = (
    (cms_df["payment_per_beneficiary"] - cms_df["hcpcs_mean_payment_per_bene"])
    / cms_df["hcpcs_std_payment_per_bene"].replace(0, np.nan)
)

cms_df["h1_unusual_billing_flag"] = (
    (cms_df["billing_disbursement_z"].abs() >= 3)
    | (cms_df["payment_per_bene_z"].abs() >= 3)
).astype(int)



#  H2: GEOGRAPHIC BILLING VOLUME SPIKES


geo_hcpcs_bench = (
    cms_df.groupby(["state", "hcpcs"], observed=True)
    .agg(
        state_hcpcs_mean_services=("total_services", "mean"),
        state_hcpcs_std_services=("total_services", "std"),
        state_hcpcs_mean_benes=("total_beneficiaries", "mean"),
        state_hcpcs_std_benes=("total_beneficiaries", "std")
    )
    .reset_index()
)

cms_df = cms_df.merge(geo_hcpcs_bench, on=["state", "hcpcs"], how="left")

cms_df["geo_service_volume_z"] = (
    (cms_df["total_services"] - cms_df["state_hcpcs_mean_services"])
    / cms_df["state_hcpcs_std_services"].replace(0, np.nan)
)

cms_df["geo_beneficiary_volume_z"] = (
    (cms_df["total_beneficiaries"] - cms_df["state_hcpcs_mean_benes"])
    / cms_df["state_hcpcs_std_benes"].replace(0, np.nan)
)

cms_df["h2_geo_volume_spike_flag"] = (
    (cms_df["geo_service_volume_z"] >= 3)
    | (cms_df["geo_beneficiary_volume_z"] >= 3)
).astype(int)



# H3: ABOVE-AVERAGE HCPCS COST ANOMALIES

hcpcs_cost_bench = (
    cms_df.groupby("hcpcs", observed=True)
    .agg(
        hcpcs_mean_payment=("avg_medicare_payment", "mean"),
        hcpcs_std_payment=("avg_medicare_payment", "std"),
        hcpcs_mean_charge=("avg_submitted_charge", "mean"),
        hcpcs_std_charge=("avg_submitted_charge", "std")
    )
    .reset_index()
)

cms_df = cms_df.merge(hcpcs_cost_bench, on="hcpcs", how="left")

cms_df["hcpcs_payment_cost_z"] = (
    (cms_df["avg_medicare_payment"] - cms_df["hcpcs_mean_payment"])
    / cms_df["hcpcs_std_payment"].replace(0, np.nan)
)

cms_df["hcpcs_submitted_charge_z"] = (
    (cms_df["avg_submitted_charge"] - cms_df["hcpcs_mean_charge"])
    / cms_df["hcpcs_std_charge"].replace(0, np.nan)
)

cms_df["h3_above_avg_hcpcs_cost_flag"] = (
    (cms_df["hcpcs_payment_cost_z"] >= 3)
    | (cms_df["hcpcs_submitted_charge_z"] >= 3)
).astype(int)



#  PROVIDER-LEVEL AGGREGATION


provider_id_cols = [
    "npi", "last_name", "first_name", "city", "state", "provider_type"
]

provider_id_cols = [col for col in provider_id_cols if col in cms_df.columns]

agg_dict = {
    "total_services": ["sum", "mean", "max"],
    "total_beneficiaries": ["sum", "mean", "max"],
    "avg_submitted_charge": ["mean", "max"],
    "avg_medicare_allowed": ["mean", "max"],
    "avg_medicare_payment": ["mean", "max"],
    "estimated_total_payment": ["sum", "mean", "max"],
    "estimated_total_charge": ["sum", "mean", "max"],
    "services_per_beneficiary": ["mean", "max"],
    "charge_to_payment_ratio": ["mean", "max"],
    "charge_to_allowed_ratio": ["mean", "max"],
    "payment_to_allowed_ratio": ["mean", "max"],
    "billing_disbursement_z": ["mean", "max"],
    "payment_per_bene_z": ["mean", "max"],
    "geo_service_volume_z": ["mean", "max"],
    "geo_beneficiary_volume_z": ["mean", "max"],
    "hcpcs_payment_cost_z": ["mean", "max"],
    "hcpcs_submitted_charge_z": ["mean", "max"],
    "h1_unusual_billing_flag": ["sum", "mean"],
    "h2_geo_volume_spike_flag": ["sum", "mean"],
    "h3_above_avg_hcpcs_cost_flag": ["sum", "mean"],
    "hcpcs": pd.Series.nunique
}

provider_df = cms_df.groupby(provider_id_cols, observed=True).agg(agg_dict)

provider_df.columns = [
    "_".join(col).replace("nunique", "unique_count")
    for col in provider_df.columns
]

provider_df = provider_df.reset_index()

provider_df = provider_df.rename(columns={
    "hcpcs_unique_count": "unique_hcpcs_count"
})

print("\nProvider-level data shape:", provider_df.shape)



#  HCPCS CONCENTRATION FEATURE


hcpcs_share = (
    cms_df.groupby(["npi", "hcpcs"], observed=True)["total_services"]
    .sum()
    .reset_index()
)

provider_total = (
    hcpcs_share.groupby("npi", observed=True)["total_services"]
    .sum()
    .reset_index()
    .rename(columns={"total_services": "provider_total_services"})
)

hcpcs_share = hcpcs_share.merge(provider_total, on="npi", how="left")

hcpcs_share["hcpcs_service_share"] = (
    hcpcs_share["total_services"]
    / hcpcs_share["provider_total_services"].replace(0, np.nan)
)

max_hcpcs_share = (
    hcpcs_share.groupby("npi", observed=True)["hcpcs_service_share"]
    .max()
    .reset_index()
    .rename(columns={"hcpcs_service_share": "max_hcpcs_service_share"})
)

provider_df = provider_df.merge(max_hcpcs_share, on="npi", how="left")



#  PREPARE LOCAL OIG FRAUD LABELS (using data from the OFFICE OF THE INSPECTOR GENERAL)

leie_df.columns = (
    leie_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

if "npi" not in leie_df.columns:
    raise KeyError(f"'npi' column not found in OIG file. Columns found: {leie_df.columns.tolist()}")

leie_df["npi"] = pd.to_numeric(leie_df["npi"], errors="coerce").astype("Int64")

leie_df = leie_df.dropna(subset=["npi"])
leie_df = leie_df[leie_df["npi"] != 0]

fraud_labels = leie_df[["npi"]].drop_duplicates()
fraud_labels["is_fraud"] = 1

print("\nOIG records:", leie_df.shape)
print("Fraud-label records:", fraud_labels.shape)

#######################
######################
#  MERGE FRAUD LABELS
#####################
#####################


features_df = provider_df.merge(fraud_labels, on="npi", how="left")

features_df["is_fraud"] = features_df["is_fraud"].fillna(0).astype(int)

features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

numeric_features = features_df.select_dtypes(include=["number"]).columns
features_df[numeric_features] = features_df[numeric_features].fillna(0)

text_features = features_df.select_dtypes(include=["object", "category"]).columns

for col in text_features:
    if str(features_df[col].dtype) == "category":
        if "Unknown" not in features_df[col].cat.categories:
            features_df[col] = features_df[col].cat.add_categories(["Unknown"])

    features_df[col] = features_df[col].fillna("Unknown")

print("\nFinal feature dataset shape:", features_df.shape)
print("\nFraud label distribution:")
print(features_df["is_fraud"].value_counts())



#  HYPOTHESIS SUMMARY


hypothesis_summary = (
    features_df.groupby("is_fraud")
    .agg(
        h1_flag_count_mean=("h1_unusual_billing_flag_sum", "mean"),
        h1_flag_rate_mean=("h1_unusual_billing_flag_mean", "mean"),
        h2_flag_count_mean=("h2_geo_volume_spike_flag_sum", "mean"),
        h2_flag_rate_mean=("h2_geo_volume_spike_flag_mean", "mean"),
        h3_flag_count_mean=("h3_above_avg_hcpcs_cost_flag_sum", "mean"),
        h3_flag_rate_mean=("h3_above_avg_hcpcs_cost_flag_mean", "mean"),
        billing_disbursement_z_max_mean=("billing_disbursement_z_max", "mean"),
        geo_service_volume_z_max_mean=("geo_service_volume_z_max", "mean"),
        hcpcs_payment_cost_z_max_mean=("hcpcs_payment_cost_z_max", "mean")
    )
    .reset_index()
)

print("\nHypothesis summary:")
print(hypothesis_summary)



#  PREPARE MODELING DATA


drop_cols = [
    "npi", "last_name", "first_name", "city",
    "state", "provider_type", "is_fraud"
]

X = features_df.drop(columns=[col for col in drop_cols if col in features_df.columns])
y = features_df["is_fraud"]

X = X.replace([np.inf, -np.inf], np.nan)
numeric_X_cols = X.select_dtypes(include=["number"]).columns
X = X[numeric_X_cols].fillna(0)

print("\nModeling feature count:", X.shape[1])

if y.nunique() < 2:
    print("\nWARNING: Only one class found in is_fraud.")
    print("Supervised modeling will be skipped.")
    run_supervised_models = False
else:
    run_supervised_models = True


#########################
#  SUPERVISED MODELS
#########################
########################
if run_supervised_models:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    log_model.fit(X_train_scaled, y_train)

    log_pred = log_model.predict(X_test_scaled)
    log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

    print("\n================ Logistic Regression ================")
    print(confusion_matrix(y_test, log_pred))
    print(classification_report(y_test, log_pred))
    print("ROC AUC:", roc_auc_score(y_test, log_prob))

    rf_model = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

    print("\n================ Random Forest ================")
    print(confusion_matrix(y_test, rf_pred))
    print(classification_report(y_test, rf_pred))
    print("ROC AUC:", roc_auc_score(y_test, rf_prob))

    feature_importance = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": rf_model.feature_importances_
        })
        .sort_values("importance", ascending=False)
    )

    features_df["fraud_probability_rf"] = rf_model.predict_proba(X)[:, 1]

else:
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": np.nan
    })

    features_df["fraud_probability_rf"] = 0


###################################
#  UNSUPERVISED ANOMALY DETECTION
##################################
##################################

iso_model = IsolationForest(
    n_estimators=150,
    contamination=0.01,
    random_state=42,
    n_jobs=-1
)

iso_model.fit(X)

features_df["anomaly_score"] = iso_model.decision_function(X)

features_df["is_anomaly"] = iso_model.predict(X)

features_df["is_anomaly"] = features_df["is_anomaly"].map({
    1: 0,
    -1: 1
})

print("\nAnomaly distribution:")
print(features_df["is_anomaly"].value_counts())



# COMPOSITE RISK SCORE


features_df["hypothesis_risk_score"] = (
    0.30 * features_df["h1_unusual_billing_flag_mean"]
    + 0.30 * features_df["h2_geo_volume_spike_flag_mean"]
    + 0.30 * features_df["h3_above_avg_hcpcs_cost_flag_mean"]
    + 0.10 * features_df["max_hcpcs_service_share"]
)

features_df["final_risk_score"] = (
    0.50 * features_df["fraud_probability_rf"]
    + 0.25 * features_df["is_anomaly"]
    + 0.25 * features_df["hypothesis_risk_score"]
)

high_risk = features_df.sort_values("final_risk_score", ascending=False)


#################################
# DISPLAY TOP HIGH-RISK PROVIDERS
#############################
#########@@@@@@@@@@

display_cols = [
    "npi", "last_name", "first_name", "city", "state", "provider_type",
    "is_fraud", "fraud_probability_rf", "is_anomaly",
    "h1_unusual_billing_flag_sum",
    "h2_geo_volume_spike_flag_sum",
    "h3_above_avg_hcpcs_cost_flag_sum",
    "hypothesis_risk_score",
    "final_risk_score"
]

display_cols = [col for col in display_cols if col in high_risk.columns]

print("\nTop 25 high-risk providers:")
print(high_risk[display_cols].head(25))


