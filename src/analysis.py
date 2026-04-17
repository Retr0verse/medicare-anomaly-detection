import pandas as pd
import numpy as np

# Load dataset
file_path = "data/raw/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv"

use_cols = [
    "Rndrng_NPI",
    "HCPCS_Cd",
    "Place_Of_Srvc",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_Zip5",
    "Rndrng_Prvdr_RUCA",
    "Tot_Srvcs",
    "Tot_Benes",
    "Avg_Mdcr_Alowd_Amt",
    "Avg_Sbmtd_Chrg",
    "Avg_Mdcr_Stdzd_Amt",
    "Avg_Mdcr_Pymt_Amt"
]

df = pd.read_csv(file_path, usecols=use_cols, low_memory=False)

# Rename columns
df = df.rename(columns={
    "Rndrng_NPI": "npi_id",
    "HCPCS_Cd": "hcpcs",
    "Place_Of_Srvc": "place_of_service",
    "Rndrng_Prvdr_Type": "provider_type",
    "Rndrng_Prvdr_Zip5": "zip5",
    "Rndrng_Prvdr_RUCA": "ruca",
    "Tot_Srvcs": "tot_srvcs",
    "Tot_Benes": "tot_benes",
    "Avg_Mdcr_Alowd_Amt": "avg_allowed_amt",
    "Avg_Sbmtd_Chrg": "avg_submitted_charge",
    "Avg_Mdcr_Stdzd_Amt": "avg_standardized_amt",
    "Avg_Mdcr_Pymt_Amt": "avg_payment_amt"
})

# Clean numeric columns
numeric_cols = [
    "tot_srvcs",
    "tot_benes",
    "avg_allowed_amt",
    "avg_submitted_charge",
    "avg_standardized_amt",
    "avg_payment_amt"
]

for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"[^\d.-]", "", regex=True)
        .replace("", np.nan)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Clean categorical columns
cat_cols = ["npi_id", "hcpcs", "place_of_service", "provider_type", "zip5", "ruca"]

for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({"": np.nan, "nan": np.nan, "NAN": np.nan})

# Drop missing key fields
df = df.dropna(subset=cat_cols + numeric_cols).copy()

# Keep positive values only
df = df[
    (df["tot_srvcs"] > 0) &
    (df["tot_benes"] > 0) &
    (df["avg_allowed_amt"] > 0) &
    (df["avg_submitted_charge"] > 0) &
    (df["avg_standardized_amt"] > 0) &
    (df["avg_payment_amt"] > 0)
].copy()

print("Cleaned dataset shape:", df.shape)

# Finding 1: payment outliers within provider type + HCPCS
group_cols = ["provider_type", "hcpcs"]

group_stats = (
    df.groupby(group_cols, as_index=False)
      .agg(
          mean_allowed_amt=("avg_allowed_amt", "mean"),
          std_allowed_amt=("avg_allowed_amt", "std"),
          mean_payment_amt=("avg_payment_amt", "mean"),
          std_payment_amt=("avg_payment_amt", "std")
      )
)

df = df.merge(group_stats, on=group_cols, how="left")

df["allowed_zscore"] = (
    (df["avg_allowed_amt"] - df["mean_allowed_amt"]) / df["std_allowed_amt"]
)

df["payment_zscore"] = (
    (df["avg_payment_amt"] - df["mean_payment_amt"]) / df["std_payment_amt"]
)

# Fix inf values from zero std deviation
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["payment_zscore"])

payment_outliers = df[df["payment_zscore"] > 3].copy()

print("\nTop payment outliers:")
print(payment_outliers[
    ["npi_id", "provider_type", "hcpcs", "avg_payment_amt", "payment_zscore"]
].sort_values("payment_zscore", ascending=False).head(10))

print("\nTotal payment outliers:", len(payment_outliers))

# Finding 2: submitted charge vs allowed amount ratio
df["charge_to_allowed_ratio"] = df["avg_submitted_charge"] / df["avg_allowed_amt"]

high_ratio = df.sort_values("charge_to_allowed_ratio", ascending=False)

print("\nTop charge-to-allowed ratios:")
print(high_ratio[
    ["npi_id", "provider_type", "hcpcs", "avg_submitted_charge", "avg_allowed_amt", "charge_to_allowed_ratio"]
].head(10))

# Finding 3 prep: grouped regression-ready dataset
regression_df = (
    df.groupby(
        ["provider_type", "place_of_service", "ruca"],
        as_index=False
    ).agg(
        avg_allowed_amt=("avg_allowed_amt", "mean"),
        avg_payment_amt=("avg_payment_amt", "mean"),
        avg_standardized_amt=("avg_standardized_amt", "mean"),
        avg_services=("tot_srvcs", "mean"),
        avg_beneficiaries=("tot_benes", "mean")
    )
)

print("\nRegression-ready grouped dataset:")
print(regression_df.head(10))
print("\nRegression-ready shape:", regression_df.shape)