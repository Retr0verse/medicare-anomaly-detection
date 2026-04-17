# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:19:54 2026

@author: edwin
"""

import pandas as pd
import numpy as np
from pathlib import Path

# load (change path because I was unable to upload csv in the SRC folder)
file_path = Path(r"C:\Users\edwin\Desktop\Project-Data Science Finals\cms\med.csv")

# Proposed variables
use_cols = [
    "Rndrng_NPI",
    "Tot_Srvcs",
    "Avg_Mdcr_Pymt_Amt",
    "HCPCS_Cd",
    "Avg_Mdcr_Stdzd_Amt",
    "Avg_Sbmtd_Chrg",
    "Rndrng_Prvdr_State_Abrvtn",
    "Place_Of_Srvc",
    "Rndrng_Prvdr_Type",
    "Rndrng_Prvdr_Zip5",
    "Rndrng_Prvdr_RUCA",
    "Tot_Benes"
]

df = pd.read_csv(file_path, usecols=use_cols, low_memory=False)

# rename variabes
df = df.rename(columns={
    "Rndrng_NPI": "npi_id",
    "Tot_Srvcs": "tot_srvcs",
    "Avg_Mdcr_Pymt_Amt": "avg_mdcr_pmt",
    "HCPCS_Cd": "hcpcs",
    "Avg_Mdcr_Stdzd_Amt": "stdzd_amt",
    "Avg_Sbmtd_Chrg": "sbmt_chrg",
    "Rndrng_Prvdr_State_Abrvtn": "state",
    "Place_Of_Srvc": "place_of_service",
    "Rndrng_Prvdr_Type": "provider_type",
    "Rndrng_Prvdr_Zip5": "zip5",
    "Rndrng_Prvdr_RUCA": "ruca",
    "Tot_Benes": "tot_benes"
})

# inspect dat
print(df.dtypes)
print(df.head(6))

# numeric vars
numeric_cols = [
    "tot_srvcs",
    "avg_mdcr_pmt",
    "sbmt_chrg",
    "stdzd_amt",
    "tot_benes"
]

for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"[^\d.-]", "", regex=True)
        .replace("", np.nan)
    )
    df[col] = pd.to_numeric(df[col])

# categorical vars
df["npi_id"] = df["npi_id"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
df["state"] = df["state"].astype(str).str.strip().str.upper().replace({"": np.nan, "NAN": np.nan})
df["hcpcs"] = df["hcpcs"].astype(str).str.strip().str.upper().replace({"": np.nan, "NAN": np.nan})
df["place_of_service"] = df["place_of_service"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
df["provider_type"] = df["provider_type"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

df["zip5"] = (
    df["zip5"]
    .astype(str)
    .str.strip()
    .replace({"": np.nan, "nan": np.nan, "NAN": np.nan})
    .str.zfill(5)
)

df["ruca"] = (
    df["ruca"]
    .astype(str)
    .str.strip()
    .replace({"": np.nan, "nan": np.nan, "NAN": np.nan})
)


df["ruca"] = df["ruca"].str.replace(r"\.0$", "", regex=True)

# missingness
df = df.dropna(subset=[
    "npi_id", "state", "hcpcs",
    "place_of_service", "provider_type", "zip5", "ruca"
] + numeric_cols)

# eliminate non-positives
df = df[
    (df["tot_srvcs"] > 0) &
    (df["avg_mdcr_pmt"] > 0) &
    (df["sbmt_chrg"] > 0) &
    (df["stdzd_amt"] > 0) &
    (df["tot_benes"] > 0)
].copy()

# drop duplicates
df = df.drop_duplicates().copy()

# categorical casting
cat_cols = ["state", "hcpcs", "place_of_service", "provider_type", "zip5", "ruca"]
for col in cat_cols:
    df[col] = df[col].astype("category")

# preview dat
print("\npreview:")
print(df.head())


#provider-procedure
provider_procedure = (
    df.groupby(["npi_id", "hcpcs"], observed=True, as_index=False)
      .agg(
          total_services=("tot_srvcs", "sum"),
          total_beneficiaries=("tot_benes", "sum"),
          mean_submitted_charge=("sbmt_chrg", "mean"),
          mean_medicare_payment=("avg_mdcr_pmt", "mean"),
          mean_standardized_amount=("stdzd_amt", "mean"),
          provider_type=("provider_type", "first"),
          state=("state", "first"),
          zip5=("zip5", "first"),
          ruca=("ruca", "first"),
          place_of_service=("place_of_service", "first")
      )
)

print(provider_procedure.head())




# hcps × ZIP code
hcpcs_zip = (
    df.groupby(["hcpcs", "zip5"], observed=True, as_index=False)
      .agg(
          total_services=("tot_srvcs", "sum"),
          total_beneficiaries=("tot_benes", "sum"),
          mean_submitted_charge=("sbmt_chrg", "mean"),
          mean_medicare_payment=("avg_mdcr_pmt", "mean"),
          mean_standardized_amount=("stdzd_amt", "mean"),
          provider_count=("npi_id", "nunique")
      )
)
print(hcpcs_zip.head())

#hcps × RUCA (rural-urban divide)
hcpcs_ruca = (
    df.groupby(["hcpcs", "ruca"], observed=True, as_index=False)
      .agg(
          total_services=("tot_srvcs", "sum"),
          total_beneficiaries=("tot_benes", "sum"),
          mean_submitted_charge=("sbmt_chrg", "mean"),
          mean_medicare_payment=("avg_mdcr_pmt", "mean"),
          mean_standardized_amount=("stdzd_amt", "mean"),
          provider_count=("npi_id", "nunique")
      )
)

print(hcpcs_ruca.head())

#hcps × State
hcpcs_state = (
    df.groupby(["hcpcs", "state"], observed=True, as_index=False)
      .agg(
          total_services=("tot_srvcs", "sum"),
          total_beneficiaries=("tot_benes", "sum"),
          mean_submitted_charge=("sbmt_chrg", "mean"),
          mean_medicare_payment=("avg_mdcr_pmt", "mean"),
          mean_standardized_amount=("stdzd_amt", "mean"),
          provider_count=("npi_id", "nunique")
      )
)

print(hcpcs_state.head())


#provider type by hcps
provider_type_hcpcs = (
    df.groupby(["provider_type", "hcpcs"], observed=True, as_index=False)
      .agg(
          total_services=("tot_srvcs", "sum"),
          total_beneficiaries=("tot_benes", "sum"),
          mean_submitted_charge=("sbmt_chrg", "mean"),
          mean_medicare_payment=("avg_mdcr_pmt", "mean"),
          mean_standardized_amount=("stdzd_amt", "mean"),
          provider_count=("npi_id", "nunique")
      )
)

print(provider_type_hcpcs.head())