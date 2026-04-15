import pandas as pd
import numpy as np
from pathlib import Path

# load data (excel found in SRC)
file_path = Path(r"C:\Users\edwin\Desktop\Project-Data Science Finals\cms\medicare-anomaly-detection\src\med.csv")


# expanded columns for analysis (hcpc_cd and other variables included to expand opportunities for further exploration)
#eg. examining billing behaviors of providers by assessing gaps between submitted billing vs. approved
use_cols = [
    "Rndrng_NPI",
    "Tot_Srvcs",
    "Avg_Mdcr_Pymt_Amt",
    "HCPCS_Cd",
    "Avg_Mdcr_Stdzd_Amt",
    "Avg_Sbmtd_Chrg",
    "Rndrng_Prvdr_State_Abrvtn"
]

# read data
df = pd.read_csv(file_path, usecols=use_cols)

# rename columns
df = df.rename(columns={
    "Rndrng_NPI": "npi_id",
    "Tot_Srvcs": "tot_srvcs",
    "Avg_Mdcr_Pymt_Amt": "avg_mdcr_pmt",
    "HCPCS_Cd": "hcpcs",
    "Avg_Mdcr_Stdzd_Amt": "stdzd_amt",
    "Avg_Sbmtd_Chrg": "sbmt_chrg",
    "Rndrng_Prvdr_State_Abrvtn": "state"
})

# inspect
print(df.dtypes)
print(df.head(6))

# numeric vars
numeric_cols = ["tot_srvcs", "avg_mdcr_pmt", "sbmt_chrg", "stdzd_amt"]

for col in numeric_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r"[^\d.-]", "", regex=True)
        .replace("", np.nan)
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

# categorical vars
df["npi_id"] = df["npi_id"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
df["state"] = df["state"].astype(str).str.strip().str.upper().replace({"": np.nan, "NAN": np.nan})
df["hcpcs"] = df["hcpcs"].astype(str).str.strip().str.upper().replace({"": np.nan, "NAN": np.nan})

# missingness
df = df.dropna(subset=["npi_id", "state", "hcpcs"] + numeric_cols)

# non-positive values
df = df[
    (df["tot_srvcs"] > 0) &
    (df["avg_mdcr_pmt"] > 0) &
    (df["sbmt_chrg"] > 0) &
    (df["stdzd_amt"] > 0)
].copy()

# drop duplicates 
df = df.drop_duplicates().copy()

# state & hcpcs as cats
df["state"] = df["state"].astype("category")
df["hcpcs"] = df["hcpcs"].astype("category")

# preview
print("\ndata preview:")
print(df.head())
