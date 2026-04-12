import pandas as pd

file_path = r"..\Medicare Physician & Other Practitioners - by Provider and Service\2023\MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv"

use_cols = ["Rndrng_NPI", "Tot_Srvcs", "Avg_Mdcr_Pymt_Amt"]

df = pd.read_csv(file_path, usecols=use_cols)

df = df.dropna()

# Group by provider
provider = df.groupby("Rndrng_NPI", as_index=False).agg({
    "Avg_Mdcr_Pymt_Amt": "mean",
    "Tot_Srvcs": "sum"
})

# Z-score
mean_val = provider["Avg_Mdcr_Pymt_Amt"].mean()
std_val = provider["Avg_Mdcr_Pymt_Amt"].std()

provider["z_score"] = (provider["Avg_Mdcr_Pymt_Amt"] - mean_val) / std_val

# Outliers
outliers = provider[provider["z_score"] > 3].sort_values("z_score", ascending=False)

print("\nTop 10 payment outliers:\n")
print(outliers.head(10))

print(f"\nTotal providers analyzed: {len(provider):,}")
print(f"Providers with z-score > 3: {len(outliers):,}")