"""
Medicare Anomaly Detection - Final Analysis Script
MedData Integrity Team (MIT)

Purpose:
This script is designed to better align the final project analysis with the
original proposal and professor feedback. Instead of only reporting descriptive
statistics, it builds hypothesis-driven findings around unusual Medicare billing
patterns, provider-level anomaly detection, HCPCS cost outliers, and contextual
regression analysis.

Expected local project structure:
medicare-anomaly-detection/
    data/raw/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv
    src/analysis.py
    reports/

Run from the project root:
    python src/analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# -----------------------------------------------------------------------------
# 0. Paths and setup
# -----------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data/raw/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv"
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Load and clean data
# -----------------------------------------------------------------------------
def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    """Load selected CMS fields and perform basic cleaning."""

    use_cols = [
        "Rndrng_NPI",
        "HCPCS_Cd",
        "Place_Of_Srvc",
        "Rndrng_Prvdr_Type",
        "Rndrng_Prvdr_State_Abrvtn",
        "Rndrng_Prvdr_Zip5",
        "Rndrng_Prvdr_RUCA",
        "Tot_Srvcs",
        "Tot_Benes",
        "Avg_Mdcr_Alowd_Amt",
        "Avg_Sbmtd_Chrg",
        "Avg_Mdcr_Stdzd_Amt",
        "Avg_Mdcr_Pymt_Amt",
    ]

    df = pd.read_csv(file_path, usecols=use_cols, low_memory=False)

    df = df.rename(
        columns={
            "Rndrng_NPI": "npi_id",
            "HCPCS_Cd": "hcpcs",
            "Place_Of_Srvc": "place_of_service",
            "Rndrng_Prvdr_Type": "provider_type",
            "Rndrng_Prvdr_State_Abrvtn": "state",
            "Rndrng_Prvdr_Zip5": "zip5",
            "Rndrng_Prvdr_RUCA": "ruca",
            "Tot_Srvcs": "tot_srvcs",
            "Tot_Benes": "tot_benes",
            "Avg_Mdcr_Alowd_Amt": "avg_allowed_amt",
            "Avg_Sbmtd_Chrg": "avg_submitted_charge",
            "Avg_Mdcr_Stdzd_Amt": "avg_standardized_amt",
            "Avg_Mdcr_Pymt_Amt": "avg_payment_amt",
        }
    )

    numeric_cols = [
        "tot_srvcs",
        "tot_benes",
        "avg_allowed_amt",
        "avg_submitted_charge",
        "avg_standardized_amt",
        "avg_payment_amt",
    ]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d.-]", "", regex=True)
            .replace("", np.nan)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_cols = [
        "npi_id",
        "hcpcs",
        "place_of_service",
        "provider_type",
        "state",
        "zip5",
        "ruca",
    ]

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "NAN": np.nan})

    df["state"] = df["state"].str.upper()
    df["hcpcs"] = df["hcpcs"].str.upper()
    df["zip5"] = df["zip5"].str.zfill(5)

    df = df.dropna(subset=categorical_cols + numeric_cols).copy()

    # Keep valid positive observations only.
    df = df[
        (df["tot_srvcs"] > 0)
        & (df["tot_benes"] > 0)
        & (df["avg_allowed_amt"] > 0)
        & (df["avg_submitted_charge"] > 0)
        & (df["avg_standardized_amt"] > 0)
        & (df["avg_payment_amt"] > 0)
    ].copy()

    df = df.drop_duplicates().copy()
    return df


# -----------------------------------------------------------------------------
# 2. Utility functions
# -----------------------------------------------------------------------------
def add_zscore(df: pd.DataFrame, col: str, group_cols: list[str] | None = None) -> pd.Series:
    """Return a z-score series, optionally within groups."""
    if group_cols:
        mean = df.groupby(group_cols)[col].transform("mean")
        std = df.groupby(group_cols)[col].transform("std")
    else:
        mean = df[col].mean()
        std = df[col].std()

    z = (df[col] - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)
    return z


def save_plot(path: Path) -> None:
    """Save the current matplotlib figure cleanly."""
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# 3. Finding 1 - Provider-level anomaly detection
# -----------------------------------------------------------------------------
def finding_1_provider_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core project finding:
    Do unusual billing patterns identify anomalous providers?

    Builds provider-level features, standardizes them, and creates a combined
    anomaly score. This shifts the project from descriptive statistics into a
    directly proposal-aligned anomaly detection framework.
    """

    provider_df = (
        df.groupby("npi_id", as_index=False)
        .agg(
            total_services=("tot_srvcs", "sum"),
            total_beneficiaries=("tot_benes", "sum"),
            avg_allowed_amt=("avg_allowed_amt", "mean"),
            avg_payment_amt=("avg_payment_amt", "mean"),
            avg_submitted_charge=("avg_submitted_charge", "mean"),
            avg_standardized_amt=("avg_standardized_amt", "mean"),
            provider_type=("provider_type", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            state=("state", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            zip5=("zip5", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            ruca=("ruca", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            unique_hcpcs=("hcpcs", "nunique"),
        )
    )

    provider_df["services_per_beneficiary"] = (
        provider_df["total_services"] / provider_df["total_beneficiaries"]
    )
    provider_df["charge_to_allowed_ratio"] = (
        provider_df["avg_submitted_charge"] / provider_df["avg_allowed_amt"]
    )
    provider_df["payment_to_allowed_ratio"] = (
        provider_df["avg_payment_amt"] / provider_df["avg_allowed_amt"]
    )

    anomaly_features = [
        "avg_allowed_amt",
        "avg_payment_amt",
        "avg_submitted_charge",
        "services_per_beneficiary",
        "charge_to_allowed_ratio",
    ]

    for col in anomaly_features:
        provider_df[f"{col}_z"] = add_zscore(provider_df, col)

    z_cols = [f"{col}_z" for col in anomaly_features]

    # Use absolute z-scores so unusually high or unusually low patterns can be flagged.
    provider_df["anomaly_score"] = provider_df[z_cols].abs().sum(axis=1)
    provider_df["max_abs_zscore"] = provider_df[z_cols].abs().max(axis=1)
    provider_df["is_high_risk_outlier"] = provider_df["max_abs_zscore"] >= 3

    top_provider_anomalies = provider_df.sort_values(
        "anomaly_score", ascending=False
    ).head(25)

    # Save outputs.
    provider_df.to_csv(TABLES_DIR / "provider_anomaly_scores_all.csv", index=False)
    top_provider_anomalies.to_csv(TABLES_DIR / "top_provider_anomalies.csv", index=False)

    # Chart 1: distribution of anomaly scores.
    plt.figure(figsize=(10, 6))
    plt.hist(provider_df["anomaly_score"].dropna(), bins=60)
    plt.title("Distribution of Provider Anomaly Scores")
    plt.xlabel("Combined Anomaly Score")
    plt.ylabel("Number of Providers")
    save_plot(FIGURES_DIR / "provider_anomaly_score_distribution.png")

    # Chart 2: top 10 anomaly scores.
    top10 = top_provider_anomalies.head(10).copy()
    top10["npi_id"] = top10["npi_id"].astype(str)
    plt.figure(figsize=(10, 6))
    plt.barh(top10["npi_id"], top10["anomaly_score"])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Provider Anomaly Scores")
    plt.xlabel("Combined Anomaly Score")
    plt.ylabel("Provider NPI")
    save_plot(FIGURES_DIR / "top_10_provider_anomaly_scores.png")

    return provider_df


# -----------------------------------------------------------------------------
# 4. Finding 2 - HCPCS cost outliers
# -----------------------------------------------------------------------------
def finding_2_hcpcs_cost_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Support finding:
    Do certain billing codes exhibit unusually high cost variation across providers?

    Compares providers within HCPCS billing codes to identify payment/allowed
    amount outliers relative to code-specific peer groups.
    """

    hcpcs_df = df.copy()

    # Restrict to HCPCS groups with enough observations to make z-scores meaningful.
    hcpcs_counts = hcpcs_df["hcpcs"].value_counts()
    valid_hcpcs = hcpcs_counts[hcpcs_counts >= 30].index
    hcpcs_df = hcpcs_df[hcpcs_df["hcpcs"].isin(valid_hcpcs)].copy()

    hcpcs_df["hcpcs_allowed_zscore"] = add_zscore(hcpcs_df, "avg_allowed_amt", ["hcpcs"])
    hcpcs_df["hcpcs_payment_zscore"] = add_zscore(hcpcs_df, "avg_payment_amt", ["hcpcs"])
    hcpcs_df["hcpcs_charge_zscore"] = add_zscore(hcpcs_df, "avg_submitted_charge", ["hcpcs"])

    hcpcs_df["hcpcs_cost_outlier_score"] = (
        hcpcs_df[
            ["hcpcs_allowed_zscore", "hcpcs_payment_zscore", "hcpcs_charge_zscore"]
        ]
        .abs()
        .sum(axis=1)
    )

    hcpcs_outliers = hcpcs_df[
        (hcpcs_df["hcpcs_allowed_zscore"] >= 3)
        | (hcpcs_df["hcpcs_payment_zscore"] >= 3)
        | (hcpcs_df["hcpcs_charge_zscore"] >= 3)
    ].copy()

    top_hcpcs_outliers = hcpcs_outliers.sort_values(
        "hcpcs_cost_outlier_score", ascending=False
    ).head(25)

    hcpcs_summary = (
        hcpcs_df.groupby("hcpcs", as_index=False)
        .agg(
            observation_count=("npi_id", "count"),
            provider_count=("npi_id", "nunique"),
            mean_allowed_amt=("avg_allowed_amt", "mean"),
            std_allowed_amt=("avg_allowed_amt", "std"),
            mean_payment_amt=("avg_payment_amt", "mean"),
            std_payment_amt=("avg_payment_amt", "std"),
            outlier_count=("hcpcs_cost_outlier_score", lambda x: int((x >= 9).sum())),
        )
        .sort_values("outlier_count", ascending=False)
    )

    top_hcpcs_outliers.to_csv(TABLES_DIR / "top_hcpcs_cost_outliers.csv", index=False)
    hcpcs_summary.to_csv(TABLES_DIR / "hcpcs_cost_variability_summary.csv", index=False)

    # Chart: Top HCPCS codes by count of extreme outlier records.
    top_codes = hcpcs_summary[hcpcs_summary["outlier_count"] > 0].head(10)
    if not top_codes.empty:
        plt.figure(figsize=(10, 6))
        plt.barh(top_codes["hcpcs"].astype(str), top_codes["outlier_count"])
        plt.gca().invert_yaxis()
        plt.title("HCPCS Codes With Most Extreme Cost Outliers")
        plt.xlabel("Number of Extreme Outlier Records")
        plt.ylabel("HCPCS Code")
        save_plot(FIGURES_DIR / "hcpcs_codes_with_most_cost_outliers.png")

    return top_hcpcs_outliers


# -----------------------------------------------------------------------------
# 5. Finding 3 - Contextual regression analysis
# -----------------------------------------------------------------------------
def finding_3_contextual_regression(df: pd.DataFrame) -> tuple[object, object]:
    """
    Support finding:
    Do geographic and provider characteristics explain variation in allowed amounts?

    Regression provides context so that anomaly detection is not simply flagging
    providers because of specialty, service setting, or rural/urban differences.
    """

    regression_df = (
        df.groupby(["provider_type", "place_of_service", "ruca"], as_index=False)
        .agg(
            avg_allowed_amt=("avg_allowed_amt", "mean"),
            avg_payment_amt=("avg_payment_amt", "mean"),
            avg_standardized_amt=("avg_standardized_amt", "mean"),
            avg_services=("tot_srvcs", "mean"),
            avg_beneficiaries=("tot_benes", "mean"),
            record_count=("npi_id", "count"),
        )
    )

    regression_df = regression_df.dropna().copy()
    regression_df = regression_df[regression_df["record_count"] >= 5].copy()

    regression_df["provider_type"] = regression_df["provider_type"].astype("category")
    regression_df["place_of_service"] = regression_df["place_of_service"].astype("category")
    regression_df["ruca"] = regression_df["ruca"].astype(str).astype("category")

    base_formula = """
    avg_allowed_amt ~ C(provider_type) + C(place_of_service) + C(ruca) + avg_services + avg_beneficiaries
    """

    interaction_formula = """
    avg_allowed_amt ~ C(provider_type) + C(place_of_service) + C(ruca) + avg_services * avg_beneficiaries
    """

    base_model = smf.ols(formula=base_formula, data=regression_df).fit()
    interaction_model = smf.ols(formula=interaction_formula, data=regression_df).fit()

    # Save regression summaries.
    with open(REPORTS_DIR / "base_regression_summary.txt", "w", encoding="utf-8") as f:
        f.write(base_model.summary().as_text())

    with open(REPORTS_DIR / "interaction_regression_summary.txt", "w", encoding="utf-8") as f:
        f.write(interaction_model.summary().as_text())

    base_coef = pd.DataFrame(
        {
            "variable": base_model.params.index,
            "coefficient": base_model.params.values,
            "p_value": base_model.pvalues.values,
        }
    )
    interaction_coef = pd.DataFrame(
        {
            "variable": interaction_model.params.index,
            "coefficient": interaction_model.params.values,
            "p_value": interaction_model.pvalues.values,
        }
    )

    base_coef.to_csv(TABLES_DIR / "base_regression_coefficients.csv", index=False)
    interaction_coef.to_csv(TABLES_DIR / "interaction_regression_coefficients.csv", index=False)

    model_summary = pd.DataFrame(
        [
            {
                "model": "Base OLS",
                "dependent_variable": "avg_allowed_amt",
                "r_squared": base_model.rsquared,
                "adj_r_squared": base_model.rsquared_adj,
                "observations": int(base_model.nobs),
                "aic": base_model.aic,
                "bic": base_model.bic,
            },
            {
                "model": "Interaction OLS",
                "dependent_variable": "avg_allowed_amt",
                "r_squared": interaction_model.rsquared,
                "adj_r_squared": interaction_model.rsquared_adj,
                "observations": int(interaction_model.nobs),
                "aic": interaction_model.aic,
                "bic": interaction_model.bic,
            },
        ]
    )
    model_summary.to_csv(TABLES_DIR / "regression_model_summary.csv", index=False)

    return base_model, interaction_model


# -----------------------------------------------------------------------------
# 6. Executive summary output for final report drafting
# -----------------------------------------------------------------------------
def write_project_summary(
    df: pd.DataFrame,
    provider_scores: pd.DataFrame,
    top_hcpcs_outliers: pd.DataFrame,
    base_model,
    interaction_model,
) -> None:
    """Write a short plain-English summary for the final report."""

    total_records = len(df)
    total_providers = df["npi_id"].nunique()
    total_hcpcs = df["hcpcs"].nunique()
    high_risk_providers = int(provider_scores["is_high_risk_outlier"].sum())

    summary_text = f"""
Medicare Anomaly Detection - Final Analysis Summary

Dataset after cleaning:
- Records analyzed: {total_records:,}
- Unique providers: {total_providers:,}
- Unique HCPCS codes: {total_hcpcs:,}

Finding 1 - Provider-Level Anomaly Detection:
- Built provider-level features for payment, allowed amount, submitted charge, services per beneficiary, and charge-to-allowed ratio.
- Standardized each feature using z-scores.
- Created a combined anomaly score to rank providers with unusual billing behavior.
- Providers with any absolute z-score >= 3 were flagged as high-risk outliers.
- High-risk provider outliers identified: {high_risk_providers:,}

Finding 2 - HCPCS Cost Outliers:
- Compared provider billing amounts within HCPCS billing-code peer groups.
- Flagged providers with extreme allowed, payment, or submitted-charge z-scores.
- Top HCPCS cost outlier records saved to reports/tables/top_hcpcs_cost_outliers.csv.

Finding 3 - Contextual Regression:
- Estimated OLS models to explain average allowed amount using provider type, place of service, RUCA, services, and beneficiaries.
- Base model R-squared: {base_model.rsquared:.3f}
- Interaction model R-squared: {interaction_model.rsquared:.3f}
- Regression adds context so outlier detection is not interpreted without considering provider specialty and service setting.

Interpretation:
The final analysis supports the proposal by showing that unusual billing patterns can be quantified and used as proxies for Medicare reimbursement anomalies. These results should be interpreted as risk indicators for further review, not proof of fraud.
""".strip()

    with open(REPORTS_DIR / "final_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)


# -----------------------------------------------------------------------------
# 7. Main execution
# -----------------------------------------------------------------------------
def main() -> None:
    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"Cleaned dataset shape: {df.shape}")

    print("\nRunning Finding 1: provider-level anomaly detection...")
    provider_scores = finding_1_provider_anomaly_detection(df)
    print("Saved provider anomaly outputs.")

    print("\nRunning Finding 2: HCPCS cost outlier analysis...")
    top_hcpcs_outliers = finding_2_hcpcs_cost_outliers(df)
    print("Saved HCPCS cost outlier outputs.")

    print("\nRunning Finding 3: contextual regression analysis...")
    base_model, interaction_model = finding_3_contextual_regression(df)
    print("Saved regression outputs.")

    print("\nWriting final analysis summary...")
    write_project_summary(df, provider_scores, top_hcpcs_outliers, base_model, interaction_model)

    print("\nDONE. Key outputs saved to:")
    print("- reports/final_analysis_summary.txt")
    print("- reports/tables/top_provider_anomalies.csv")
    print("- reports/tables/top_hcpcs_cost_outliers.csv")
    print("- reports/tables/regression_model_summary.csv")
    print("- reports/figures/provider_anomaly_score_distribution.png")
    print("- reports/figures/top_10_provider_anomaly_scores.png")


if __name__ == "__main__":
    main()
