# 05_ci_tc_error_regression_summary.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =========================
# CONFIG
# =========================

FITS_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"
CONFIDENCE_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"
PRICE_PATH = "eurostoxx600_prices.csv"

OUTPUT_PANEL = "ci_tc_error_regression_panel.csv"
OUTPUT_RESULTS = "ci_tc_error_regression_summary.csv"

DATE_COL = "Date"
EVENT_COL = "tc_literature"
CI_COL = "positive_bubble_confidence"

LOOKBACK_TRADING_DAYS = 250
VALID_ONLY = True
MAX_ABS_ERROR_FOR_TEST = 750

N_BOOT = 10000
SEED = 42

# =========================
# LOAD DATA
# =========================

fits = pd.read_csv(FITS_PATH)
confidence = pd.read_csv(CONFIDENCE_PATH)
prices = pd.read_csv(PRICE_PATH)

for col in ["t1", "t2", "tc_predicted"]:
    fits[col] = pd.to_datetime(fits[col], errors="coerce")

confidence["t2"] = pd.to_datetime(confidence["t2"], errors="coerce")
prices[DATE_COL] = pd.to_datetime(prices[DATE_COL], errors="coerce")

fits["positive_lppls_valid"] = (
    fits["positive_lppls_valid"]
    .astype(str)
    .str.lower()
    .eq("true")
)

prices = prices.sort_values(DATE_COL).reset_index(drop=True)

fits = fits.merge(
    confidence[["t2", CI_COL]].dropna(),
    on="t2",
    how="left"
)

events = prices.loc[
    prices[EVENT_COL] == 1,
    DATE_COL
].dropna().sort_values().to_list()

# =========================
# BUILD EVENT-FIT PANEL
# =========================

chunks = []

for event_date in events:
    event_idx = prices.index[prices[DATE_COL] == event_date][0]
    start_idx = max(0, event_idx - LOOKBACK_TRADING_DAYS)
    start_date = prices.loc[start_idx, DATE_COL]

    g = fits[
        (fits["t2"] >= start_date)
        & (fits["t2"] < event_date)
        & (fits["tc_predicted"].notna())
        & (fits[CI_COL].notna())
    ].copy()

    if VALID_ONLY:
        g = g[g["positive_lppls_valid"]].copy()

    if len(g) == 0:
        continue

    g["event_date"] = event_date.strftime("%Y-%m-%d")
    g["signed_tc_error"] = (g["tc_predicted"] - event_date).dt.days
    g["absolute_tc_error"] = g["signed_tc_error"].abs()

    chunks.append(g)

if len(chunks) == 0:
    raise ValueError("No valid event-fit observations found.")

panel = pd.concat(chunks, ignore_index=True)

if MAX_ABS_ERROR_FOR_TEST is not None:
    panel = panel[panel["absolute_tc_error"] <= MAX_ABS_ERROR_FOR_TEST].copy()

panel = panel[
    [CI_COL, "signed_tc_error", "absolute_tc_error", "event_date"]
].dropna().copy()

panel = panel.rename(columns={CI_COL: "ci"})

panel.to_csv(OUTPUT_PANEL, index=False)

# =========================
# FUNCTIONS
# =========================

def p_format(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def ci_format(low, high):
    return f"[{low:.1f}, {high:.1f}]"


def interp_for(dep_var, model_name, slope):
    if dep_var == "Signed tc error":
        if slope < 0:
            return "Higher CI reduces late bias"
        else:
            return "Higher CI increases late bias"

    if dep_var == "Absolute tc error":
        if slope < 0:
            return "Higher CI improves accuracy"
        else:
            return "Higher CI does not improve accuracy"

    return ""


def bootstrap_slope_ci(x, y, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)

    boot_slopes = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]

        Xb = sm.add_constant(xb)
        fit_b = sm.OLS(yb, Xb).fit()

        boot_slopes[i] = fit_b.params[1]

    ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])

    boot_p = 2 * min(
        np.mean(boot_slopes <= 0),
        np.mean(boot_slopes >= 0)
    )

    return ci_low, ci_high, boot_p


def run_models(dep_col, dep_label, seed_offset=0):
    rows = []

    x = panel["ci"].astype(float).to_numpy()
    y = panel[dep_col].astype(float).to_numpy()

    X = sm.add_constant(x)

    # Baseline OLS
    ols = sm.OLS(y, X).fit()
    slope = ols.params[1]
    intercept = ols.params[0]
    ci_low, ci_high = ols.conf_int()[1]
    p_val = ols.pvalues[1]

    rows.append({
        "Dependent variable": dep_label,
        "Model": "OLS",
        "CI coefficient": slope,
        "95% CI": ci_format(ci_low, ci_high),
        "p-value": p_format(p_val),
        "R²": ols.rsquared,
        "Interpretation": interp_for(dep_label, "OLS", slope),
    })

    # OLS with HC3 robust SE
    ols_hc3 = ols.get_robustcov_results(cov_type="HC3")
    hc3_ci_low, hc3_ci_high = ols_hc3.conf_int()[1]
    hc3_p = ols_hc3.pvalues[1]

    rows.append({
        "Dependent variable": dep_label,
        "Model": "OLS, HC3 robust SE",
        "CI coefficient": slope,
        "95% CI": ci_format(hc3_ci_low, hc3_ci_high),
        "p-value": p_format(hc3_p),
        "R²": ols.rsquared,
        "Interpretation": interp_for(dep_label, "OLS HC3", slope),
    })

    # Pooled bootstrap slope CI
    boot_ci_low, boot_ci_high, boot_p = bootstrap_slope_ci(
        x=x,
        y=y,
        n_boot=N_BOOT,
        seed=SEED + seed_offset
    )

    rows.append({
        "Dependent variable": dep_label,
        "Model": "Bootstrap slope CI",
        "CI coefficient": slope,
        "95% CI": ci_format(boot_ci_low, boot_ci_high),
        "p-value": p_format(boot_p),
        "R²": ols.rsquared,
        "Interpretation": interp_for(dep_label, "Bootstrap", slope),
    })

    # Event fixed effects with HC3 robust SE
    fe = smf.ols(
        f"{dep_col} ~ ci + C(event_date)",
        data=panel
    ).fit(cov_type="HC3")

    fe_slope = fe.params["ci"]
    fe_p = fe.pvalues["ci"]
    fe_ci_low, fe_ci_high = fe.conf_int().loc["ci"]

    rows.append({
        "Dependent variable": dep_label,
        "Model": "Event FE, HC3 robust SE",
        "CI coefficient": fe_slope,
        "95% CI": ci_format(fe_ci_low, fe_ci_high),
        "p-value": p_format(fe_p),
        "R²": fe.rsquared,
        "Interpretation": interp_for(dep_label, "Event FE HC3", fe_slope),
    })

    return rows


# =========================
# RUN TESTS
# =========================

rows = []

rows.extend(
    run_models(
        dep_col="signed_tc_error",
        dep_label="Signed tc error",
        seed_offset=0
    )
)

rows.extend(
    run_models(
        dep_col="absolute_tc_error",
        dep_label="Absolute tc error",
        seed_offset=100
    )
)

results = pd.DataFrame(rows)

results["CI coefficient"] = results["CI coefficient"].round(2)
results["R²"] = results["R²"].round(4)

results.to_csv(OUTPUT_RESULTS, index=False)

# =========================
# PRINT TABLE
# =========================

print("\nSaved panel:", OUTPUT_PANEL)
print("Saved results:", OUTPUT_RESULTS)

print("\nCI-dependent LPPLS tc estimation results:")
print(results.to_string(index=False))

print("\nReadable thesis table:")
for dep_var, g in results.groupby("Dependent variable", sort=False):
    print(f"\n{dep_var}")
    print(
        g[
            ["Model", "CI coefficient", "95% CI", "p-value", "Interpretation"]
        ].to_string(index=False)
    )

print("\nNotes:")
print("Signed tc error = tc_predicted - tc_true. Negative coefficients mean higher CI shifts tc estimates earlier / reduces late bias.")
print("Absolute tc error = |tc_predicted - tc_true|. Negative coefficients mean higher CI is associated with lower timing error.")
print("Coefficients are measured in calendar days per 1-unit increase in CI. A 0.10 increase in CI corresponds to one-tenth of the coefficient.")