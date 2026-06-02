# 05a: run confidence-conditioned regressions

import numpy as np  # numerical operations
import pandas as pd  # handle dataframes
import statsmodels.api as sm  # run statistical models
import statsmodels.formula.api as smf  # run formula regressions

FITS_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"  # lppl fit input
CONFIDENCE_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"  # confidence input
PRICE_PATH = "eurostoxx600_prices.csv"  # price and event input
OUTPUT_PANEL = "ci_tc_error_regression_panel.csv"  # regression panel output
OUTPUT_RESULTS = "ci_tc_error_regression_summary.csv"  # regression summary output
DATE_COL = "Date"  # price date column
EVENT_COL = "tc_literature"  # event label column
CI_COL = "positive_bubble_confidence"  # confidence column
LOOKBACK_TRADING_DAYS = 250  # pre-event lookback
VALID_ONLY = True  # keep only valid fits
MAX_ABS_ERROR_FOR_TEST = 750  # max error filter
N_BOOT = 10000  # bootstrap count
SEED = 42  # random seed

fits = pd.read_csv(FITS_PATH)  # load lppl fits
confidence = pd.read_csv(CONFIDENCE_PATH)  # load confidence data
prices = pd.read_csv(PRICE_PATH)  # load price data

for col in ["t1", "t2", "tc_predicted"]:  # loop over fit date columns
    fits[col] = pd.to_datetime(fits[col], errors="coerce")  # parse fit dates

confidence["t2"] = pd.to_datetime(confidence["t2"], errors="coerce")  # parse confidence dates
prices[DATE_COL] = pd.to_datetime(prices[DATE_COL], errors="coerce")  # parse price dates

fits["positive_lppls_valid"] = (
    fits["positive_lppls_valid"]
    .astype(str)
    .str.lower()
    .eq("true")
)  # convert validity flag

prices = prices.sort_values(DATE_COL).reset_index(drop=True)  # sort prices

fits = fits.merge(
    confidence[["t2", CI_COL]].dropna(),
    on="t2",
    how="left"
)  # merge ci onto fits

events = prices.loc[
    prices[EVENT_COL] == 1,
    DATE_COL
].dropna().sort_values().to_list()  # get event dates

chunks = []  # store event panels

for event_date in events:  # loop over events
    event_idx = prices.index[prices[DATE_COL] == event_date][0]  # get event index
    start_idx = max(0, event_idx - LOOKBACK_TRADING_DAYS)  # get lookback start
    start_date = prices.loc[start_idx, DATE_COL]  # get lookback date
    g = fits[
        (fits["t2"] >= start_date)
        & (fits["t2"] < event_date)
        & (fits["tc_predicted"].notna())
        & (fits[CI_COL].notna())
    ].copy()  # keep pre-event fits
    if VALID_ONLY:  # check valid-only setting
        g = g[g["positive_lppls_valid"]].copy()  # keep valid fits
    if len(g) == 0:  # check event observations
        continue  # skip empty events
    g["event_date"] = event_date.strftime("%Y-%m-%d")  # store event date
    g["signed_tc_error"] = (g["tc_predicted"] - event_date).dt.days  # calculate signed error
    g["absolute_tc_error"] = g["signed_tc_error"].abs()  # calculate absolute error
    chunks.append(g)  # store event data

if len(chunks) == 0:  # check panel exists
    raise ValueError("No valid event-fit observations found.")  # stop if empty

panel = pd.concat(chunks, ignore_index=True)  # combine event panels

if MAX_ABS_ERROR_FOR_TEST is not None:  # check error filter
    panel = panel[panel["absolute_tc_error"] <= MAX_ABS_ERROR_FOR_TEST].copy()  # filter large errors

panel = panel[
    [CI_COL, "signed_tc_error", "absolute_tc_error", "event_date"]
].dropna().copy()  # keep regression columns

panel = panel.rename(columns={CI_COL: "ci"})  # shorten ci column name
panel.to_csv(OUTPUT_PANEL, index=False)  # save regression panel

def p_format(p):
    if pd.isna(p):  # check missing p-value
        return ""  # return blank
    if p < 0.001:  # check small p-value
        return "<0.001"  # format small p-value
    return f"{p:.4f}"  # format p-value

def ci_format(low, high):
    return f"[{low:.1f}, {high:.1f}]"  # format confidence interval

def interp_for(dep_var, model_name, slope):
    if dep_var == "Signed tc error":  # check signed error model
        if slope < 0:  # check negative slope
            return "Higher CI reduces late bias"  # interpretation
        else:
            return "Higher CI increases late bias"  # interpretation
    if dep_var == "Absolute tc error":  # check absolute error model
        if slope < 0:  # check negative slope
            return "Higher CI improves accuracy"  # interpretation
        else:
            return "Higher CI does not improve accuracy"  # interpretation
    return ""  # fallback interpretation

def bootstrap_slope_ci(x, y, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)  # initialise random generator
    n = len(x)  # get sample size
    boot_slopes = np.empty(n_boot)  # store bootstrap slopes
    for i in range(n_boot):  # loop over bootstraps
        idx = rng.integers(0, n, size=n)  # sample rows
        xb = x[idx]  # bootstrap ci values
        yb = y[idx]  # bootstrap response values
        Xb = sm.add_constant(xb)  # add intercept
        fit_b = sm.OLS(yb, Xb).fit()  # fit bootstrap model
        boot_slopes[i] = fit_b.params[1]  # store slope
    ci_low, ci_high = np.percentile(boot_slopes, [2.5, 97.5])  # bootstrap ci
    boot_p = 2 * min(
        np.mean(boot_slopes <= 0),
        np.mean(boot_slopes >= 0)
    )  # bootstrap p-value
    return ci_low, ci_high, boot_p  # return bootstrap results

def run_models(dep_col, dep_label, seed_offset=0):
    rows = []  # store model rows
    x = panel["ci"].astype(float).to_numpy()  # get ci values
    y = panel[dep_col].astype(float).to_numpy()  # get response values
    X = sm.add_constant(x)  # add intercept
    ols = sm.OLS(y, X).fit()  # fit ols
    slope = ols.params[1]  # get slope
    ci_low, ci_high = ols.conf_int()[1]  # get ols ci
    p_val = ols.pvalues[1]  # get p-value
    rows.append({
        "Dependent variable": dep_label,
        "Model": "OLS",
        "CI coefficient": slope,
        "95% CI": ci_format(ci_low, ci_high),
        "p-value": p_format(p_val),
        "R²": ols.rsquared,
        "Interpretation": interp_for(dep_label, "OLS", slope),
    })  # store ols result
    ols_hc3 = ols.get_robustcov_results(cov_type="HC3")  # get robust standard errors
    hc3_ci_low, hc3_ci_high = ols_hc3.conf_int()[1]  # get robust ci
    hc3_p = ols_hc3.pvalues[1]  # get robust p-value
    rows.append({
        "Dependent variable": dep_label,
        "Model": "OLS, HC3 robust SE",
        "CI coefficient": slope,
        "95% CI": ci_format(hc3_ci_low, hc3_ci_high),
        "p-value": p_format(hc3_p),
        "R²": ols.rsquared,
        "Interpretation": interp_for(dep_label, "OLS HC3", slope),
    })  # store robust ols result
    boot_ci_low, boot_ci_high, boot_p = bootstrap_slope_ci(
        x=x,
        y=y,
        n_boot=N_BOOT,
        seed=SEED + seed_offset
    )  # run bootstrap slope test
    rows.append({
        "Dependent variable": dep_label,
        "Model": "Bootstrap slope CI",
        "CI coefficient": slope,
        "95% CI": ci_format(boot_ci_low, boot_ci_high),
        "p-value": p_format(boot_p),
        "R²": ols.rsquared,
        "Interpretation": interp_for(dep_label, "Bootstrap", slope),
    })  # store bootstrap result
    fe = smf.ols(
        f"{dep_col} ~ ci + C(event_date)",
        data=panel
    ).fit(cov_type="HC3")  # fit event fixed-effects model
    fe_slope = fe.params["ci"]  # get fixed-effects slope
    fe_p = fe.pvalues["ci"]  # get fixed-effects p-value
    fe_ci_low, fe_ci_high = fe.conf_int().loc["ci"]  # get fixed-effects ci
    rows.append({
        "Dependent variable": dep_label,
        "Model": "Event FE, HC3 robust SE",
        "CI coefficient": fe_slope,
        "95% CI": ci_format(fe_ci_low, fe_ci_high),
        "p-value": p_format(fe_p),
        "R²": fe.rsquared,
        "Interpretation": interp_for(dep_label, "Event FE HC3", fe_slope),
    })  # store fixed-effects result
    return rows  # return model rows

rows = []  # store all results
rows.extend(
    run_models(
        dep_col="signed_tc_error",
        dep_label="Signed tc error",
        seed_offset=0
    )
)  # run signed error models

rows.extend(
    run_models(
        dep_col="absolute_tc_error",
        dep_label="Absolute tc error",
        seed_offset=100
    )
)  # run absolute error models

results = pd.DataFrame(rows)  # create results table
results["CI coefficient"] = results["CI coefficient"].round(2)  # round coefficients
results["R²"] = results["R²"].round(4)  # round r squared
results.to_csv(OUTPUT_RESULTS, index=False)  # save results

print("\nSaved panel:", OUTPUT_PANEL)  # print panel output
print("Saved results:", OUTPUT_RESULTS)  # print summary output
print("\nCI-dependent LPPL tc estimation results:")  # print title
print(results.to_string(index=False))  # print results table

print("\nReadable thesis table:")  # print readable table header
for dep_var, g in results.groupby("Dependent variable", sort=False):  # loop over dependent variables
    print(f"\n{dep_var}")  # print dependent variable
    print(
        g[
            ["Model", "CI coefficient", "95% CI", "p-value", "Interpretation"]
        ].to_string(index=False)
    )  # print selected columns

print("\nNotes:")  # print notes header
print("Signed tc error = tc_predicted - tc_true. Negative coefficients mean higher CI shifts tc estimates earlier / reduces late bias.")  # explain signed error
print("Absolute tc error = |tc_predicted - tc_true|. Negative coefficients mean higher CI is associated with lower timing error.")  # explain absolute error
print("Coefficients are measured in calendar days per 1-unit increase in CI. A 0.10 increase in CI corresponds to one-tenth of the coefficient.")  # explain coefficient scale