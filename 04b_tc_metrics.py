# 04b: test critical time estimation performance

import numpy as np  # numerical operations
import pandas as pd  # handle dataframes

RESULTS_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"  # lppl fit input
PRICE_PATH = "eurostoxx600_prices.csv"  # price and event input
OUTPUT_PATH = "tc_estimation_literature.csv"  # output file
EVENT_COL = "tc_literature"  # event label column
DATE_COL = "Date"  # price date column
LOOKBACK_TRADING_DAYS = 250  # pre-event lookback
HIT_WINDOWS = [30, 60, 120, 180]  # hit-rate windows
N_SIMULATIONS = 20000  # simulation count
SEED = 42  # random seed
TC_MIN_AFTER_T2 = 1  # minimum random tc after t2
TC_MAX_AFTER_T2 = 500  # maximum random tc after t2

fits = pd.read_csv(RESULTS_PATH)  # load lppl fits
prices = pd.read_csv(PRICE_PATH)  # load price data
prices[DATE_COL] = pd.to_datetime(prices[DATE_COL])  # parse price dates

for col in ["t1", "t2", "tc_predicted"]:  # loop over fit date columns
    fits[col] = pd.to_datetime(fits[col], errors="coerce")  # parse fit dates

fits["positive_lppls_valid"] = (
    fits["positive_lppls_valid"]
    .astype(str)
    .str.lower()
    .eq("true")
)  # convert validity flag

prices = prices.sort_values(DATE_COL).reset_index(drop=True)  # sort prices

if EVENT_COL not in prices.columns:  # check event column exists
    raise ValueError(f"Could not find event column: {EVENT_COL}")  # stop if missing

events = prices.loc[prices[EVENT_COL] == 1, DATE_COL].sort_values().to_list()  # get event dates

if len(events) == 0:  # check events exist
    raise ValueError(f"No events found where {EVENT_COL} == 1")  # stop if none

def bootstrap_bias_test(errors, n_simulations=20000, seed=42):
    rng = np.random.default_rng(seed)  # initialise random generator
    errors = np.asarray(errors)  # convert errors to array
    n = len(errors)  # count errors
    boot_means = np.empty(n_simulations)  # store bootstrap means
    boot_medians = np.empty(n_simulations)  # store bootstrap medians
    for i in range(n_simulations):  # loop over bootstrap samples
        sample = rng.choice(errors, size=n, replace=True)  # resample errors
        boot_means[i] = np.mean(sample)  # store mean error
        boot_medians[i] = np.median(sample)  # store median error
    mean_ci_low, mean_ci_high = np.percentile(boot_means, [2.5, 97.5])  # mean ci bounds
    median_ci_low, median_ci_high = np.percentile(boot_medians, [2.5, 97.5])  # median ci bounds
    return {
        "mean_bias_ci_low": mean_ci_low,
        "mean_bias_ci_high": mean_ci_high,
        "mean_bias_significant": not (mean_ci_low <= 0 <= mean_ci_high),
        "median_bias_ci_low": median_ci_low,
        "median_bias_ci_high": median_ci_high,
        "median_bias_significant": not (median_ci_low <= 0 <= median_ci_high),
    }  # return bias test results

def random_admissible_tc_test(event_fits_valid, event_date, hit_window, n_simulations=20000, seed=42):
    rng = np.random.default_rng(seed)  # initialise random generator
    real_errors = (
        event_fits_valid["tc_predicted"] - event_date
    ).dt.days.to_numpy()  # calculate real signed errors
    real_abs_errors = np.abs(real_errors)  # calculate real absolute errors
    real_hit_rate = np.mean(real_abs_errors <= hit_window)  # calculate real hit rate
    real_mae = np.mean(real_abs_errors)  # calculate real mae
    t2_minus_event = (
        event_fits_valid["t2"] - event_date
    ).dt.days.to_numpy()  # calculate t2 relative to event
    random_hit_rates = np.empty(n_simulations)  # store random hit rates
    random_maes = np.empty(n_simulations)  # store random maes
    for i in range(n_simulations):  # loop over simulations
        random_tc_after_t2 = rng.integers(
            TC_MIN_AFTER_T2,
            TC_MAX_AFTER_T2 + 1,
            size=len(event_fits_valid)
        )  # sample random tc offsets
        random_errors = t2_minus_event + random_tc_after_t2  # calculate random errors
        random_abs_errors = np.abs(random_errors)  # calculate random absolute errors
        random_hit_rates[i] = np.mean(random_abs_errors <= hit_window)  # store random hit rate
        random_maes[i] = np.mean(random_abs_errors)  # store random mae
    p_hit = (1 + np.sum(random_hit_rates >= real_hit_rate)) / (n_simulations + 1)  # hit-rate p-value
    p_mae = (1 + np.sum(random_maes <= real_mae)) / (n_simulations + 1)  # mae p-value
    return {
        f"hit_rate_pm_{hit_window}d": real_hit_rate,
        f"random_hit_rate_pm_{hit_window}d": random_hit_rates.mean(),
        f"p_hit_pm_{hit_window}d": p_hit,
        f"hit_pm_{hit_window}d_sig_10pct": p_hit < 0.10,
        f"mae_days_pm_{hit_window}d": real_mae,
        f"random_mae_days_pm_{hit_window}d": random_maes.mean(),
        f"p_mae_pm_{hit_window}d": p_mae,
        f"mae_pm_{hit_window}d_sig_10pct": p_mae < 0.10,
    }  # return benchmark results

rows = []  # store event results

for event_date in events:  # loop over events
    event_idx = prices.index[prices[DATE_COL] == event_date][0]  # get event index
    start_idx = max(0, event_idx - LOOKBACK_TRADING_DAYS)  # get lookback start index
    start_date = prices.loc[start_idx, DATE_COL]  # get lookback start date
    event_fits_all = fits[
        (fits["t2"] >= start_date) &
        (fits["t2"] < event_date)
    ].copy()  # keep pre-event fits
    event_fits_valid = event_fits_all[
        (event_fits_all["positive_lppls_valid"]) &
        (event_fits_all["tc_predicted"].notna())
    ].copy()  # keep valid fits
    n_total = len(event_fits_all)  # count all fits
    n_valid = len(event_fits_valid)  # count valid fits
    row = {
        "event_date": event_date.strftime("%Y-%m-%d"),
        "n_total_fits": n_total,
        "n_valid_fits": n_valid,
        "valid_fit_pct": n_valid / n_total if n_total > 0 else np.nan,
    }  # create event row
    if n_valid == 0:  # check valid fits exist
        rows.append(row)  # store empty event row
        continue  # skip event
    event_fits_valid["tc_error_days"] = (
        event_fits_valid["tc_predicted"] - event_date
    ).dt.days  # calculate signed errors
    errors = event_fits_valid["tc_error_days"].to_numpy()  # get signed errors
    abs_errors = np.abs(errors)  # get absolute errors
    row.update({
        "mean_tc_error_days": np.mean(errors),
        "median_tc_error_days": np.median(errors),
        "median_abs_tc_error_days": np.median(abs_errors),
        "iqr_tc_error_days": np.percentile(errors, 75) - np.percentile(errors, 25),
    })  # add descriptive statistics
    for h in HIT_WINDOWS:  # loop over hit windows
        row[f"hit_rate_pm_{h}d"] = np.mean(abs_errors <= h)  # add hit rate
    row.update(
        bootstrap_bias_test(
            errors=errors,
            n_simulations=N_SIMULATIONS,
            seed=SEED
        )
    )  # add bootstrap bias tests
    for h in HIT_WINDOWS:  # loop over hit windows
        row.update(
            random_admissible_tc_test(
                event_fits_valid=event_fits_valid,
                event_date=event_date,
                hit_window=h,
                n_simulations=N_SIMULATIONS,
                seed=SEED + h
            )
        )  # add random benchmark tests
    rows.append(row)  # store event row

results = pd.DataFrame(rows)  # create results table
results.to_csv(OUTPUT_PATH, index=False)  # save results

print("\nSaved combined results to:", OUTPUT_PATH)  # print save path
print("\nResults:")  # print results header
print(results.to_string(index=False))  # print results table

print("\nSignificant hit-rate tests at p < 0.10:")  # print hit-rate header
for h in HIT_WINDOWS:  # loop over hit windows
    col = f"p_hit_pm_{h}d"  # get p-value column
    sig = results.loc[results[col] < 0.10, ["event_date", col]]  # filter significant rows
    if len(sig):  # check significant rows
        print(f"\n±{h} days")  # print window label
        print(sig.to_string(index=False))  # print significant rows

print("\nSignificant MAE tests at p < 0.10:")  # print mae header
for h in HIT_WINDOWS:  # loop over hit windows
    col = f"p_mae_pm_{h}d"  # get p-value column
    sig = results.loc[results[col] < 0.10, ["event_date", col]]  # filter significant rows
    if len(sig):  # check significant rows
        print(f"\n±{h} days")  # print window label
        print(sig.to_string(index=False))  # print significant rows

print("\nSignificant mean bias tests:")  # print mean bias header
sig_mean = results.loc[
    results["mean_bias_significant"] == True,
    ["event_date", "mean_tc_error_days", "mean_bias_ci_low", "mean_bias_ci_high"]
]  # filter mean bias results
print(sig_mean.to_string(index=False) if len(sig_mean) else "None")  # print mean bias results

print("\nSignificant median bias tests:")  # print median bias header
sig_median = results.loc[
    results["median_bias_significant"] == True,
    ["event_date", "median_tc_error_days", "median_bias_ci_low", "median_bias_ci_high"]
]  # filter median bias results
print(sig_median.to_string(index=False) if len(sig_median) else "None")  # print median bias results