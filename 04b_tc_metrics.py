import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================

RESULTS_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"
PRICE_PATH = "eurostoxx600_prices.csv"
OUTPUT_PATH = "tc_estimation_literature.csv"

EVENT_COL = "tc_literature"   # change to tc_gsadf / tc_drawdown if needed
DATE_COL = "Date"

LOOKBACK_TRADING_DAYS = 250
HIT_WINDOWS = [30, 60, 120, 180]

N_SIMULATIONS = 20000
SEED = 42

TC_MIN_AFTER_T2 = 1
TC_MAX_AFTER_T2 = 500


# =========================
# LOAD DATA
# =========================

fits = pd.read_csv(RESULTS_PATH)
prices = pd.read_csv(PRICE_PATH)

prices[DATE_COL] = pd.to_datetime(prices[DATE_COL])

for col in ["t1", "t2", "tc_predicted"]:
    fits[col] = pd.to_datetime(fits[col], errors="coerce")

fits["positive_lppls_valid"] = (
    fits["positive_lppls_valid"]
    .astype(str)
    .str.lower()
    .eq("true")
)

prices = prices.sort_values(DATE_COL).reset_index(drop=True)

if EVENT_COL not in prices.columns:
    raise ValueError(f"Could not find event column: {EVENT_COL}")

events = prices.loc[prices[EVENT_COL] == 1, DATE_COL].sort_values().to_list()

if len(events) == 0:
    raise ValueError(f"No events found where {EVENT_COL} == 1")


# =========================
# FUNCTIONS
# =========================

def bootstrap_bias_test(errors, n_simulations=20000, seed=42):
    """
    Bootstrap test for systematic tc prediction bias.

    Signed error = tc_predicted - true_event_date.
    Positive values mean tc_predicted is late.
    """

    rng = np.random.default_rng(seed)
    errors = np.asarray(errors)
    n = len(errors)

    boot_means = np.empty(n_simulations)
    boot_medians = np.empty(n_simulations)

    for i in range(n_simulations):
        sample = rng.choice(errors, size=n, replace=True)
        boot_means[i] = np.mean(sample)
        boot_medians[i] = np.median(sample)

    mean_ci_low, mean_ci_high = np.percentile(boot_means, [2.5, 97.5])
    median_ci_low, median_ci_high = np.percentile(boot_medians, [2.5, 97.5])

    return {
        "mean_bias_ci_low": mean_ci_low,
        "mean_bias_ci_high": mean_ci_high,
        "mean_bias_significant": not (mean_ci_low <= 0 <= mean_ci_high),

        "median_bias_ci_low": median_ci_low,
        "median_bias_ci_high": median_ci_high,
        "median_bias_significant": not (median_ci_low <= 0 <= median_ci_high),
    }


def random_admissible_tc_test(event_fits_valid, event_date, hit_window, n_simulations=20000, seed=42):
    """
    Monte Carlo test for tc timing accuracy.

    Null:
        For each actual fit ending at t2, random tc is drawn from:
        [t2 + 1 day, t2 + 500 days]

    Tests:
        1. Hit rate: observed hit rate within ±hit_window days vs random.
        2. MAE: observed mean absolute error vs random.
    """

    rng = np.random.default_rng(seed)

    real_errors = (
        event_fits_valid["tc_predicted"] - event_date
    ).dt.days.to_numpy()

    real_abs_errors = np.abs(real_errors)

    real_hit_rate = np.mean(real_abs_errors <= hit_window)
    real_mae = np.mean(real_abs_errors)

    t2_minus_event = (
        event_fits_valid["t2"] - event_date
    ).dt.days.to_numpy()

    random_hit_rates = np.empty(n_simulations)
    random_maes = np.empty(n_simulations)

    for i in range(n_simulations):
        random_tc_after_t2 = rng.integers(
            TC_MIN_AFTER_T2,
            TC_MAX_AFTER_T2 + 1,
            size=len(event_fits_valid)
        )

        random_errors = t2_minus_event + random_tc_after_t2
        random_abs_errors = np.abs(random_errors)

        random_hit_rates[i] = np.mean(random_abs_errors <= hit_window)
        random_maes[i] = np.mean(random_abs_errors)

    p_hit = (1 + np.sum(random_hit_rates >= real_hit_rate)) / (n_simulations + 1)
    p_mae = (1 + np.sum(random_maes <= real_mae)) / (n_simulations + 1)

    return {
        f"hit_rate_pm_{hit_window}d": real_hit_rate,
        f"random_hit_rate_pm_{hit_window}d": random_hit_rates.mean(),
        f"p_hit_pm_{hit_window}d": p_hit,
        f"hit_pm_{hit_window}d_sig_10pct": p_hit < 0.10,

        f"mae_days_pm_{hit_window}d": real_mae,
        f"random_mae_days_pm_{hit_window}d": random_maes.mean(),
        f"p_mae_pm_{hit_window}d": p_mae,
        f"mae_pm_{hit_window}d_sig_10pct": p_mae < 0.10,
    }


# =========================
# MAIN LOOP
# =========================

rows = []

for event_date in events:
    event_idx = prices.index[prices[DATE_COL] == event_date][0]

    start_idx = max(0, event_idx - LOOKBACK_TRADING_DAYS)
    start_date = prices.loc[start_idx, DATE_COL]

    event_fits_all = fits[
        (fits["t2"] >= start_date) &
        (fits["t2"] < event_date)
    ].copy()

    event_fits_valid = event_fits_all[
        (event_fits_all["positive_lppls_valid"]) &
        (event_fits_all["tc_predicted"].notna())
    ].copy()

    n_total = len(event_fits_all)
    n_valid = len(event_fits_valid)

    row = {
        "event_date": event_date.strftime("%Y-%m-%d"),
        "n_total_fits": n_total,
        "n_valid_fits": n_valid,
        "valid_fit_pct": n_valid / n_total if n_total > 0 else np.nan,
    }

    if n_valid == 0:
        rows.append(row)
        continue

    event_fits_valid["tc_error_days"] = (
        event_fits_valid["tc_predicted"] - event_date
    ).dt.days

    errors = event_fits_valid["tc_error_days"].to_numpy()
    abs_errors = np.abs(errors)

    # Descriptive metrics
    row.update({
        "mean_tc_error_days": np.mean(errors),
        "median_tc_error_days": np.median(errors),
        "median_abs_tc_error_days": np.median(abs_errors),
        "iqr_tc_error_days": np.percentile(errors, 75) - np.percentile(errors, 25),
    })

    # Descriptive hit rates
    for h in HIT_WINDOWS:
        row[f"hit_rate_pm_{h}d"] = np.mean(abs_errors <= h)

    # Bootstrap bias tests
    row.update(
        bootstrap_bias_test(
            errors=errors,
            n_simulations=N_SIMULATIONS,
            seed=SEED
        )
    )

    # Monte Carlo random admissible tc tests
    for h in HIT_WINDOWS:
        row.update(
            random_admissible_tc_test(
                event_fits_valid=event_fits_valid,
                event_date=event_date,
                hit_window=h,
                n_simulations=N_SIMULATIONS,
                seed=SEED + h
            )
        )

    rows.append(row)


# =========================
# SAVE ONE CSV
# =========================

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_PATH, index=False)

print("\nSaved combined results to:", OUTPUT_PATH)
print("\nResults:")
print(results.to_string(index=False))

print("\nSignificant hit-rate tests at p < 0.10:")
for h in HIT_WINDOWS:
    col = f"p_hit_pm_{h}d"
    sig = results.loc[results[col] < 0.10, ["event_date", col]]
    if len(sig):
        print(f"\n±{h} days")
        print(sig.to_string(index=False))

print("\nSignificant MAE tests at p < 0.10:")
for h in HIT_WINDOWS:
    col = f"p_mae_pm_{h}d"
    sig = results.loc[results[col] < 0.10, ["event_date", col]]
    if len(sig):
        print(f"\n±{h} days")
        print(sig.to_string(index=False))

print("\nSignificant mean bias tests:")
sig_mean = results.loc[
    results["mean_bias_significant"] == True,
    ["event_date", "mean_tc_error_days", "mean_bias_ci_low", "mean_bias_ci_high"]
]
print(sig_mean.to_string(index=False) if len(sig_mean) else "None")

print("\nSignificant median bias tests:")
sig_median = results.loc[
    results["median_bias_significant"] == True,
    ["event_date", "median_tc_error_days", "median_bias_ci_low", "median_bias_ci_high"]
]
print(sig_median.to_string(index=False) if len(sig_median) else "None")