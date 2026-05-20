import pandas as pd
import numpy as np

# ==========================================
# CONFIG
# ==========================================

CSV_FILE = "baseline_gfc.csv"
TC_TRUE = "2007-06-01"
DATE_COL = "tc_predicted"

WINDOW_DAYS = 90
N_SIMULATIONS = 10000
RANDOM_SEED = 42

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(CSV_FILE)

tc_true = pd.to_datetime(TC_TRUE)

predicted_tc = pd.to_datetime(
    df[DATE_COL],
    errors="coerce"
).dropna()

# ==========================================
# COMPUTE TC ERRORS
# ==========================================

errors = (predicted_tc - tc_true).dt.days.to_numpy()

print("\n===================================")
print("LPPL STATISTICAL SIGNIFICANCE TEST")
print("===================================\n")

print(f"Number of fits: {len(errors)}")

# ==========================================
# OBSERVED METRICS
# ==========================================

observed_hit_rate = np.mean(np.abs(errors) <= WINDOW_DAYS)

observed_mae = np.mean(np.abs(errors))

observed_mean_error = np.mean(errors)

observed_median_error = np.median(errors)

print(f"\nObserved ±{WINDOW_DAYS} day hit rate:")
print(f"{observed_hit_rate * 100:.2f}%")

print(f"\nObserved MAE:")
print(f"{observed_mae:.2f} days")

print(f"\nObserved mean tc error:")
print(f"{observed_mean_error:.2f} days")

print(f"\nObserved median tc error:")
print(f"{observed_median_error:.2f} days")

# ==========================================
# RANDOM / PERMUTATION TEST
# ==========================================

rng = np.random.default_rng(RANDOM_SEED)

min_date = predicted_tc.min()
max_date = predicted_tc.max()

date_span_days = (max_date - min_date).days

random_hit_rates = []
random_maes = []

print("\nRunning permutation test...")

for _ in range(N_SIMULATIONS):

    fake_tc = min_date + pd.to_timedelta(
        rng.integers(0, date_span_days + 1),
        unit="D"
    )

    fake_errors = (predicted_tc - fake_tc).dt.days.to_numpy()

    random_hit_rates.append(
        np.mean(np.abs(fake_errors) <= WINDOW_DAYS)
    )

    random_maes.append(
        np.mean(np.abs(fake_errors))
    )

random_hit_rates = np.array(random_hit_rates)
random_maes = np.array(random_maes)

p_value_hit_rate = np.mean(
    random_hit_rates >= observed_hit_rate
)

p_value_mae = np.mean(
    random_maes <= observed_mae
)

# ==========================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ==========================================

print("Running bootstrap test...")

bootstrap_means = []
bootstrap_medians = []

for _ in range(N_SIMULATIONS):

    sample = rng.choice(
        errors,
        size=len(errors),
        replace=True
    )

    bootstrap_means.append(
        np.mean(sample)
    )

    bootstrap_medians.append(
        np.median(sample)
    )

bootstrap_means = np.array(bootstrap_means)
bootstrap_medians = np.array(bootstrap_medians)

mean_ci_low, mean_ci_high = np.percentile(
    bootstrap_means,
    [2.5, 97.5]
)

median_ci_low, median_ci_high = np.percentile(
    bootstrap_medians,
    [2.5, 97.5]
)

mean_bias_significant = not (
    mean_ci_low <= 0 <= mean_ci_high
)

median_bias_significant = not (
    median_ci_low <= 0 <= median_ci_high
)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n===================================")
print("PERMUTATION TEST RESULTS")
print("===================================\n")

print(f"Observed hit rate: {observed_hit_rate * 100:.2f}%")
print(f"Random mean hit rate: {random_hit_rates.mean() * 100:.2f}%")
print(f"P-value (hit rate): {p_value_hit_rate:.6f}")

if p_value_hit_rate < 0.05:
    print("=> Hit rate IS statistically significant")
else:
    print("=> Hit rate is NOT statistically significant")

print()

print(f"Observed MAE: {observed_mae:.2f} days")
print(f"Random mean MAE: {random_maes.mean():.2f} days")
print(f"P-value (MAE): {p_value_mae:.6f}")

if p_value_mae < 0.05:
    print("=> MAE IS statistically significant")
else:
    print("=> MAE is NOT statistically significant")

# ==========================================
# PRINT BOOTSTRAP RESULTS
# ==========================================

print("\n===================================")
print("BOOTSTRAP BIAS TEST")
print("===================================\n")

print(f"Mean tc error: {observed_mean_error:.2f} days")
print(f"95% CI: [{mean_ci_low:.2f}, {mean_ci_high:.2f}]")

if mean_bias_significant:
    print("=> Mean bias IS statistically significant")
else:
    print("=> Mean bias is NOT statistically significant")

print()

print(f"Median tc error: {observed_median_error:.2f} days")
print(f"95% CI: [{median_ci_low:.2f}, {median_ci_high:.2f}]")

if median_bias_significant:
    print("=> Median bias IS statistically significant")
else:
    print("=> Median bias is NOT statistically significant")

# ==========================================
# SAVE RESULTS
# ==========================================

results = {
    "observed_hit_rate_pct": observed_hit_rate * 100,
    "random_hit_rate_mean_pct": random_hit_rates.mean() * 100,
    "p_value_hit_rate": p_value_hit_rate,

    "observed_mae_days": observed_mae,
    "random_mae_mean_days": random_maes.mean(),
    "p_value_mae": p_value_mae,

    "mean_error_days": observed_mean_error,
    "mean_ci_low": mean_ci_low,
    "mean_ci_high": mean_ci_high,

    "median_error_days": observed_median_error,
    "median_ci_low": median_ci_low,
    "median_ci_high": median_ci_high,
}

results_df = pd.DataFrame(
    results.items(),
    columns=["metric", "value"]
)

results_df.to_csv(
    "lppl_significance_results.csv",
    index=False
)

print("\nSaved results to:")
print("lppl_significance_results.csv")