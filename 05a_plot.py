# 06_ci_bins_tc_accuracy.py

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

# =========================
# CONFIG
# =========================

FITS_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"
CONFIDENCE_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"
PRICE_PATH = "eurostoxx600_prices.csv"

OUTPUT_PANEL = "ci_bins_tc_accuracy_panel.csv"
OUTPUT_SUMMARY = "ci_bins_tc_accuracy_summary.csv"
OUTPUT_TESTS = "ci_bins_tc_accuracy_tests.csv"

DATE_COL = "Date"
EVENT_COL = "tc_literature"   # change to tc_drawdown if needed

CI_COL = "positive_bubble_confidence"

LOOKBACK_TRADING_DAYS = 250
VALID_ONLY = True

HIT_WINDOWS = [60, 120]

# Choose either "terciles" or "fixed"
BIN_METHOD = "terciles"

# Used only if BIN_METHOD = "fixed"
LOW_CI_CUTOFF = 0.05
HIGH_CI_CUTOFF = 0.15


# =========================
# LOAD
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

if EVENT_COL not in prices.columns:
    raise ValueError(f"Missing event column: {EVENT_COL}")

if CI_COL not in confidence.columns:
    raise ValueError(f"Missing CI column: {CI_COL}")


# =========================
# MERGE CI ONTO FITS
# =========================

confidence_small = confidence[["t2", CI_COL]].dropna().copy()

fits = fits.merge(
    confidence_small,
    on="t2",
    how="left"
)


# =========================
# BUILD EVENT PANEL
# =========================

events = prices.loc[prices[EVENT_COL] == 1, DATE_COL].dropna().sort_values().to_list()

chunks = []

for event_date in events:
    event_idx_arr = prices.index[prices[DATE_COL] == event_date].to_numpy()
    if len(event_idx_arr) == 0:
        continue

    event_idx = event_idx_arr[0]
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

    g["event_date"] = event_date
    g["tc_error_days"] = (g["tc_predicted"] - event_date).dt.days
    g["abs_tc_error_days"] = g["tc_error_days"].abs()

    for h in HIT_WINDOWS:
        g[f"hit_pm_{h}d"] = (g["abs_tc_error_days"] <= h).astype(int)

    chunks.append(g)

if len(chunks) == 0:
    raise ValueError("No valid matched observations found.")

panel = pd.concat(chunks, ignore_index=True)


# =========================
# ASSIGN CI BINS WITHIN EACH EVENT
# =========================

def assign_bins(g):
    g = g.copy()

    if BIN_METHOD == "terciles":
        try:
            g["ci_bin"] = pd.qcut(
                g[CI_COL],
                q=3,
                labels=["Low CI", "Medium CI", "High CI"],
                duplicates="drop"
            )
        except ValueError:
            g["ci_bin"] = np.nan

    elif BIN_METHOD == "fixed":
        conditions = [
            g[CI_COL] <= LOW_CI_CUTOFF,
            (g[CI_COL] > LOW_CI_CUTOFF) & (g[CI_COL] < HIGH_CI_CUTOFF),
            g[CI_COL] >= HIGH_CI_CUTOFF,
        ]
        choices = ["Low CI", "Medium CI", "High CI"]
        g["ci_bin"] = np.select(conditions, choices, default=np.nan)

    else:
        raise ValueError("BIN_METHOD must be 'terciles' or 'fixed'.")

    return g

panel = (
    panel
    .groupby("event_date", group_keys=False)
    .apply(assign_bins)
    .dropna(subset=["ci_bin"])
    .copy()
)

panel["ci_bin"] = panel["ci_bin"].astype(str)

panel.to_csv(OUTPUT_PANEL, index=False)


# =========================
# SUMMARY BY EVENT AND CI BIN
# =========================

agg_dict = {
    "n": ("abs_tc_error_days", "size"),
    "mean_ci": (CI_COL, "mean"),
    "median_ci": (CI_COL, "median"),
    "mean_abs_error": ("abs_tc_error_days", "mean"),
    "median_abs_error": ("abs_tc_error_days", "median"),
    "mae": ("abs_tc_error_days", "mean"),
    "mean_bias": ("tc_error_days", "mean"),
    "median_bias": ("tc_error_days", "median"),
}

for h in HIT_WINDOWS:
    agg_dict[f"hit_rate_pm_{h}d"] = (f"hit_pm_{h}d", "mean")

summary = (
    panel
    .groupby(["event_date", "ci_bin"])
    .agg(**agg_dict)
    .reset_index()
)

summary["event_date"] = pd.to_datetime(summary["event_date"]).dt.strftime("%Y-%m-%d")
summary.to_csv(OUTPUT_SUMMARY, index=False)


# =========================
# STATISTICAL TESTS
# =========================

test_rows = []

# Pooled tests
groups = {
    name: g["abs_tc_error_days"].to_numpy()
    for name, g in panel.groupby("ci_bin")
}

if all(k in groups for k in ["Low CI", "Medium CI", "High CI"]):
    kw = kruskal(groups["Low CI"], groups["Medium CI"], groups["High CI"])

    test_rows.append({
        "scope": "pooled",
        "test": "Kruskal-Wallis abs tc error across CI bins",
        "statistic": kw.statistic,
        "p_value": kw.pvalue,
        "interpretation": "tests whether abs error differs across Low/Medium/High CI bins",
    })

    mw = mannwhitneyu(
        groups["High CI"],
        groups["Low CI"],
        alternative="less"  # high CI should have lower abs error
    )

    test_rows.append({
        "scope": "pooled",
        "test": "Mann-Whitney High CI abs error < Low CI abs error",
        "statistic": mw.statistic,
        "p_value": mw.pvalue,
        "interpretation": "one-sided; significant means High CI has lower abs tc error than Low CI",
    })

    # Hit rate difference tests via normal approximation
    for h in HIT_WINDOWS:
        high = panel.loc[panel["ci_bin"] == "High CI", f"hit_pm_{h}d"].to_numpy()
        low = panel.loc[panel["ci_bin"] == "Low CI", f"hit_pm_{h}d"].to_numpy()

        high_rate = high.mean()
        low_rate = low.mean()
        diff = high_rate - low_rate

        test_rows.append({
            "scope": "pooled",
            "test": f"Hit-rate difference High CI - Low CI ±{h}d",
            "statistic": diff,
            "p_value": np.nan,
            "high_ci_hit_rate": high_rate,
            "low_ci_hit_rate": low_rate,
            "interpretation": "positive means High CI has better hit rate; no p-value here",
        })


# Event-level tests
for event_date, g_event in panel.groupby("event_date"):
    groups_event = {
        name: gg["abs_tc_error_days"].to_numpy()
        for name, gg in g_event.groupby("ci_bin")
    }

    if not all(k in groups_event for k in ["Low CI", "Medium CI", "High CI"]):
        continue

    if min(len(v) for v in groups_event.values()) < 5:
        continue

    kw = kruskal(
        groups_event["Low CI"],
        groups_event["Medium CI"],
        groups_event["High CI"]
    )

    test_rows.append({
        "scope": "event",
        "event_date": pd.Timestamp(event_date).strftime("%Y-%m-%d"),
        "test": "Kruskal-Wallis abs tc error across CI bins",
        "statistic": kw.statistic,
        "p_value": kw.pvalue,
        "interpretation": "tests whether abs error differs across CI bins within event",
    })

    mw = mannwhitneyu(
        groups_event["High CI"],
        groups_event["Low CI"],
        alternative="less"
    )

    test_rows.append({
        "scope": "event",
        "event_date": pd.Timestamp(event_date).strftime("%Y-%m-%d"),
        "test": "Mann-Whitney High CI abs error < Low CI abs error",
        "statistic": mw.statistic,
        "p_value": mw.pvalue,
        "interpretation": "one-sided; significant means High CI has lower abs tc error than Low CI",
    })

tests = pd.DataFrame(test_rows)
tests.to_csv(OUTPUT_TESTS, index=False)


# =========================
# PRINT RESULTS
# =========================

print("\nSaved:")
print(OUTPUT_PANEL)
print(OUTPUT_SUMMARY)
print(OUTPUT_TESTS)

print("\nSUMMARY BY EVENT AND CI BIN:")
print(summary.to_string(index=False))

print("\nSTATISTICAL TESTS:")
print(tests.to_string(index=False))

print("\nQUICK TAKEAWAY:")

pooled_mw = tests[
    (tests["scope"] == "pooled") &
    (tests["test"] == "Mann-Whitney High CI abs error < Low CI abs error")
]

if len(pooled_mw):
    p = pooled_mw["p_value"].iloc[0]

    high_med = panel.loc[panel["ci_bin"] == "High CI", "abs_tc_error_days"].median()
    low_med = panel.loc[panel["ci_bin"] == "Low CI", "abs_tc_error_days"].median()

    print(f"Pooled median abs error: High CI = {high_med:.2f} days, Low CI = {low_med:.2f} days.")
    print(f"One-sided Mann-Whitney p-value for High CI < Low CI: {p:.4g}.")

    if high_med < low_med and p < 0.10:
        print("Result: High CI fits have significantly lower tc error.")
    elif high_med < low_med:
        print("Result: High CI fits have lower median error, but not statistically significant.")
    else:
        print("Result: High CI fits do not improve tc accuracy.")