# 08_recent_future_tc_distribution.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde

# ============================================================
# CONFIG
# ============================================================

FITS_CSV = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"
OUTPUT_DIR = "daily_fullscale"

OUTPUT_PNG = "recent_750d_future_tc_distribution.png"
OUTPUT_CSV = "recent_750d_future_tc_predictions.csv"
OUTPUT_SUMMARY = "recent_750d_future_tc_summary.csv"

T2_COL = "t2"
TC_PRED_COL = "tc_predicted"
VALID_COL = "positive_lppls_valid"

LOOKBACK_DAYS = 750
VALID_ONLY = True
FUTURE_ONLY = True          # only keep tc_predicted after latest t2
MIN_FITS_FOR_KDE = 5

# Plot style
FIGSIZE = (17.76, 9.95)
KDE_LINEWIDTH = 3.2
MEAN_LINEWIDTH = 2.8
MEDIAN_LINEWIDTH = 2.8
GRID_LINEWIDTH = 1.2

KDE_GRID_POINTS = 2000
KDE_PAD_DAYS = 180

# ============================================================
# LOAD DATA
# ============================================================

fits = pd.read_csv(FITS_CSV)

fits[T2_COL] = pd.to_datetime(fits[T2_COL], errors="coerce")
fits[TC_PRED_COL] = pd.to_datetime(fits[TC_PRED_COL], errors="coerce")

fits[VALID_COL] = (
    fits[VALID_COL]
    .astype(str)
    .str.lower()
    .eq("true")
)

fits = fits.dropna(subset=[T2_COL, TC_PRED_COL]).copy()
fits = fits.sort_values(T2_COL).reset_index(drop=True)

latest_t2 = fits[T2_COL].max()
start_date = latest_t2 - pd.Timedelta(days=LOOKBACK_DAYS)

recent = fits[
    (fits[T2_COL] >= start_date) &
    (fits[T2_COL] <= latest_t2)
].copy()

if VALID_ONLY:
    recent = recent[recent[VALID_COL]].copy()

if FUTURE_ONLY:
    recent = recent[recent[TC_PRED_COL] > latest_t2].copy()

if len(recent) == 0:
    raise ValueError("No recent future tc predictions found after filtering.")

# ============================================================
# SUMMARY STATS
# ============================================================

recent["tc_days_from_latest_t2"] = (
    recent[TC_PRED_COL] - latest_t2
).dt.days

mean_tc = recent[TC_PRED_COL].mean()
median_tc = recent[TC_PRED_COL].median()

mean_days = recent["tc_days_from_latest_t2"].mean()
median_days = recent["tc_days_from_latest_t2"].median()

q05 = recent[TC_PRED_COL].quantile(0.05)
q25 = recent[TC_PRED_COL].quantile(0.25)
q75 = recent[TC_PRED_COL].quantile(0.75)
q95 = recent[TC_PRED_COL].quantile(0.95)

summary = pd.DataFrame([{
    "latest_t2": latest_t2,
    "lookback_start": start_date,
    "lookback_days": LOOKBACK_DAYS,
    "n_fits_used": len(recent),
    "mean_future_tc": mean_tc,
    "median_future_tc": median_tc,
    "mean_days_from_latest_t2": mean_days,
    "median_days_from_latest_t2": median_days,
    "tc_5pct": q05,
    "tc_25pct": q25,
    "tc_75pct": q75,
    "tc_95pct": q95,
}])

os.makedirs(OUTPUT_DIR, exist_ok=True)

recent.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV), index=False)
summary.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_SUMMARY), index=False)

print("\nRecent future tc distribution")
print("--------------------------------")
print(f"Latest t2:              {latest_t2.date()}")
print(f"Lookback start:         {start_date.date()}")
print(f"Fits used:              {len(recent)}")
print(f"Mean expected tc:       {mean_tc.date()}  ({mean_days:.1f} days after latest t2)")
print(f"Median expected tc:     {median_tc.date()}  ({median_days:.1f} days after latest t2)")
print(f"5%-95% range:           {q05.date()} to {q95.date()}")
print(f"25%-75% range:          {q25.date()} to {q75.date()}")

# ============================================================
# KDE PLOT
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 34,
    "axes.labelsize": 34,
    "axes.titlesize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 26,
})

tc_dates = recent[TC_PRED_COL].dropna()

fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

if len(tc_dates) >= MIN_FITS_FOR_KDE and tc_dates.nunique() >= 2:
    x_num = mdates.date2num(tc_dates)

    kde = gaussian_kde(x_num)

    x_min = tc_dates.min() - pd.Timedelta(days=KDE_PAD_DAYS)
    x_max = tc_dates.max() + pd.Timedelta(days=KDE_PAD_DAYS)

    x_grid = pd.date_range(x_min, x_max, periods=KDE_GRID_POINTS)
    y = kde(mdates.date2num(x_grid))

    y[0] = 0
    y[-1] = 0

    ax.plot(
        x_grid,
        y,
        color="black",
        linewidth=KDE_LINEWIDTH,
        label="KDE density"
    )

    ax.fill_between(
        x_grid,
        y,
        color="black",
        alpha=0.12
    )

else:
    ax.hist(
        tc_dates,
        bins=min(10, len(tc_dates)),
        density=True,
        color="black",
        alpha=0.25,
        label="Histogram"
    )

ax.axvline(
    mean_tc,
    color="red",
    linewidth=MEAN_LINEWIDTH,
    linestyle="-",
    label=f"Mean: {mean_tc.date()}"
)

ax.axvline(
    median_tc,
    color="#0000cc",
    linewidth=MEDIAN_LINEWIDTH,
    linestyle="--",
    label=f"Median: {median_tc.date()}"
)

ax.axvline(
    latest_t2,
    color="gray",
    linewidth=2.0,
    linestyle=":",
    label=f"Latest $t_2$: {latest_t2.date()}"
)

ax.set_xlabel("Predicted future $t_c$")
ax.set_ylabel("Density")

ax.grid(True, which="major", alpha=0.30, linewidth=GRID_LINEWIDTH)
ax.minorticks_off()

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

ax.legend(frameon=False, loc="upper right")

plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PNG), dpi=300)
plt.show()

print(f"\nSaved plot: {os.path.join(OUTPUT_DIR, OUTPUT_PNG)}")
print(f"Saved predictions: {os.path.join(OUTPUT_DIR, OUTPUT_CSV)}")
print(f"Saved summary: {os.path.join(OUTPUT_DIR, OUTPUT_SUMMARY)}")