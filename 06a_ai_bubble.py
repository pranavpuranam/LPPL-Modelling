# 06a: plot recent future critical time distribution

import os  # handle folders
import pandas as pd  # handle dataframes
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plot charts
import matplotlib.dates as mdates  # format date axis
from scipy.stats import gaussian_kde  # estimate kde density

FITS_CSV = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"  # lppl fit input
OUTPUT_DIR = "daily_fullscale"  # output folder
OUTPUT_PNG = "recent_750d_future_tc_distribution.png"  # plot output
OUTPUT_CSV = "recent_750d_future_tc_predictions.csv"  # prediction output
OUTPUT_SUMMARY = "recent_750d_future_tc_summary.csv"  # summary output
T2_COL = "t2"  # endpoint column
TC_PRED_COL = "tc_predicted"  # predicted tc column
VALID_COL = "positive_lppls_valid"  # validity column
LOOKBACK_DAYS = 750  # recent lookback window
VALID_ONLY = True  # keep only valid fits
FUTURE_ONLY = True  # keep only future tc estimates
MIN_FITS_FOR_KDE = 5  # minimum kde fits
FIGSIZE = (17.76, 9.95)  # figure size
KDE_LINEWIDTH = 3.2  # kde line width
MEAN_LINEWIDTH = 2.8  # mean line width
MEDIAN_LINEWIDTH = 2.8  # median line width
GRID_LINEWIDTH = 1.2  # grid line width
KDE_GRID_POINTS = 2000  # kde grid size
KDE_PAD_DAYS = 180  # kde date padding

fits = pd.read_csv(FITS_CSV)  # load lppl fits
fits[T2_COL] = pd.to_datetime(fits[T2_COL], errors="coerce")  # parse endpoints
fits[TC_PRED_COL] = pd.to_datetime(fits[TC_PRED_COL], errors="coerce")  # parse predicted tc
fits[VALID_COL] = (
    fits[VALID_COL]
    .astype(str)
    .str.lower()
    .eq("true")
)  # convert validity flag

fits = fits.dropna(subset=[T2_COL, TC_PRED_COL]).copy()  # remove missing dates
fits = fits.sort_values(T2_COL).reset_index(drop=True)  # sort by endpoint
latest_t2 = fits[T2_COL].max()  # get latest endpoint
start_date = latest_t2 - pd.Timedelta(days=LOOKBACK_DAYS)  # set lookback start

recent = fits[
    (fits[T2_COL] >= start_date) &
    (fits[T2_COL] <= latest_t2)
].copy()  # keep recent fits

if VALID_ONLY:  # check valid-only setting
    recent = recent[recent[VALID_COL]].copy()  # keep valid fits

if FUTURE_ONLY:  # check future-only setting
    recent = recent[recent[TC_PRED_COL] > latest_t2].copy()  # keep future tc estimates

if len(recent) == 0:  # check fits remain
    raise ValueError("No recent future tc predictions found after filtering.")  # stop if empty

recent["tc_days_from_latest_t2"] = (
    recent[TC_PRED_COL] - latest_t2
).dt.days  # calculate days from latest endpoint

mean_tc = recent[TC_PRED_COL].mean()  # calculate mean tc
median_tc = recent[TC_PRED_COL].median()  # calculate median tc
mean_days = recent["tc_days_from_latest_t2"].mean()  # calculate mean days ahead
median_days = recent["tc_days_from_latest_t2"].median()  # calculate median days ahead
q05 = recent[TC_PRED_COL].quantile(0.05)  # calculate 5th percentile
q25 = recent[TC_PRED_COL].quantile(0.25)  # calculate 25th percentile
q75 = recent[TC_PRED_COL].quantile(0.75)  # calculate 75th percentile
q95 = recent[TC_PRED_COL].quantile(0.95)  # calculate 95th percentile

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
}])  # create summary table

os.makedirs(OUTPUT_DIR, exist_ok=True)  # create output folder
recent.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV), index=False)  # save predictions
summary.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_SUMMARY), index=False)  # save summary

print("\nRecent future tc distribution")  # print title
print("--------------------------------")  # print divider
print(f"Latest t2:              {latest_t2.date()}")  # print latest endpoint
print(f"Lookback start:         {start_date.date()}")  # print lookback start
print(f"Fits used:              {len(recent)}")  # print fit count
print(f"Mean expected tc:       {mean_tc.date()}  ({mean_days:.1f} days after latest t2)")  # print mean tc
print(f"Median expected tc:     {median_tc.date()}  ({median_days:.1f} days after latest t2)")  # print median tc
print(f"5%-95% range:           {q05.date()} to {q95.date()}")  # print wide range
print(f"25%-75% range:          {q25.date()} to {q75.date()}")  # print iqr range

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 34,
    "axes.labelsize": 34,
    "axes.titlesize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 26,
})  # set plot style

tc_dates = recent[TC_PRED_COL].dropna()  # get predicted tc dates
fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)  # create figure

if len(tc_dates) >= MIN_FITS_FOR_KDE and tc_dates.nunique() >= 2:  # check kde possible
    x_num = mdates.date2num(tc_dates)  # convert dates to numbers
    kde = gaussian_kde(x_num)  # fit kde
    x_min = tc_dates.min() - pd.Timedelta(days=KDE_PAD_DAYS)  # set lower grid date
    x_max = tc_dates.max() + pd.Timedelta(days=KDE_PAD_DAYS)  # set upper grid date
    x_grid = pd.date_range(x_min, x_max, periods=KDE_GRID_POINTS)  # create kde grid
    y = kde(mdates.date2num(x_grid))  # evaluate kde
    y[0] = 0  # force curve to zero
    y[-1] = 0  # force curve to zero
    ax.plot(
        x_grid,
        y,
        color="black",
        linewidth=KDE_LINEWIDTH,
        label="KDE density"
    )  # plot kde curve
    ax.fill_between(
        x_grid,
        y,
        color="black",
        alpha=0.12
    )  # shade kde area
else:
    ax.hist(
        tc_dates,
        bins=min(10, len(tc_dates)),
        density=True,
        color="black",
        alpha=0.25,
        label="Histogram"
    )  # plot histogram fallback

ax.axvline(
    mean_tc,
    color="red",
    linewidth=MEAN_LINEWIDTH,
    linestyle="-",
    label=f"Mean: {mean_tc.date()}"
)  # plot mean tc

ax.axvline(
    median_tc,
    color="#0000cc",
    linewidth=MEDIAN_LINEWIDTH,
    linestyle="--",
    label=f"Median: {median_tc.date()}"
)  # plot median tc

ax.axvline(
    latest_t2,
    color="gray",
    linewidth=2.0,
    linestyle=":",
    label=f"Latest $t_2$: {latest_t2.date()}"
)  # plot latest endpoint

ax.set_xlabel("Predicted future $t_c$")  # set x label
ax.set_ylabel("Density")  # set y label
ax.grid(True, which="major", alpha=0.30, linewidth=GRID_LINEWIDTH)  # add grid
ax.minorticks_off()  # remove minor ticks
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # set three-month ticks
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))  # format date labels
ax.legend(frameon=False, loc="upper right")  # add legend

plt.savefig(os.path.join(OUTPUT_DIR, OUTPUT_PNG), dpi=300)  # save plot
plt.show()  # show plot

print(f"\nSaved plot: {os.path.join(OUTPUT_DIR, OUTPUT_PNG)}")  # print plot output
print(f"Saved predictions: {os.path.join(OUTPUT_DIR, OUTPUT_CSV)}")  # print prediction output
print(f"Saved summary: {os.path.join(OUTPUT_DIR, OUTPUT_SUMMARY)}")  # print summary output