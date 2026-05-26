# 03b_tc_estimation_drawdown_5plots.py

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde


# ============================================================
# FILE PATHS
# ============================================================

PRICE_CSV = "eurostoxx600_prices.csv"
FITS_CSV = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"
OUTPUT_DIR = "daily_fullscale"


# ============================================================
# COLUMN NAMES
# ============================================================

DATE_COL = "Date"
TC_PRED_COL = "tc_predicted"
VALID_COL = "positive_lppls_valid"
T2_COL = "t2"
TC_UPPER_COL = "global_tc_upper_date"

# Use drawdown critical-time labels from eurostoxx600_prices.csv
EVENT_COL = "tc_drawdown"


# ============================================================
# STYLE / TUNING
# ============================================================

RED = "red"

TRUE_TC_TEXT_Y_OFFSETS = [0.06, 0.06, -0.20, 0.08, 0.06]

KDE_FADE_THRESHOLD_FRAC = 0.002
TRUE_TC_ZERO_THRESHOLD_FRAC = 0.002
KDE_GRID_POINTS = 2000
KDE_SEARCH_PAD_DAYS = 180

TC_MIN_DAYS = 1
FALLBACK_TC_MAX_DAYS = 500

# Exactly 5 plots: 3 on first row, 2 on second row
N_PLOTS = 5
N_ROWS = 2
N_COLS = 3
PANEL_WIDTH = 6
PANEL_HEIGHT = 5


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    prices = pd.read_csv(PRICE_CSV)
    fits = pd.read_csv(FITS_CSV)

    prices[DATE_COL] = pd.to_datetime(prices[DATE_COL], errors="coerce")
    fits[T2_COL] = pd.to_datetime(fits[T2_COL], errors="coerce")
    fits[TC_PRED_COL] = pd.to_datetime(fits[TC_PRED_COL], errors="coerce")

    if TC_UPPER_COL in fits.columns:
        fits[TC_UPPER_COL] = pd.to_datetime(fits[TC_UPPER_COL], errors="coerce")

    fits[VALID_COL] = (
        fits[VALID_COL]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    prices = prices.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
    fits = fits.dropna(subset=[T2_COL, TC_PRED_COL]).sort_values(T2_COL).reset_index(drop=True)

    if EVENT_COL not in prices.columns:
        raise ValueError(f"Could not find event column '{EVENT_COL}' in {PRICE_CSV}")

    return prices, fits


# ============================================================
# EVENT SUBSETTING
# ============================================================

def get_event_subset(fits, true_date, lookback_days=500, only_valid=True):
    subset = fits[
        (fits[T2_COL] >= true_date - pd.Timedelta(days=lookback_days)) &
        (fits[T2_COL] <= true_date)
    ].copy()

    if only_valid:
        subset = subset[subset[VALID_COL]]

    subset = subset.dropna(subset=[TC_PRED_COL])

    return subset


def select_events_to_plot(
    prices,
    fits,
    event_col=EVENT_COL,
    n_plots=N_PLOTS,
    lookback_days=500,
    only_valid=True,
    min_fits=5,
    exclude_dates=None,
):
    if exclude_dates is None:
        exclude_dates = []

    exclude_dates = set(pd.to_datetime(exclude_dates))

    raw_event_dates = (
        prices.loc[prices[event_col] == 1, DATE_COL]
        .dropna()
        .sort_values()
        .tolist()
    )

    selected = []
    skipped = []

    for true_date in raw_event_dates:
        if true_date in exclude_dates:
            skipped.append((true_date, "manually excluded"))
            continue

        subset = get_event_subset(
            fits=fits,
            true_date=true_date,
            lookback_days=lookback_days,
            only_valid=only_valid,
        )

        n = len(subset)

        if n < min_fits:
            skipped.append((true_date, f"too few fits, N={n}"))
            continue

        selected.append(true_date)

        if len(selected) == n_plots:
            break

    return selected, skipped


# ============================================================
# HELPERS
# ============================================================

def get_full_possible_tc_bounds(
    subset,
    tc_min_days=TC_MIN_DAYS,
    fallback_tc_max_days=FALLBACK_TC_MAX_DAYS,
):
    x_min = subset[T2_COL].min() + pd.Timedelta(days=tc_min_days)

    if TC_UPPER_COL in subset.columns and subset[TC_UPPER_COL].notna().any():
        x_max = subset[TC_UPPER_COL].max()
    else:
        x_max = subset[T2_COL].max() + pd.Timedelta(days=fallback_tc_max_days)

    return x_min, x_max


def get_kde_support_bounds(tc_dates):
    x_num = mdates.date2num(tc_dates)
    kde = gaussian_kde(x_num)

    tc_min = tc_dates.min()
    tc_max = tc_dates.max()

    search_min = tc_min - pd.Timedelta(days=KDE_SEARCH_PAD_DAYS)
    search_max = tc_max + pd.Timedelta(days=KDE_SEARCH_PAD_DAYS)

    x_grid = pd.date_range(search_min, search_max, periods=KDE_GRID_POINTS)
    x_grid_num = mdates.date2num(x_grid)
    y = kde(x_grid_num)

    y_max = float(np.max(y))
    threshold = KDE_FADE_THRESHOLD_FRAC * y_max

    mask = y >= threshold

    if mask.any():
        first_idx = int(np.argmax(mask))
        last_idx = int(len(mask) - 1 - np.argmax(mask[::-1]))

        x_left = x_grid[first_idx]
        x_right = x_grid[last_idx]

        span_days = max((x_right - x_left).days, 1)
        margin_days = max(int(0.03 * span_days), 10)

        x_left = x_left - pd.Timedelta(days=margin_days)
        x_right = x_right + pd.Timedelta(days=margin_days)
    else:
        x_left = search_min
        x_right = search_max

    return kde, x_grid, y, x_left, x_right, y_max


def get_combined_axis_bounds(subset, tc_dates):
    kde, x_grid, y, kde_left, kde_right, y_max = get_kde_support_bounds(tc_dates)
    possible_left, possible_right = get_full_possible_tc_bounds(subset)

    final_left = min(kde_left, possible_left)
    final_right = max(kde_right, possible_right)

    return kde, x_grid, y, final_left, final_right, y_max


# ============================================================
# PLOTTING
# ============================================================

def plot_tc_kde_panels(
    event_col=EVENT_COL,
    output_png="daily_fullscale/tc_estimation_kde_drawdown_5plots.png",
    n_plots=N_PLOTS,
    lookback_days=500,
    only_valid=True,
    min_fits=5,
    exclude_dates=None,
):
    prices, fits = load_data()

    event_dates, skipped = select_events_to_plot(
        prices=prices,
        fits=fits,
        event_col=event_col,
        n_plots=n_plots,
        lookback_days=lookback_days,
        only_valid=only_valid,
        min_fits=min_fits,
        exclude_dates=exclude_dates,
    )

    if len(event_dates) == 0:
        raise ValueError(f"No plottable events found for event_col='{event_col}'.")

    print("\nSelected drawdown events:")
    for d in event_dates:
        print("  ", d.strftime("%Y-%m-%d"))

    print("\nSkipped drawdown events:")
    for d, reason in skipped:
        print("  ", d.strftime("%Y-%m-%d"), "|", reason)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
    })

    fig, axes = plt.subplots(
        N_ROWS,
        N_COLS,
        figsize=(PANEL_WIDTH * N_COLS, PANEL_HEIGHT * N_ROWS),
        squeeze=False,
    )

    axes = axes.ravel()

    for i, (ax, true_date) in enumerate(zip(axes, event_dates)):
        subset = get_event_subset(
            fits=fits,
            true_date=true_date,
            lookback_days=lookback_days,
            only_valid=only_valid,
        )

        n_fits = len(subset)
        tc_dates = subset[TC_PRED_COL].dropna()

        ax.set_box_aspect(0.75)

        kde, x_grid, y, x_left, x_right, y_max = get_combined_axis_bounds(subset, tc_dates)

        ax.plot(x_grid, y, color="black", linewidth=1.8)

        y_true = float(kde(mdates.date2num(true_date)))
        is_true_tc_at_zero = y_true <= TRUE_TC_ZERO_THRESHOLD_FRAC * y_max

        offset = TRUE_TC_TEXT_Y_OFFSETS[i % len(TRUE_TC_TEXT_Y_OFFSETS)]

        if is_true_tc_at_zero:
            ax.scatter(
                true_date,
                0,
                color=RED,
                s=55,
                zorder=5,
                clip_on=False,
            )

            label_y = offset * y_max
            label_y = max(0.03 * y_max, min(label_y, y_max * 1.08))

        else:
            ax.vlines(
                true_date,
                ymin=0,
                ymax=y_true,
                color=RED,
                linewidth=2.0,
            )

            label_y = y_true + offset * y_max
            label_y = max(0, min(label_y, y_max * 1.08))

        ax.text(
            true_date,
            label_y,
            "true $t_c$",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=18,
            color=RED,
        )

        # Date only, no event-name text
        ax.set_title(true_date.strftime("%Y-%m-%d"))

        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(bottom=0)

        if i in [0, 3]:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")

        if i in [3, 4]:
            ax.set_xlabel("Estimated $t_c$")
        else:
            ax.set_xlabel("")

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Hide the unused 6th panel
    for ax in axes[len(event_dates):]:
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nSaved: {output_png}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    EXCLUDE_DATES = [
        # Add dates here if you want to exclude any drawdown events manually
    ]

    plot_tc_kde_panels(
        event_col="tc_drawdown",
        output_png="daily_fullscale/tc_estimation_kde_drawdown_5plots.png",
        n_plots=5,
        lookback_days=500,
        only_valid=True,
        min_fits=5,
        exclude_dates=EXCLUDE_DATES,
    )