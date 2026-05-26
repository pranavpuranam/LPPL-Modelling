# 03b_tc_estimation.py

import os
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


# ============================================================
# STYLE / TUNING
# ============================================================

RED = "red"

# Moves ONLY the red "true tc" text up/down in each panel
# Positive = higher, negative = lower
TRUE_TC_TEXT_Y_OFFSETS = [0.06, 0.06, -0.20, 0.08, 0.16, 0.22]

# KDE support trimming:
# define "effectively zero" as below this fraction of the KDE peak
KDE_FADE_THRESHOLD_FRAC = 0.002

# if the true tc lands at or below this fraction of peak KDE density,
# draw a red dot on the x-axis instead of a vertical line
TRUE_TC_ZERO_THRESHOLD_FRAC = 0.002

# smoother support detection
KDE_GRID_POINTS = 2000

# pad around observed tc predictions when searching KDE support
KDE_SEARCH_PAD_DAYS = 180

# additional visual padding so KDE curve returns to x-axis
KDE_VISUAL_PAD_FRAC = 0.10
KDE_VISUAL_PAD_MIN_DAYS = 60

# minimum tc after t2 for "full possible predicted date" axis
TC_MIN_DAYS = 1

# fallback upper bound if global_tc_upper_date is unavailable
FALLBACK_TC_MAX_DAYS = 500


# ============================================================
# EVENT LABELS
# ============================================================

EVENT_LABELS = {
    pd.Timestamp("2007-06-01"): "Global Financial Crisis",
    pd.Timestamp("2011-02-17"): "Sovereign Debt Crisis",
    pd.Timestamp("2015-07-20"): "China Devaluation",
    pd.Timestamp("2018-01-23"): "Eurozone Equity Correction",
    pd.Timestamp("2020-02-19"): "COVID-19 Pandemic",
    pd.Timestamp("2022-01-05"): "Post-COVID Inflation Scare",
}


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

    prices = (
        prices
        .dropna(subset=[DATE_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    fits = (
        fits
        .dropna(subset=[T2_COL, TC_PRED_COL])
        .sort_values(T2_COL)
        .reset_index(drop=True)
    )

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
    event_col="tc_literature",
    n_panels=6,
    lookback_days=500,
    only_valid=True,
    min_fits=30,
    exclude_dates=None,
):
    """
    Select events where the number of valid tc estimates is STRICTLY greater than min_fits.

    With min_fits=30, this means:
        keep only events with N > 30
    """
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

        # STRICT FILTER:
        # Keep only N > min_fits.
        # Therefore, with min_fits=30, N=30 is rejected and N=31 is kept.
        if n <= min_fits:
            skipped.append((true_date, f"too few fits, N={n}; require N > {min_fits}"))
            continue

        selected.append(true_date)

        if len(selected) == n_panels:
            break

    return selected, skipped


# ============================================================
# HELPERS
# ============================================================

def get_event_title(true_date):
    true_date = pd.Timestamp(true_date).normalize()

    for dt, label in EVENT_LABELS.items():
        if pd.Timestamp(dt).normalize() == true_date:
            return label

    return true_date.strftime("%Y-%m-%d")


def get_full_possible_tc_bounds(
    subset,
    tc_min_days=TC_MIN_DAYS,
    fallback_tc_max_days=FALLBACK_TC_MAX_DAYS,
):
    """
    Full possible predicted-date range implied by the included fits:
    earliest possible tc = min(t2) + tc_min_days
    latest possible tc   = max(global_tc_upper_date), or fallback if unavailable
    """
    x_min = subset[T2_COL].min() + pd.Timedelta(days=tc_min_days)

    if TC_UPPER_COL in subset.columns and subset[TC_UPPER_COL].notna().any():
        x_max = subset[TC_UPPER_COL].max()
    else:
        x_max = subset[T2_COL].max() + pd.Timedelta(days=fallback_tc_max_days)

    return x_min, x_max


def get_kde_support_bounds(tc_dates):
    """
    Build KDE support bounds based on where density has effectively faded to zero.
    """
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
    """
    Final x-axis = broader of:
    (1) KDE support range
    (2) full possible predicted-date range

    Then rebuild the KDE plotting grid on the final axis range and force
    the first and last density values to zero so the plotted curve visibly
    returns to the x-axis.
    """
    kde, _, _, kde_left, kde_right, _ = get_kde_support_bounds(tc_dates)
    possible_left, possible_right = get_full_possible_tc_bounds(subset)

    final_left = min(kde_left, possible_left)
    final_right = max(kde_right, possible_right)

    # Add visual padding so the KDE has room to decay toward the x-axis
    span_days = max((final_right - final_left).days, 1)
    pad_days = max(
        int(KDE_VISUAL_PAD_FRAC * span_days),
        KDE_VISUAL_PAD_MIN_DAYS,
    )

    final_left = final_left - pd.Timedelta(days=pad_days)
    final_right = final_right + pd.Timedelta(days=pad_days)

    # Recompute the KDE curve over the final visible axis range
    x_grid = pd.date_range(final_left, final_right, periods=KDE_GRID_POINTS)
    x_grid_num = mdates.date2num(x_grid)
    y = kde(x_grid_num)

    # Force visual contact with the x-axis at both ends of the plotted curve
    y[0] = 0.0
    y[-1] = 0.0

    y_max = float(np.max(y))

    return kde, x_grid, y, final_left, final_right, y_max


# ============================================================
# PLOTTING
# ============================================================

def plot_tc_kde_panels(
    event_col="tc_literature",
    output_png="daily_fullscale/tc_estimation_kde_literature_Ngt30_3x2.png",
    n_panels=6,
    lookback_days=500,
    only_valid=True,
    min_fits=30,
    exclude_dates=None,
):
    prices, fits = load_data()

    event_dates, skipped = select_events_to_plot(
        prices=prices,
        fits=fits,
        event_col=event_col,
        n_panels=n_panels,
        lookback_days=lookback_days,
        only_valid=only_valid,
        min_fits=min_fits,
        exclude_dates=exclude_dates,
    )

    if len(event_dates) == 0:
        raise ValueError(f"No plottable events found with N > {min_fits}.")

    print(f"\nSelected events with N > {min_fits}:")
    for d in event_dates:
        subset = get_event_subset(
            fits=fits,
            true_date=d,
            lookback_days=lookback_days,
            only_valid=only_valid,
        )
        print("  ", d.strftime("%Y-%m-%d"), "|", get_event_title(d), "| N =", len(subset))

    print("\nSkipped events:")
    for d, reason in skipped:
        print("  ", d.strftime("%Y-%m-%d"), "|", reason)

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    })

    # Keep 2 rows x 3 columns.
    # If only 5 events pass N > 30, the sixth panel is blank.
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))
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

        kde, x_grid, y, x_left, x_right, y_max = get_combined_axis_bounds(
            subset,
            tc_dates,
        )

        ax.plot(x_grid, y, color="black", linewidth=1.8)

        y_true = float(kde(mdates.date2num(true_date)))
        is_true_tc_at_zero = y_true <= TRUE_TC_ZERO_THRESHOLD_FRAC * y_max

        if is_true_tc_at_zero:
            ax.scatter(
                true_date,
                0,
                color=RED,
                s=55,
                zorder=5,
                clip_on=False,
            )

            label_y = TRUE_TC_TEXT_Y_OFFSETS[i] * y_max
            label_y = max(0.03 * y_max, min(label_y, y_max * 1.08))

        else:
            ax.vlines(
                true_date,
                ymin=0,
                ymax=y_true,
                color=RED,
                linewidth=2.0,
            )

            label_y = y_true + TRUE_TC_TEXT_Y_OFFSETS[i] * y_max
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

        event_title = get_event_title(true_date)
        ax.set_title(f"{event_title} (N={n_fits})")

        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(bottom=0)

        # Y-axis label only on left column.
        # In a 2x3 grid, left-column panels are indices 0 and 3.
        if i in [0, 3]:
            ax.set_ylabel("Density")
        else:
            ax.set_ylabel("")

        # X-axis label only on bottom row.
        # In a 2x3 grid, bottom-row panels are indices 3, 4, 5.
        if i in [3, 4, 5]:
            ax.set_xlabel("Estimated $t_c$")
        else:
            ax.set_xlabel("")

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Turn off unused axes.
    # Example: if five events pass N > 30, panel index 5 is blank.
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
        # Add dates here if you want to exclude any events manually, e.g.
        # "2007-06-01",
        # "2020-02-19",
    ]

    plot_tc_kde_panels(
        event_col="tc_literature",
        output_png="daily_fullscale/tc_estimation_kde_literature_Ngt30_3x2.png",
        n_panels=6,
        lookback_days=500,
        only_valid=True,
        min_fits=30,
        exclude_dates=EXCLUDE_DATES,
    )