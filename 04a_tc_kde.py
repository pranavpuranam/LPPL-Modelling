# 04a: plot critical time kde panels

import os  # handle folders
import pandas as pd  # handle dataframes
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plot charts
import matplotlib.dates as mdates  # format date axis
from scipy.stats import gaussian_kde  # estimate kde density

PRICE_CSV = "eurostoxx600_prices.csv"  # input price data
FITS_CSV = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"  # input fit data
OUTPUT_DIR = "daily_fullscale"  # output folder
DATE_COL = "Date"  # price date column
TC_PRED_COL = "tc_predicted"  # predicted tc column
VALID_COL = "positive_lppls_valid"  # validity column
T2_COL = "t2"  # endpoint column
TC_UPPER_COL = "global_tc_upper_date"  # tc upper bound column
RED = "red"  # true tc colour
TRUE_TC_TEXT_Y_OFFSETS = [0.06, 0.06, -0.20, 0.08, 0.16, 0.22]  # label offsets
KDE_FADE_THRESHOLD_FRAC = 0.002  # kde support threshold
TRUE_TC_ZERO_THRESHOLD_FRAC = 0.002  # true tc zero threshold
KDE_GRID_POINTS = 2000  # kde grid size
KDE_SEARCH_PAD_DAYS = 180  # kde search padding
KDE_VISUAL_PAD_FRAC = 0.10  # visual padding fraction
KDE_VISUAL_PAD_MIN_DAYS = 60  # minimum visual padding
TC_MIN_DAYS = 1  # minimum tc after t2
FALLBACK_TC_MAX_DAYS = 500  # fallback tc max

EVENT_LABELS = {
    pd.Timestamp("2007-06-01"): "Global Financial Crisis",
    pd.Timestamp("2011-02-17"): "Sovereign Debt Crisis",
    pd.Timestamp("2015-07-20"): "China Devaluation",
    pd.Timestamp("2018-01-23"): "Eurozone Equity Correction",
    pd.Timestamp("2020-02-19"): "COVID-19 Pandemic",
    pd.Timestamp("2022-01-05"): "Post-COVID Inflation Scare",
}  # event name map

def load_data():
    prices = pd.read_csv(PRICE_CSV)  # load prices
    fits = pd.read_csv(FITS_CSV)  # load fits
    prices[DATE_COL] = pd.to_datetime(prices[DATE_COL], errors="coerce")  # parse price dates
    fits[T2_COL] = pd.to_datetime(fits[T2_COL], errors="coerce")  # parse endpoints
    fits[TC_PRED_COL] = pd.to_datetime(fits[TC_PRED_COL], errors="coerce")  # parse predicted tc
    if TC_UPPER_COL in fits.columns:  # check upper bound column
        fits[TC_UPPER_COL] = pd.to_datetime(fits[TC_UPPER_COL], errors="coerce")  # parse upper bounds
    fits[VALID_COL] = (
        fits[VALID_COL]
        .astype(str)
        .str.lower()
        .eq("true")
    )  # convert validity flag
    prices = (
        prices
        .dropna(subset=[DATE_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )  # clean prices
    fits = (
        fits
        .dropna(subset=[T2_COL, TC_PRED_COL])
        .sort_values(T2_COL)
        .reset_index(drop=True)
    )  # clean fits
    return prices, fits  # return datasets

def get_event_subset(fits, true_date, lookback_days=500, only_valid=True):
    subset = fits[
        (fits[T2_COL] >= true_date - pd.Timedelta(days=lookback_days)) &
        (fits[T2_COL] <= true_date)
    ].copy()  # keep pre-event fits
    if only_valid:  # check valid-only setting
        subset = subset[subset[VALID_COL]]  # keep valid fits
    subset = subset.dropna(subset=[TC_PRED_COL])  # remove missing predictions
    return subset  # return event subset

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
    if exclude_dates is None:  # check exclusion list
        exclude_dates = []  # set empty list
    exclude_dates = set(pd.to_datetime(exclude_dates))  # convert exclusions
    raw_event_dates = (
        prices.loc[prices[event_col] == 1, DATE_COL]
        .dropna()
        .sort_values()
        .tolist()
    )  # get event dates
    selected = []  # store selected events
    skipped = []  # store skipped events
    for true_date in raw_event_dates:  # loop through events
        if true_date in exclude_dates:  # check manual exclusion
            skipped.append((true_date, "manually excluded"))  # store exclusion
            continue  # skip event
        subset = get_event_subset(
            fits=fits,
            true_date=true_date,
            lookback_days=lookback_days,
            only_valid=only_valid,
        )  # get event fits
        n = len(subset)  # count fits
        if n <= min_fits:  # require enough fits
            skipped.append((true_date, f"too few fits, N={n}; require N > {min_fits}"))  # store skip reason
            continue  # skip event
        selected.append(true_date)  # keep event
        if len(selected) == n_panels:  # check panel limit
            break  # stop selection
    return selected, skipped  # return selected and skipped events

def get_event_title(true_date):
    true_date = pd.Timestamp(true_date).normalize()  # normalise date
    for dt, label in EVENT_LABELS.items():  # loop through labels
        if pd.Timestamp(dt).normalize() == true_date:  # match date
            return label  # return event label
    return true_date.strftime("%Y-%m-%d")  # fallback to date

def get_full_possible_tc_bounds(
    subset,
    tc_min_days=TC_MIN_DAYS,
    fallback_tc_max_days=FALLBACK_TC_MAX_DAYS,
):
    x_min = subset[T2_COL].min() + pd.Timedelta(days=tc_min_days)  # earliest possible tc
    if TC_UPPER_COL in subset.columns and subset[TC_UPPER_COL].notna().any():  # check upper bounds
        x_max = subset[TC_UPPER_COL].max()  # latest possible tc
    else:
        x_max = subset[T2_COL].max() + pd.Timedelta(days=fallback_tc_max_days)  # fallback latest tc
    return x_min, x_max  # return bounds

def get_kde_support_bounds(tc_dates):
    x_num = mdates.date2num(tc_dates)  # convert dates to numbers
    kde = gaussian_kde(x_num)  # fit kde
    tc_min = tc_dates.min()  # earliest predicted tc
    tc_max = tc_dates.max()  # latest predicted tc
    search_min = tc_min - pd.Timedelta(days=KDE_SEARCH_PAD_DAYS)  # search lower bound
    search_max = tc_max + pd.Timedelta(days=KDE_SEARCH_PAD_DAYS)  # search upper bound
    x_grid = pd.date_range(search_min, search_max, periods=KDE_GRID_POINTS)  # create kde grid
    x_grid_num = mdates.date2num(x_grid)  # convert grid to numbers
    y = kde(x_grid_num)  # evaluate kde
    y_max = float(np.max(y))  # get max density
    threshold = KDE_FADE_THRESHOLD_FRAC * y_max  # set fade threshold
    mask = y >= threshold  # find nonzero support
    if mask.any():  # check support exists
        first_idx = int(np.argmax(mask))  # first support index
        last_idx = int(len(mask) - 1 - np.argmax(mask[::-1]))  # last support index
        x_left = x_grid[first_idx]  # support left
        x_right = x_grid[last_idx]  # support right
        span_days = max((x_right - x_left).days, 1)  # support span
        margin_days = max(int(0.03 * span_days), 10)  # support margin
        x_left = x_left - pd.Timedelta(days=margin_days)  # pad left
        x_right = x_right + pd.Timedelta(days=margin_days)  # pad right
    else:
        x_left = search_min  # fallback left
        x_right = search_max  # fallback right
    return kde, x_grid, y, x_left, x_right, y_max  # return kde support

def get_combined_axis_bounds(subset, tc_dates):
    kde, _, _, kde_left, kde_right, _ = get_kde_support_bounds(tc_dates)  # get kde support
    possible_left, possible_right = get_full_possible_tc_bounds(subset)  # get possible tc range
    final_left = min(kde_left, possible_left)  # combine left bound
    final_right = max(kde_right, possible_right)  # combine right bound
    span_days = max((final_right - final_left).days, 1)  # calculate span
    pad_days = max(
        int(KDE_VISUAL_PAD_FRAC * span_days),
        KDE_VISUAL_PAD_MIN_DAYS,
    )  # calculate plot padding
    final_left = final_left - pd.Timedelta(days=pad_days)  # pad left
    final_right = final_right + pd.Timedelta(days=pad_days)  # pad right
    x_grid = pd.date_range(final_left, final_right, periods=KDE_GRID_POINTS)  # create final grid
    x_grid_num = mdates.date2num(x_grid)  # convert grid to numbers
    y = kde(x_grid_num)  # evaluate kde
    y[0] = 0.0  # force curve to zero
    y[-1] = 0.0  # force curve to zero
    y_max = float(np.max(y))  # get max density
    return kde, x_grid, y, final_left, final_right, y_max  # return final axis data

def plot_tc_kde_panels(
    event_col="tc_literature",
    output_png="daily_fullscale/tc_estimation_kde_literature_Ngt30_3x2.png",
    n_panels=6,
    lookback_days=500,
    only_valid=True,
    min_fits=30,
    exclude_dates=None,
):
    prices, fits = load_data()  # load data
    event_dates, skipped = select_events_to_plot(
        prices=prices,
        fits=fits,
        event_col=event_col,
        n_panels=n_panels,
        lookback_days=lookback_days,
        only_valid=only_valid,
        min_fits=min_fits,
        exclude_dates=exclude_dates,
    )  # select plottable events
    if len(event_dates) == 0:  # check selected events
        raise ValueError(f"No plottable events found with N > {min_fits}.")  # stop if none
    print(f"\nSelected events with N > {min_fits}:")  # print selected header
    for d in event_dates:  # loop through selected events
        subset = get_event_subset(
            fits=fits,
            true_date=d,
            lookback_days=lookback_days,
            only_valid=only_valid,
        )  # get event subset
        print("  ", d.strftime("%Y-%m-%d"), "|", get_event_title(d), "| N =", len(subset))  # print event
    print("\nSkipped events:")  # print skipped header
    for d, reason in skipped:  # loop through skipped events
        print("  ", d.strftime("%Y-%m-%d"), "|", reason)  # print skip reason
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 22,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
    })  # set plot style
    fig, axes = plt.subplots(2, 3, figsize=(21, 11))  # create panel grid
    axes = axes.ravel()  # flatten axes
    for i, (ax, true_date) in enumerate(zip(axes, event_dates)):  # loop through panels
        subset = get_event_subset(
            fits=fits,
            true_date=true_date,
            lookback_days=lookback_days,
            only_valid=only_valid,
        )  # get event subset
        n_fits = len(subset)  # count fits
        tc_dates = subset[TC_PRED_COL].dropna()  # get predicted tc dates
        ax.set_box_aspect(0.75)  # set panel aspect
        kde, x_grid, y, x_left, x_right, y_max = get_combined_axis_bounds(
            subset,
            tc_dates,
        )  # build kde curve
        ax.plot(x_grid, y, color="black", linewidth=1.8)  # plot kde
        y_true = float(kde(mdates.date2num(true_date)))  # get density at true tc
        is_true_tc_at_zero = y_true <= TRUE_TC_ZERO_THRESHOLD_FRAC * y_max  # check near-zero true density
        if is_true_tc_at_zero:  # if true tc near zero
            ax.scatter(
                true_date,
                0,
                color=RED,
                s=55,
                zorder=5,
                clip_on=False,
            )  # mark true tc point
            label_y = TRUE_TC_TEXT_Y_OFFSETS[i] * y_max  # set label y
            label_y = max(0.03 * y_max, min(label_y, y_max * 1.08))  # bound label y
        else:
            ax.vlines(
                true_date,
                ymin=0,
                ymax=y_true,
                color=RED,
                linewidth=2.0,
            )  # draw true tc line
            label_y = y_true + TRUE_TC_TEXT_Y_OFFSETS[i] * y_max  # set label y
            label_y = max(0, min(label_y, y_max * 1.08))  # bound label y
        ax.text(
            true_date,
            label_y,
            "true $t_c$",
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=18,
            color=RED,
        )  # add true tc label
        event_title = get_event_title(true_date)  # get event title
        ax.set_title(f"{event_title} (N={n_fits})")  # set panel title
        ax.grid(True, alpha=0.25)  # add grid
        ax.set_xlim(x_left, x_right)  # set x limits
        ax.set_ylim(bottom=0)  # set y lower bound
        if i in [0, 3]:  # check left column
            ax.set_ylabel("Density")  # set y label
        else:
            ax.set_ylabel("")  # remove y label
        if i in [3, 4, 5]:  # check bottom row
            ax.set_xlabel("Estimated $t_c$")  # set x label
        else:
            ax.set_xlabel("")  # remove x label
        ax.xaxis.set_major_locator(mdates.YearLocator())  # set yearly ticks
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # format years
    for ax in axes[len(event_dates):]:  # loop over unused axes
        ax.axis("off")  # hide unused panels
    plt.tight_layout()  # tidy layout
    os.makedirs(os.path.dirname(output_png), exist_ok=True)  # create output folder
    plt.savefig(output_png, dpi=300, bbox_inches="tight")  # save plot
    plt.show()  # show plot
    print(f"\nSaved: {output_png}")  # print save path

if __name__ == "__main__":
    EXCLUDE_DATES = [
    ]  # manually excluded dates
    plot_tc_kde_panels(
        event_col="tc_literature",
        output_png="daily_fullscale/tc_estimation_kde_literature_Ngt30_3x2.png",
        n_panels=6,
        lookback_days=500,
        only_valid=True,
        min_fits=30,
        exclude_dates=EXCLUDE_DATES,
    )  # plot kde panels