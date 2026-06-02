# 02a: plot stacked confidence indicators

import glob  # find files by pattern
import pandas as pd  # handle dataframes
import matplotlib.pyplot as plt  # plot charts
import matplotlib.dates as mdates  # format date axis

PRICE_CSV = "eurostoxx600_prices.csv"  # input price data
LONG_CONFIDENCE_CSV = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"  # long ci file
SHU_CONFIDENCE_GLOB = "shu_short/*positive_confidence*.csv"  # short ci file pattern
OUTPUT_PNG = "stacked_lppl_chart.png"  # plot output
PRICE_DATE_COL = "Date"  # price date column
PRICE_COL = "Close"  # price column
CONF_DATE_COL = "t2"  # confidence date column
CONF_COL = "positive_bubble_confidence"  # confidence column
PRICE_SMOOTH_WINDOW = 5  # price smoothing window
CONF_SMOOTH_WINDOW = 15  # confidence smoothing window
PRICE_COLOR = "black"  # price colour
LONG_CONF_COLOR = "red"  # long ci colour
SHU_CONF_COLOR = "#0000ce"  # short ci colour
PRICE_LINEWIDTH = 1.2  # price line width
CONF_LINEWIDTH = 1.4  # confidence line width
ZERO_VISUAL_OFFSET = 0.004  # visual zero offset
FIGSIZE = (12, 10)  # figure size

def find_file(pattern):
    matches = sorted(glob.glob(pattern))  # find matching files
    if not matches:  # check matches exist
        raise FileNotFoundError(f"No file found for pattern: {pattern}")  # stop if missing
    return matches[0]  # return first match

def load_price():
    prices = pd.read_csv(PRICE_CSV)  # load price data
    prices[PRICE_DATE_COL] = pd.to_datetime(prices[PRICE_DATE_COL], errors="coerce")  # parse dates
    prices[PRICE_COL] = pd.to_numeric(prices[PRICE_COL], errors="coerce")  # force numeric prices
    return (
        prices.dropna(subset=[PRICE_DATE_COL, PRICE_COL])
        .sort_values(PRICE_DATE_COL)
        .reset_index(drop=True)
    )  # clean and sort prices

def load_confidence(path):
    conf = pd.read_csv(path)  # load confidence data
    conf[CONF_DATE_COL] = pd.to_datetime(conf[CONF_DATE_COL], errors="coerce")  # parse dates
    conf[CONF_COL] = pd.to_numeric(conf[CONF_COL], errors="coerce")  # force numeric confidence
    return (
        conf.dropna(subset=[CONF_DATE_COL, CONF_COL])
        .sort_values(CONF_DATE_COL)
        .reset_index(drop=True)
    )  # clean and sort confidence

def prepare_panel(prices, conf, axis_start, axis_end):
    p = prices[
        (prices[PRICE_DATE_COL] >= axis_start) &
        (prices[PRICE_DATE_COL] <= axis_end)
    ].copy().reset_index(drop=True)  # slice prices to range
    c = conf[
        (conf[CONF_DATE_COL] >= axis_start) &
        (conf[CONF_DATE_COL] <= axis_end)
    ].copy().reset_index(drop=True)  # slice confidence to range
    conf_on_price_dates = (
        pd.merge(
            p[[PRICE_DATE_COL]].rename(columns={PRICE_DATE_COL: "Date"}),
            c[[CONF_DATE_COL, CONF_COL]].rename(columns={CONF_DATE_COL: "Date"}),
            on="Date",
            how="left"
        )
        .sort_values("Date")
        .reset_index(drop=True)
        .rename(columns={"Date": PRICE_DATE_COL})
    )  # align confidence to price dates
    first_conf_date = c[CONF_DATE_COL].min() if not c.empty else pd.NaT  # get first confidence date
    last_conf_date = c[CONF_DATE_COL].max() if not c.empty else pd.NaT  # get last confidence date
    conf_on_price_dates[CONF_COL] = conf_on_price_dates[CONF_COL].interpolate(
        method="linear",
        limit_direction="both"
    )  # interpolate confidence values
    if pd.notna(first_conf_date) and pd.notna(last_conf_date):  # check confidence range
        conf_on_price_dates.loc[
            (conf_on_price_dates[PRICE_DATE_COL] < first_conf_date) |
            (conf_on_price_dates[PRICE_DATE_COL] > last_conf_date),
            CONF_COL
        ] = pd.NA  # remove values outside confidence range
    p["price_plot"] = p[PRICE_COL].rolling(
        window=PRICE_SMOOTH_WINDOW,
        min_periods=1
    ).mean()  # smooth price
    conf_on_price_dates["conf_plot"] = conf_on_price_dates[CONF_COL].rolling(
        window=CONF_SMOOTH_WINDOW,
        min_periods=1
    ).mean()  # smooth confidence
    return p, conf_on_price_dates, first_conf_date  # return prepared panel

def plot_panel(
    ax_price,
    prices,
    conf_interp,
    axis_start,
    axis_end,
    conf_color,
    conf_label,
    add_undefined_prefix=False,
    first_conf_date=None
):
    ax_price.plot(
        prices[PRICE_DATE_COL],
        prices["price_plot"],
        color=PRICE_COLOR,
        linewidth=PRICE_LINEWIDTH,
        label="Price"
    )  # plot price
    ax_price.set_ylabel("Price", color="black")  # set price axis label
    ax_price.tick_params(axis="x", colors="black")  # set x tick colour
    ax_price.tick_params(axis="y", colors="black")  # set y tick colour
    ax_conf = ax_price.twinx()  # create confidence axis
    if add_undefined_prefix and pd.notna(first_conf_date):  # check undefined prefix
        pre_mask = conf_interp[PRICE_DATE_COL] < first_conf_date  # identify prefix dates
        if pre_mask.any():  # check prefix exists
            ax_conf.plot(
                conf_interp.loc[pre_mask, PRICE_DATE_COL],
                [ZERO_VISUAL_OFFSET] * pre_mask.sum(),
                color=conf_color,
                linewidth=CONF_LINEWIDTH,
                linestyle="--",
                zorder=5
            )  # plot dashed undefined prefix
        first_row = conf_interp.loc[conf_interp[PRICE_DATE_COL] == first_conf_date]  # get first confidence row
        if not first_row.empty:  # check first row exists
            first_val = first_row["conf_plot"].iloc[0]  # get first confidence value
            if pd.notna(first_val):  # check value exists
                ax_conf.plot(
                    [first_conf_date, first_conf_date],
                    [0, first_val],
                    color=conf_color,
                    linewidth=CONF_LINEWIDTH,
                    linestyle="--",
                    zorder=5
                )  # connect prefix to first value
    ax_conf.plot(
        conf_interp[PRICE_DATE_COL],
        conf_interp["conf_plot"],
        color=conf_color,
        linewidth=CONF_LINEWIDTH,
        label=conf_label,
        zorder=6
    )  # plot confidence
    ax_conf.set_ylabel("Confidence Score", color="black")  # set confidence label
    ax_conf.set_ylim(0, 1)  # set confidence limits
    ax_conf.set_yticks([0, 0.5, 1])  # set confidence ticks
    ax_conf.set_yticklabels(["0", "0.5", "1"])  # set confidence tick labels
    ax_conf.tick_params(axis="y", colors="black")  # set confidence tick colour
    ax_price.set_xlim(axis_start, axis_end)  # set x limits
    ax_price.margins(x=0)  # remove price x padding
    ax_conf.margins(x=0)  # remove confidence x padding
    ax_price.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.8)  # add grid
    ax_conf.grid(False)  # remove confidence grid
    ax_price.xaxis.set_major_locator(mdates.YearLocator(2))  # set two-year ticks
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # format years
    lines1, labels1 = ax_price.get_legend_handles_labels()  # get price legend
    lines2, labels2 = ax_conf.get_legend_handles_labels()  # get confidence legend
    ax_price.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        frameon=False
    )  # combine legends

def main():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "path.simplify": True,
        "path.simplify_threshold": 1.0,
    })  # set plot style
    shu_confidence_csv = find_file(SHU_CONFIDENCE_GLOB)  # find short ci file
    prices = load_price()  # load price data
    long_conf = load_confidence(LONG_CONFIDENCE_CSV)  # load long ci
    shu_conf = load_confidence(shu_confidence_csv)  # load short ci
    axis_start = shu_conf[CONF_DATE_COL].min()  # set shared start
    axis_end = shu_conf[CONF_DATE_COL].max()  # set shared end
    long_prices, long_conf_interp, long_first_conf_date = prepare_panel(
        prices, long_conf, axis_start, axis_end
    )  # prepare long ci panel
    shu_prices, shu_conf_interp, shu_first_conf_date = prepare_panel(
        prices, shu_conf, axis_start, axis_end
    )  # prepare short ci panel
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=FIGSIZE,
        sharex=False
    )  # create stacked figure
    plot_panel(
        axes[0],
        long_prices,
        long_conf_interp,
        axis_start,
        axis_end,
        conf_color=LONG_CONF_COLOR,
        conf_label="Long CI",
        add_undefined_prefix=True,
        first_conf_date=long_first_conf_date
    )  # plot long ci panel
    plot_panel(
        axes[1],
        shu_prices,
        shu_conf_interp,
        axis_start,
        axis_end,
        conf_color=SHU_CONF_COLOR,
        conf_label="Shu Short CI",
        add_undefined_prefix=False,
        first_conf_date=shu_first_conf_date
    )  # plot short ci panel
    axes[1].set_xlabel("Date", color="black")  # set bottom x label
    plt.tight_layout()  # tidy layout
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")  # save plot
    plt.show()  # show plot
    print("Saved plot to:", OUTPUT_PNG)  # print output path
    print("Long confidence file:", LONG_CONFIDENCE_CSV)  # print long ci path
    print("Shu short confidence file:", shu_confidence_csv)  # print short ci path
    print("Shared range:", axis_start.date(), "to", axis_end.date())  # print shared range

if __name__ == "__main__":
    main()  # run script