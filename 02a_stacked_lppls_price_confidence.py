import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PRICE_CSV = "eurostoxx600_prices.csv"

LONG_CONFIDENCE_CSV = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"
SHU_CONFIDENCE_GLOB = "shu_short/*positive_confidence*.csv"

OUTPUT_PNG = "stacked_lppl_chart.png"

PRICE_DATE_COL = "Date"
PRICE_COL = "Close"

CONF_DATE_COL = "t2"
CONF_COL = "positive_bubble_confidence"

PRICE_SMOOTH_WINDOW = 5
CONF_SMOOTH_WINDOW = 15

PRICE_COLOR = "black"
LONG_CONF_COLOR = "red"
SHU_CONF_COLOR = "#0000ce"

PRICE_LINEWIDTH = 1.2
CONF_LINEWIDTH = 1.4

# Tiny lift so the dashed zero-line is visible above the x-axis
ZERO_VISUAL_OFFSET = 0.004

# Slightly taller version of same chart
FIGSIZE = (12, 10)


def find_file(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return matches[0]


def load_price():
    prices = pd.read_csv(PRICE_CSV)
    prices[PRICE_DATE_COL] = pd.to_datetime(prices[PRICE_DATE_COL], errors="coerce")
    prices[PRICE_COL] = pd.to_numeric(prices[PRICE_COL], errors="coerce")

    return (
        prices.dropna(subset=[PRICE_DATE_COL, PRICE_COL])
        .sort_values(PRICE_DATE_COL)
        .reset_index(drop=True)
    )


def load_confidence(path):
    conf = pd.read_csv(path)
    conf[CONF_DATE_COL] = pd.to_datetime(conf[CONF_DATE_COL], errors="coerce")
    conf[CONF_COL] = pd.to_numeric(conf[CONF_COL], errors="coerce")

    return (
        conf.dropna(subset=[CONF_DATE_COL, CONF_COL])
        .sort_values(CONF_DATE_COL)
        .reset_index(drop=True)
    )


def prepare_panel(prices, conf, axis_start, axis_end):
    p = prices[
        (prices[PRICE_DATE_COL] >= axis_start) &
        (prices[PRICE_DATE_COL] <= axis_end)
    ].copy().reset_index(drop=True)

    c = conf[
        (conf[CONF_DATE_COL] >= axis_start) &
        (conf[CONF_DATE_COL] <= axis_end)
    ].copy().reset_index(drop=True)

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
    )

    first_conf_date = c[CONF_DATE_COL].min() if not c.empty else pd.NaT
    last_conf_date = c[CONF_DATE_COL].max() if not c.empty else pd.NaT

    conf_on_price_dates[CONF_COL] = conf_on_price_dates[CONF_COL].interpolate(
        method="linear",
        limit_direction="both"
    )

    if pd.notna(first_conf_date) and pd.notna(last_conf_date):
        conf_on_price_dates.loc[
            (conf_on_price_dates[PRICE_DATE_COL] < first_conf_date) |
            (conf_on_price_dates[PRICE_DATE_COL] > last_conf_date),
            CONF_COL
        ] = pd.NA

    p["price_plot"] = p[PRICE_COL].rolling(
        window=PRICE_SMOOTH_WINDOW,
        min_periods=1
    ).mean()

    conf_on_price_dates["conf_plot"] = conf_on_price_dates[CONF_COL].rolling(
        window=CONF_SMOOTH_WINDOW,
        min_periods=1
    ).mean()

    return p, conf_on_price_dates, first_conf_date


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
    )

    ax_price.set_ylabel("Price", color="black")
    ax_price.tick_params(axis="x", colors="black")
    ax_price.tick_params(axis="y", colors="black")

    ax_conf = ax_price.twinx()

    if add_undefined_prefix and pd.notna(first_conf_date):
        pre_mask = conf_interp[PRICE_DATE_COL] < first_conf_date

        if pre_mask.any():
            ax_conf.plot(
                conf_interp.loc[pre_mask, PRICE_DATE_COL],
                [ZERO_VISUAL_OFFSET] * pre_mask.sum(),
                color=conf_color,
                linewidth=CONF_LINEWIDTH,
                linestyle="--",
                zorder=5
            )

        first_row = conf_interp.loc[conf_interp[PRICE_DATE_COL] == first_conf_date]
        if not first_row.empty:
            first_val = first_row["conf_plot"].iloc[0]
            if pd.notna(first_val):
                ax_conf.plot(
                    [first_conf_date, first_conf_date],
                    [0, first_val],
                    color=conf_color,
                    linewidth=CONF_LINEWIDTH,
                    linestyle="--",
                    zorder=5
                )

    ax_conf.plot(
        conf_interp[PRICE_DATE_COL],
        conf_interp["conf_plot"],
        color=conf_color,
        linewidth=CONF_LINEWIDTH,
        label=conf_label,
        zorder=6
    )

    ax_conf.set_ylabel("Confidence Score", color="black")
    ax_conf.set_ylim(0, 1)
    ax_conf.set_yticks([0, 0.5, 1])
    ax_conf.set_yticklabels(["0", "0.5", "1"])
    ax_conf.tick_params(axis="y", colors="black")

    ax_price.set_xlim(axis_start, axis_end)
    ax_price.margins(x=0)
    ax_conf.margins(x=0)

    ax_price.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.8)
    ax_conf.grid(False)

    ax_price.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    lines1, labels1 = ax_price.get_legend_handles_labels()
    lines2, labels2 = ax_conf.get_legend_handles_labels()

    ax_price.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        frameon=False
    )


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
    })

    shu_confidence_csv = find_file(SHU_CONFIDENCE_GLOB)

    prices = load_price()
    long_conf = load_confidence(LONG_CONFIDENCE_CSV)
    shu_conf = load_confidence(shu_confidence_csv)

    axis_start = shu_conf[CONF_DATE_COL].min()
    axis_end = shu_conf[CONF_DATE_COL].max()

    long_prices, long_conf_interp, long_first_conf_date = prepare_panel(
        prices, long_conf, axis_start, axis_end
    )
    shu_prices, shu_conf_interp, shu_first_conf_date = prepare_panel(
        prices, shu_conf, axis_start, axis_end
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=FIGSIZE,
        sharex=False
    )

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
    )

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
    )

    axes[1].set_xlabel("Date", color="black")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved plot to:", OUTPUT_PNG)
    print("Long confidence file:", LONG_CONFIDENCE_CSV)
    print("Shu short confidence file:", shu_confidence_csv)
    print("Shared range:", axis_start.date(), "to", axis_end.date())


if __name__ == "__main__":
    main()