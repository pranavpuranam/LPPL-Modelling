import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# SETTINGS — T2 STEP 5 SHU-LONG POSITIVE CONFIDENCE
# ============================================================

PRICE_CSV = "eurostoxx600_prices.csv"

CONFIDENCE_CSV = "eurostoxx600_lppls_t2step5_shu_long_positive_confidence.csv"
OUTPUT_PNG = "eurostoxx600_lppls_t2step5_shu_long_positive_price_confidence.png"

PRICE_DATE_COL = "Date"
PRICE_COL = "Close"

CONF_DATE_COL = "t2"
CONF_COL = "positive_bubble_confidence"

SMOOTH_PRICE = True
SMOOTH_CONFIDENCE = True

PRICE_SMOOTH_WINDOW = 5
CONF_SMOOTH_WINDOW = 15

PRICE_COLOR = "#0000ce"
CONF_COLOR = "red"

PRICE_LINEWIDTH = 1.2
CONF_LINEWIDTH = 1.4

FIGSIZE = (12, 8)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    prices = pd.read_csv(PRICE_CSV)
    conf = pd.read_csv(CONFIDENCE_CSV)

    prices[PRICE_DATE_COL] = pd.to_datetime(prices[PRICE_DATE_COL], errors="coerce")
    prices[PRICE_COL] = pd.to_numeric(prices[PRICE_COL], errors="coerce")

    conf[CONF_DATE_COL] = pd.to_datetime(conf[CONF_DATE_COL], errors="coerce")
    conf[CONF_COL] = pd.to_numeric(conf[CONF_COL], errors="coerce")

    prices = (
        prices.dropna(subset=[PRICE_DATE_COL, PRICE_COL])
        .sort_values(PRICE_DATE_COL)
        .reset_index(drop=True)
    )

    conf = (
        conf.dropna(subset=[CONF_DATE_COL, CONF_COL])
        .sort_values(CONF_DATE_COL)
        .reset_index(drop=True)
    )

    return prices, conf


# ============================================================
# TRIM TO CONFIDENCE DATE RANGE
# ============================================================

def trim_to_confidence_range(prices, conf):
    start_date = conf[CONF_DATE_COL].min()
    end_date = conf[CONF_DATE_COL].max()

    prices_trimmed = (
        prices[
            (prices[PRICE_DATE_COL] >= start_date) &
            (prices[PRICE_DATE_COL] <= end_date)
        ]
        .copy()
        .reset_index(drop=True)
    )

    conf_trimmed = (
        conf[
            (conf[CONF_DATE_COL] >= start_date) &
            (conf[CONF_DATE_COL] <= end_date)
        ]
        .copy()
        .reset_index(drop=True)
    )

    return prices_trimmed, conf_trimmed, start_date, end_date


# ============================================================
# INTERPOLATE CONFIDENCE ONTO PRICE DATE GRID
# ============================================================

def interpolate_confidence_to_price_dates(prices, conf):
    conf2 = conf[[CONF_DATE_COL, CONF_COL]].rename(columns={CONF_DATE_COL: "Date"})
    price_dates = prices[[PRICE_DATE_COL]].rename(columns={PRICE_DATE_COL: "Date"})

    merged = (
        pd.merge(price_dates, conf2, on="Date", how="left")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    merged[CONF_COL] = merged[CONF_COL].interpolate(
        method="linear",
        limit_direction="both"
    )

    return merged.rename(columns={"Date": PRICE_DATE_COL})


# ============================================================
# APPLY SMOOTHING
# ============================================================

def apply_smoothing(prices, conf_interp):
    prices = prices.copy()
    conf_interp = conf_interp.copy()

    if SMOOTH_PRICE:
        prices["price_plot"] = (
            prices[PRICE_COL]
            .rolling(window=PRICE_SMOOTH_WINDOW, min_periods=1)
            .mean()
        )
    else:
        prices["price_plot"] = prices[PRICE_COL]

    if SMOOTH_CONFIDENCE:
        conf_interp["conf_plot"] = (
            conf_interp[CONF_COL]
            .rolling(window=CONF_SMOOTH_WINDOW, min_periods=1)
            .mean()
        )
    else:
        conf_interp["conf_plot"] = conf_interp[CONF_COL]

    return prices, conf_interp


# ============================================================
# PLOT
# ============================================================

def make_plot(prices, conf_interp, start_date, end_date):
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

    fig, ax1 = plt.subplots(figsize=FIGSIZE)

    ax1.plot(
        prices[PRICE_DATE_COL],
        prices["price_plot"],
        color=PRICE_COLOR,
        linewidth=PRICE_LINEWIDTH,
        antialiased=True,
        label="Price",
    )

    ax1.set_xlabel("Date", color="black")
    ax1.set_ylabel("Price", color="black")
    ax1.tick_params(axis="x", colors="black")
    ax1.tick_params(axis="y", colors="black")
    ax1.grid(False)

    ax2 = ax1.twinx()

    ax2.plot(
        conf_interp[PRICE_DATE_COL],
        conf_interp["conf_plot"],
        color=CONF_COLOR,
        linewidth=CONF_LINEWIDTH,
        antialiased=True,
        label="Confidence Score",
    )

    ax2.set_ylabel("Confidence Score", color="black")
    ax2.tick_params(axis="y", colors="black")
    ax2.set_ylim(0, 1)
    ax2.grid(False)

    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax1.set_xlim(start_date, end_date)
    ax1.margins(x=0)
    ax2.margins(x=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    prices, conf = load_data()
    prices, conf, start_date, end_date = trim_to_confidence_range(prices, conf)
    conf_interp = interpolate_confidence_to_price_dates(prices, conf)
    prices, conf_interp = apply_smoothing(prices, conf_interp)
    make_plot(prices, conf_interp, start_date, end_date)

    print("Saved plot to:", OUTPUT_PNG)
    print("Chart starts at:", start_date.date())
    print("Chart ends at:", end_date.date())