import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# SETTINGS — EUROSTOXX 600 PRICE WITH 3 INDICATOR TYPES
# ============================================================

PRICE_CSV = "eurostoxx600_prices.csv"
OUTPUT_PNG = "eurostoxx600_price_three_indicator_dates.png"

DATE_COL = "Date"
PRICE_COL = "Close"

SMOOTH_PRICE = True
PRICE_SMOOTH_WINDOW = 5

PRICE_COLOR = "#000000"
LIT_COLOR = "red"
DRAWDOWN_COLOR = "#0000ce"
GSADF_COLOR = "green"

PRICE_LINEWIDTH = 1.2
FIGSIZE = (12, 6)

# Blue square starts more clearly above the price
BASE_OFFSET_FRACTION = 0.06

# Keep spacing between square, circle, triangle the same
OFFSET_STEP_FRACTION = 0.045

LITERATURE_DATES = [
    "2007-06-01",
    "2011-02-17",
    "2015-07-20",
    "2018-01-23",
    "2020-02-19",
    "2022-01-05",
]

DRAWDOWN_DATES = [
    "2007-06-01",
    "2009-01-06",
    "2011-02-17",
    "2015-04-15",
    "2019-04-23",
]

GSADF_BSADF_DATES = [
    "2007-06-01",
]


def load_prices():
    prices = pd.read_csv(PRICE_CSV)

    prices[DATE_COL] = pd.to_datetime(prices[DATE_COL], errors="coerce")
    prices[PRICE_COL] = pd.to_numeric(prices[PRICE_COL], errors="coerce")

    prices = (
        prices.dropna(subset=[DATE_COL, PRICE_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    return prices


def apply_smoothing(prices):
    prices = prices.copy()

    prices["price_plot"] = (
        prices[PRICE_COL].rolling(window=PRICE_SMOOTH_WINDOW, min_periods=1).mean()
        if SMOOTH_PRICE else prices[PRICE_COL]
    )

    return prices


def mark_nearest_tc_dates(prices, event_dates, col_name):
    event_df = pd.DataFrame({"event_date": pd.to_datetime(event_dates)})
    trading_dates = prices[[DATE_COL]].sort_values(DATE_COL)

    matched = pd.merge_asof(
        event_df.sort_values("event_date"),
        trading_dates,
        left_on="event_date",
        right_on=DATE_COL,
        direction="nearest",
    )

    matched_dates = matched[DATE_COL].dropna().unique()

    prices[col_name] = 0
    prices.loc[prices[DATE_COL].isin(matched_dates), col_name] = 1

    print(f"\nMatched dates for {col_name}:")
    print(matched)

    return prices


def add_tc_indicators(prices):
    prices = prices.copy()

    prices = mark_nearest_tc_dates(
        prices,
        LITERATURE_DATES,
        "tc_literature"
    )

    prices = mark_nearest_tc_dates(
        prices,
        GSADF_BSADF_DATES,
        "tc_gsadf"
    )

    prices = mark_nearest_tc_dates(
        prices,
        DRAWDOWN_DATES,
        "tc_drawdown"
    )

    return prices


def save_updated_price_file(prices):
    save_cols = [DATE_COL, PRICE_COL]

    if "log_price" in prices.columns:
        save_cols.append("log_price")

    save_cols += ["tc_literature", "tc_gsadf", "tc_drawdown"]

    prices[save_cols].to_csv(PRICE_CSV, index=False)
    print("\nUpdated existing price file:", PRICE_CSV)


def get_event_points(prices, event_dates, event_name="Event Date"):
    event_df = pd.DataFrame({event_name: pd.to_datetime(event_dates)})
    plot_df = prices[[DATE_COL, "price_plot"]].sort_values(DATE_COL)

    points = pd.merge_asof(
        event_df.sort_values(event_name),
        plot_df,
        left_on=event_name,
        right_on=DATE_COL,
        direction="nearest",
    )

    return points


def add_vertical_offsets(prices, drawdown_points, gsadf_points, lit_points):
    price_range = prices[PRICE_COL].max() - prices[PRICE_COL].min()

    base_offset = BASE_OFFSET_FRACTION * price_range
    offset_step = OFFSET_STEP_FRACTION * price_range

    drawdown_points = drawdown_points.copy()
    gsadf_points = gsadf_points.copy()
    lit_points = lit_points.copy()

    drawdown_points["y_plot"] = drawdown_points["price_plot"] + base_offset
    gsadf_points["y_plot"] = gsadf_points["price_plot"] + base_offset + offset_step
    lit_points["y_plot"] = lit_points["price_plot"] + base_offset + 2 * offset_step

    return drawdown_points, gsadf_points, lit_points


def make_plot(prices, lit_points, drawdown_points, gsadf_points):
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

    fig, ax = plt.subplots(figsize=FIGSIZE)

    price_line, = ax.plot(
        prices[DATE_COL],
        prices["price_plot"],
        color=PRICE_COLOR,
        linewidth=PRICE_LINEWIDTH,
        antialiased=True,
        label="Price",
    )

    drawdown_scatter = ax.scatter(
        drawdown_points[DATE_COL],
        drawdown_points["y_plot"],
        color=DRAWDOWN_COLOR,
        marker="s",
        s=55,
        zorder=5,
        label="Drawdown",
    )

    gsadf_scatter = ax.scatter(
        gsadf_points[DATE_COL],
        gsadf_points["y_plot"],
        color=GSADF_COLOR,
        marker="o",
        s=55,
        zorder=6,
        label="GSADF/BSADF",
    )

    literature_scatter = ax.scatter(
        lit_points[DATE_COL],
        lit_points["y_plot"],
        color=LIT_COLOR,
        marker="^",
        s=55,
        zorder=7,
        label="Literature",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_xlim(prices[DATE_COL].min(), prices[DATE_COL].max())
    ax.margins(x=0)

    raw_price_min = prices[PRICE_COL].min()
    raw_price_max = prices[PRICE_COL].max()
    price_range = raw_price_max - raw_price_min

    y_min = raw_price_min - 0.05 * price_range
    y_max = raw_price_max + 0.18 * price_range

    ax.set_ylim(y_min, y_max)

    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.8)

    ax.legend(
        [price_line, literature_scatter, gsadf_scatter, drawdown_scatter],
        ["Price", "Literature", "GSADF/BSADF", "Drawdown"],
        loc="lower right",
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved plot to:", OUTPUT_PNG)


if __name__ == "__main__":
    prices = load_prices()
    prices = apply_smoothing(prices)

    prices = add_tc_indicators(prices)
    save_updated_price_file(prices)

    lit_points = get_event_points(
        prices,
        LITERATURE_DATES,
        event_name="Literature Date"
    )

    drawdown_points = get_event_points(
        prices,
        DRAWDOWN_DATES,
        event_name="Drawdown Date"
    )

    gsadf_points = get_event_points(
        prices,
        GSADF_BSADF_DATES,
        event_name="GSADF/BSADF Date"
    )

    drawdown_points, gsadf_points, lit_points = add_vertical_offsets(
        prices,
        drawdown_points,
        gsadf_points,
        lit_points
    )

    make_plot(prices, lit_points, drawdown_points, gsadf_points)