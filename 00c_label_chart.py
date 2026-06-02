# 00c: plot critical time labels

import pandas as pd  # handle dataframes
import matplotlib.pyplot as plt  # plot charts
import matplotlib.dates as mdates  # format date axis

PRICE_CSV = "eurostoxx600_prices.csv"  # input price data
OUTPUT_PNG = "eurostoxx600_price_three_indicator_dates.png"  # plot output
DATE_COL = "Date"  # date column
PRICE_COL = "Close"  # price column
SMOOTH_PRICE = True  # smooth plotted price
PRICE_SMOOTH_WINDOW = 5  # smoothing window
PRICE_COLOR = "#000000"  # price line colour
LIT_COLOR = "red"  # literature marker colour
DRAWDOWN_COLOR = "#0000ce"  # drawdown marker colour
GSADF_COLOR = "green"  # gsadf marker colour
PRICE_LINEWIDTH = 1.2  # price line width
FIGSIZE = (12, 6)  # figure size
BASE_OFFSET_FRACTION = 0.06  # first marker offset
OFFSET_STEP_FRACTION = 0.045  # offset gap between markers

LITERATURE_DATES = [
    "2007-06-01",
    "2011-02-17",
    "2015-07-20",
    "2018-01-23",
    "2020-02-19",
    "2022-01-05",
]  # literature critical time dates

DRAWDOWN_DATES = [
    "2007-06-01",
    "2009-01-06",
    "2011-02-17",
    "2015-04-15",
    "2019-04-23",
]  # drawdown critical time dates

GSADF_BSADF_DATES = [
    "2007-06-01",
]  # gsadf critical time dates

def load_prices():
    prices = pd.read_csv(PRICE_CSV)  # load price data
    prices[DATE_COL] = pd.to_datetime(prices[DATE_COL], errors="coerce")  # parse dates
    prices[PRICE_COL] = pd.to_numeric(prices[PRICE_COL], errors="coerce")  # force numeric prices
    prices = (
        prices.dropna(subset=[DATE_COL, PRICE_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )  # clean and sort data
    return prices  # return price data

def apply_smoothing(prices):
    prices = prices.copy()  # avoid editing original
    prices["price_plot"] = (
        prices[PRICE_COL].rolling(window=PRICE_SMOOTH_WINDOW, min_periods=1).mean()
        if SMOOTH_PRICE else prices[PRICE_COL]
    )  # create plotted price
    return prices  # return smoothed data

def mark_nearest_tc_dates(prices, event_dates, col_name):
    event_df = pd.DataFrame({"event_date": pd.to_datetime(event_dates)})  # create event table
    trading_dates = prices[[DATE_COL]].sort_values(DATE_COL)  # get trading dates
    matched = pd.merge_asof(
        event_df.sort_values("event_date"),
        trading_dates,
        left_on="event_date",
        right_on=DATE_COL,
        direction="nearest",
    )  # match events to nearest trading dates
    matched_dates = matched[DATE_COL].dropna().unique()  # get matched dates
    prices[col_name] = 0  # initialise label column
    prices.loc[prices[DATE_COL].isin(matched_dates), col_name] = 1  # mark event dates
    print(f"\nMatched dates for {col_name}:")  # print header
    print(matched)  # print matched dates
    return prices  # return labelled prices

def add_tc_indicators(prices):
    prices = prices.copy()  # avoid editing original
    prices = mark_nearest_tc_dates(
        prices,
        LITERATURE_DATES,
        "tc_literature"
    )  # add literature labels
    prices = mark_nearest_tc_dates(
        prices,
        GSADF_BSADF_DATES,
        "tc_gsadf"
    )  # add gsadf labels
    prices = mark_nearest_tc_dates(
        prices,
        DRAWDOWN_DATES,
        "tc_drawdown"
    )  # add drawdown labels
    return prices  # return labelled data

def save_updated_price_file(prices):
    save_cols = [DATE_COL, PRICE_COL]  # base output columns
    if "log_price" in prices.columns:  # check for log price
        save_cols.append("log_price")  # keep log price
    save_cols += ["tc_literature", "tc_gsadf", "tc_drawdown"]  # add label columns
    prices[save_cols].to_csv(PRICE_CSV, index=False)  # overwrite price file
    print("\nUpdated existing price file:", PRICE_CSV)  # print save path

def get_event_points(prices, event_dates, event_name="Event Date"):
    event_df = pd.DataFrame({event_name: pd.to_datetime(event_dates)})  # create event table
    plot_df = prices[[DATE_COL, "price_plot"]].sort_values(DATE_COL)  # get plotted prices
    points = pd.merge_asof(
        event_df.sort_values(event_name),
        plot_df,
        left_on=event_name,
        right_on=DATE_COL,
        direction="nearest",
    )  # match to nearest plotted date
    return points  # return plot points

def add_vertical_offsets(prices, drawdown_points, gsadf_points, lit_points):
    price_range = prices[PRICE_COL].max() - prices[PRICE_COL].min()  # calculate price range
    base_offset = BASE_OFFSET_FRACTION * price_range  # calculate base offset
    offset_step = OFFSET_STEP_FRACTION * price_range  # calculate offset step
    drawdown_points = drawdown_points.copy()  # copy drawdown points
    gsadf_points = gsadf_points.copy()  # copy gsadf points
    lit_points = lit_points.copy()  # copy literature points
    drawdown_points["y_plot"] = drawdown_points["price_plot"] + base_offset  # offset drawdown points
    gsadf_points["y_plot"] = gsadf_points["price_plot"] + base_offset + offset_step  # offset gsadf points
    lit_points["y_plot"] = lit_points["price_plot"] + base_offset + 2 * offset_step  # offset literature points
    return drawdown_points, gsadf_points, lit_points  # return offset points

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
    })  # set plot style
    fig, ax = plt.subplots(figsize=FIGSIZE)  # create figure
    price_line, = ax.plot(
        prices[DATE_COL],
        prices["price_plot"],
        color=PRICE_COLOR,
        linewidth=PRICE_LINEWIDTH,
        antialiased=True,
        label="Price",
    )  # plot price
    drawdown_scatter = ax.scatter(
        drawdown_points[DATE_COL],
        drawdown_points["y_plot"],
        color=DRAWDOWN_COLOR,
        marker="s",
        s=55,
        zorder=5,
        label="Drawdown",
    )  # plot drawdown labels
    gsadf_scatter = ax.scatter(
        gsadf_points[DATE_COL],
        gsadf_points["y_plot"],
        color=GSADF_COLOR,
        marker="o",
        s=55,
        zorder=6,
        label="GSADF/BSADF",
    )  # plot gsadf labels
    literature_scatter = ax.scatter(
        lit_points[DATE_COL],
        lit_points["y_plot"],
        color=LIT_COLOR,
        marker="^",
        s=55,
        zorder=7,
        label="Literature",
    )  # plot literature labels
    ax.set_xlabel("Date")  # set x label
    ax.set_ylabel("Price")  # set y label
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # set two-year ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # format year labels
    ax.set_xlim(prices[DATE_COL].min(), prices[DATE_COL].max())  # set x limits
    ax.margins(x=0)  # remove x padding
    raw_price_min = prices[PRICE_COL].min()  # get min price
    raw_price_max = prices[PRICE_COL].max()  # get max price
    price_range = raw_price_max - raw_price_min  # calculate price range
    y_min = raw_price_min - 0.05 * price_range  # set lower y limit
    y_max = raw_price_max + 0.18 * price_range  # set upper y limit
    ax.set_ylim(y_min, y_max)  # apply y limits
    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.8)  # add grid
    ax.legend(
        [price_line, literature_scatter, gsadf_scatter, drawdown_scatter],
        ["Price", "Literature", "GSADF/BSADF", "Drawdown"],
        loc="lower right",
        frameon=False
    )  # add legend
    plt.tight_layout()  # tidy layout
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")  # save plot
    plt.show()  # show plot
    print("Saved plot to:", OUTPUT_PNG)  # print save path

if __name__ == "__main__":
    prices = load_prices()  # load price data
    prices = apply_smoothing(prices)  # smooth plotted price
    prices = add_tc_indicators(prices)  # add critical time labels
    save_updated_price_file(prices)  # save updated price data
    lit_points = get_event_points(
        prices,
        LITERATURE_DATES,
        event_name="Literature Date"
    )  # get literature plot points
    drawdown_points = get_event_points(
        prices,
        DRAWDOWN_DATES,
        event_name="Drawdown Date"
    )  # get drawdown plot points
    gsadf_points = get_event_points(
        prices,
        GSADF_BSADF_DATES,
        event_name="GSADF/BSADF Date"
    )  # get gsadf plot points
    drawdown_points, gsadf_points, lit_points = add_vertical_offsets(
        prices,
        drawdown_points,
        gsadf_points,
        lit_points
    )  # offset plot points
    make_plot(prices, lit_points, drawdown_points, gsadf_points)  # make final plot