# 05b: plot confidence-conditioned regressions

import pandas as pd  # handle dataframes
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plot charts
import statsmodels.formula.api as smf  # run formula regressions

FITS_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"  # lppl fit input
CONFIDENCE_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"  # confidence input
PRICE_PATH = "eurostoxx600_prices.csv"  # price and event input
OUTPUT_PNG = "ci_vs_tc_error_fe_stacked.png"  # plot output
OUTPUT_PANEL = "ci_vs_tc_error_fe_panel.csv"  # panel output
DATE_COL = "Date"  # price date column
EVENT_COL = "tc_literature"  # event label column
CI_COL = "positive_bubble_confidence"  # confidence column
LOOKBACK_TRADING_DAYS = 250  # pre-event lookback
VALID_ONLY = True  # keep only valid fits
MAX_ABS_ERROR_FOR_PLOT = 750  # max absolute error filter
TOP_POINT_COLOR = "red"  # signed error point colour
BOTTOM_POINT_COLOR = "#0000cc"  # absolute error point colour
LINE_COLOR = "black"  # regression line colour
FIGSIZE = (17.76, 11.98)  # figure size

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 34,
    "axes.labelsize": 34,
    "axes.titlesize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 26,
})  # set plot style

fits = pd.read_csv(FITS_PATH)  # load lppl fits
confidence = pd.read_csv(CONFIDENCE_PATH)  # load confidence data
prices = pd.read_csv(PRICE_PATH)  # load price data

for col in ["t1", "t2", "tc_predicted"]:  # loop over fit date columns
    fits[col] = pd.to_datetime(fits[col], errors="coerce")  # parse fit dates

confidence["t2"] = pd.to_datetime(confidence["t2"], errors="coerce")  # parse confidence dates
prices[DATE_COL] = pd.to_datetime(prices[DATE_COL], errors="coerce")  # parse price dates

fits["positive_lppls_valid"] = (
    fits["positive_lppls_valid"]
    .astype(str)
    .str.lower()
    .eq("true")
)  # convert validity flag

prices = prices.sort_values(DATE_COL).reset_index(drop=True)  # sort prices

fits = fits.merge(
    confidence[["t2", CI_COL]].dropna(),
    on="t2",
    how="left"
)  # merge ci onto fits

events = prices.loc[
    prices[EVENT_COL] == 1,
    DATE_COL
].dropna().sort_values().to_list()  # get event dates

chunks = []  # store event panels

for event_date in events:  # loop over events
    event_idx = prices.index[prices[DATE_COL] == event_date][0]  # get event index
    start_idx = max(0, event_idx - LOOKBACK_TRADING_DAYS)  # get lookback start
    start_date = prices.loc[start_idx, DATE_COL]  # get lookback date
    g = fits[
        (fits["t2"] >= start_date)
        & (fits["t2"] < event_date)
        & (fits["tc_predicted"].notna())
        & (fits[CI_COL].notna())
    ].copy()  # keep pre-event fits
    if VALID_ONLY:  # check valid-only setting
        g = g[g["positive_lppls_valid"]].copy()  # keep valid fits
    if len(g) == 0:  # check event observations
        continue  # skip empty events
    g["event_date"] = event_date.strftime("%Y-%m-%d")  # store event date
    g["signed_tc_error"] = (g["tc_predicted"] - event_date).dt.days  # calculate signed error
    g["absolute_tc_error"] = g["signed_tc_error"].abs()  # calculate absolute error
    chunks.append(g)  # store event data

if len(chunks) == 0:  # check panel exists
    raise ValueError("No valid event-fit observations found.")  # stop if empty

panel = pd.concat(chunks, ignore_index=True)  # combine event panels
panel = panel[
    [CI_COL, "signed_tc_error", "absolute_tc_error", "event_date"]
].dropna().copy()  # keep plot columns
panel = panel.rename(columns={CI_COL: "ci"})  # shorten ci column name

if MAX_ABS_ERROR_FOR_PLOT is not None:  # check error filter
    panel = panel[panel["absolute_tc_error"] <= MAX_ABS_ERROR_FOR_PLOT].copy()  # filter large errors

panel.to_csv(OUTPUT_PANEL, index=False)  # save panel

signed_fe = smf.ols(
    "signed_tc_error ~ ci + C(event_date)",
    data=panel
).fit(cov_type="HC3")  # fit signed error fixed-effects model

abs_fe = smf.ols(
    "absolute_tc_error ~ ci + C(event_date)",
    data=panel
).fit(cov_type="HC3")  # fit absolute error fixed-effects model

def p_format(p):
    if p < 0.001:  # check small p-value
        return "<0.001"  # format small p-value
    return f"{p:.4f}"  # format p-value

print("\nEvent FE, HC3 results:")  # print header
print(
    f"Signed error: beta={signed_fe.params['ci']:.2f}, "
    f"p={p_format(signed_fe.pvalues['ci'])}, "
    f"R2={signed_fe.rsquared:.4f}"
)  # print signed error model
print(
    f"Absolute error: beta={abs_fe.params['ci']:.2f}, "
    f"p={p_format(abs_fe.pvalues['ci'])}, "
    f"R2={abs_fe.rsquared:.4f}"
)  # print absolute error model

def add_fe_lines(ax, model):
    ci_grid = np.linspace(panel["ci"].min(), panel["ci"].max(), 100)  # create ci grid
    for event_date in sorted(panel["event_date"].unique()):  # loop over events
        pred_df = pd.DataFrame({
            "ci": ci_grid,
            "event_date": event_date
        })  # create prediction data
        y_hat = model.predict(pred_df)  # predict fitted line
        ax.plot(
            ci_grid,
            y_hat,
            color=LINE_COLOR,
            linewidth=2.2,
            alpha=0.55
        )  # plot event line

def make_panel(ax, y_col, y_label, model, point_color):
    ax.scatter(
        panel["ci"],
        panel[y_col],
        s=28,
        alpha=0.25,
        color=point_color,
        edgecolor="none"
    )  # plot observations
    add_fe_lines(ax, model)  # add fixed-effects lines
    beta = model.params["ci"]  # get ci coefficient
    ax.set_ylabel(y_label)  # set y label
    ax.text(
        0.98,
        0.98,
        rf"$\beta$ = {beta:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=30,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=4)
    )  # add beta label
    ax.grid(True, which="major", alpha=0.30, linewidth=1.2)  # add grid
    ax.minorticks_off()  # remove minor ticks

fig, axes = plt.subplots(
    2,
    1,
    figsize=FIGSIZE,
    sharex=True,
    constrained_layout=True
)  # create stacked plot

make_panel(
    ax=axes[0],
    y_col="signed_tc_error",
    y_label="Error",
    model=signed_fe,
    point_color=TOP_POINT_COLOR
)  # plot signed error panel

make_panel(
    ax=axes[1],
    y_col="absolute_tc_error",
    y_label="Absolute Error",
    model=abs_fe,
    point_color=BOTTOM_POINT_COLOR
)  # plot absolute error panel

axes[0].set_xlabel("Confidence Indicator")  # set top x label
axes[1].set_xlabel("Confidence Indicator")  # set bottom x label
axes[0].axhline(0, color="gray", linestyle="--", linewidth=1.2)  # add zero line
axes[1].axhline(0, color="gray", linestyle="--", linewidth=1.2)  # add zero line

plt.savefig(OUTPUT_PNG, dpi=300)  # save plot
plt.show()  # show plot

print(f"\nSaved plot: {OUTPUT_PNG}")  # print plot path
print(f"Saved panel: {OUTPUT_PANEL}")  # print panel path