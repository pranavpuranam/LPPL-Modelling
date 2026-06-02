# 00b: run gsadf bubble labels

import numpy as np  # numerical operations
import pandas as pd  # handle dataframes
import matplotlib.pyplot as plt  # plot charts
import matplotlib.dates as mdates  # format date axis
from statsmodels.regression.linear_model import OLS  # run regressions
from statsmodels.tools.tools import add_constant  # add intercept

PRICE_CSV = "eurostoxx600_prices.csv"  # input price data
DATE_COL = "Date"  # date column
PRICE_COL = "Close"  # price column
USE_MONTHLY = True  # resample to monthly data
NBOOT = 499  # bootstrap count
CV_LEVEL = 0.90  # critical value level
MIN_WINDOW_FRAC = 0.10  # minimum window fraction
ADF_LAGS = 0  # lag count for adf regression
MIN_BUBBLE_PERIODS = 2  # minimum bubble length
MERGE_GAP_PERIODS = 1  # merge nearby bubble periods
OUTPUT_DAILY = "eurostoxx600_bsadf_daily_labels.csv"  # labelled data output
OUTPUT_EVENTS = "eurostoxx600_bsadf_events.csv"  # event table output
OUTPUT_PNG = "eurostoxx600_bsadf_labels.png"  # plot output

np.random.seed(42)  # fix random seed

def load_data():
    df = pd.read_csv(PRICE_CSV)  # load price data
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])  # parse dates
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")  # force numeric prices
    df = df.dropna(subset=[DATE_COL, PRICE_COL]).sort_values(DATE_COL)  # clean and sort data
    if USE_MONTHLY:  # check monthly setting
        df = (
            df.set_index(DATE_COL)  # set date index
              .resample("ME")  # resample month end
              .last()  # take last monthly value
              .dropna()  # remove missing rows
              .reset_index()  # restore date column
        )
    df["log_price"] = np.log(df[PRICE_COL])  # calculate log price
    return df.reset_index(drop=True)  # return clean data

def adf_tstat(y, lags=0):
    y = np.asarray(y)  # convert to array
    dy = np.diff(y)  # calculate differences
    y_lag = y[:-1]  # lag price series
    if lags > 0:  # check lag setting
        rows = []  # store regressors
        target = []  # store response
        for t in range(lags, len(dy)):  # loop through valid rows
            row = [y_lag[t]]  # add lagged level
            row += [dy[t - j] for j in range(1, lags + 1)]  # add lagged differences
            rows.append(row)  # store row
            target.append(dy[t])  # store target
        X = np.asarray(rows)  # create regressor array
        Y = np.asarray(target)  # create target array
    else:
        X = y_lag.reshape(-1, 1)  # use only lagged level
        Y = dy  # use differences
    if len(Y) < 8:  # avoid tiny windows
        return np.nan  # return missing statistic
    X = add_constant(X)  # add intercept
    model = OLS(Y, X).fit()  # fit adf regression
    return model.tvalues[1]  # return t-statistic

def bsadf_series(y, min_window, lags=0):
    T = len(y)  # get sample length
    out = np.full(T, np.nan)  # initialise output
    for r2 in range(min_window - 1, T):  # loop over endpoints
        stats = []  # store window statistics
        for r1 in range(0, r2 - min_window + 2):  # loop over startpoints
            window = y[r1:r2 + 1]  # slice window
            stats.append(adf_tstat(window, lags=lags))  # compute statistic
        out[r2] = np.nanmax(stats)  # store supremum statistic
    return out  # return bsadf series

def bootstrap_cv(y, min_window, nboot, cv_level, lags=0):
    T = len(y)  # get sample length
    dy = np.diff(y)  # calculate differences
    mu = np.nanmean(dy)  # estimate drift
    sigma = np.nanstd(dy, ddof=1)  # estimate volatility
    boot_stats = np.full((nboot, T), np.nan)  # initialise bootstrap storage
    for b in range(nboot):  # loop over bootstraps
        eps = np.random.normal(mu, sigma, T - 1)  # simulate increments
        yb = np.r_[y[0], y[0] + np.cumsum(eps)]  # create random walk
        boot_stats[b, :] = bsadf_series(yb, min_window, lags=lags)  # compute bootstrap bsadf
        if (b + 1) % 50 == 0:  # print every 50 bootstraps
            print(f"Bootstrap {b + 1}/{nboot}")  # show progress
    return np.nanquantile(boot_stats, cv_level, axis=0)  # return critical values

def clean_labels(raw, min_len=2, merge_gap=1):
    raw = np.asarray(raw).astype(bool)  # convert labels to boolean
    idx = np.where(raw)[0]  # find active labels
    if len(idx) == 0:  # check no bubbles
        return raw.astype(int)  # return empty labels
    groups = []  # store label groups
    s = idx[0]  # start first group
    p = idx[0]  # previous index
    for i in idx[1:]:  # loop through labels
        if i - p <= merge_gap + 1:  # merge close groups
            p = i  # extend group
        else:
            groups.append((s, p))  # store group
            s = i  # start new group
            p = i  # reset previous index
    groups.append((s, p))  # store final group
    clean = np.zeros_like(raw, dtype=bool)  # initialise clean labels
    for s, e in groups:  # loop over groups
        if e - s + 1 >= min_len:  # keep long enough groups
            clean[s:e + 1] = True  # mark clean labels
    return clean.astype(int)  # return integer labels

def extract_events(df):
    events = []  # store events
    label = df["bubble_label"].values.astype(bool)  # get clean labels
    idx = np.where(label)[0]  # find labelled periods
    if len(idx) == 0:  # check no events
        return pd.DataFrame()  # return empty table
    groups = []  # store contiguous groups
    s = idx[0]  # start first group
    p = idx[0]  # previous index
    for i in idx[1:]:  # loop through labelled indices
        if i == p + 1:  # check contiguous
            p = i  # extend group
        else:
            groups.append((s, p))  # store group
            s = i  # start new group
            p = i  # reset previous index
    groups.append((s, p))  # store final group
    for s, e in groups:  # loop through events
        sub = df.iloc[s:e + 1]  # slice event period
        peak_idx = sub[PRICE_COL].idxmax()  # find peak price date
        events.append({
            "bubble_start": df.loc[s, DATE_COL],  # event start
            "bubble_end": df.loc[e, DATE_COL],  # event end
            "duration_periods": e - s + 1,  # event duration
            "peak_date": df.loc[peak_idx, DATE_COL],  # peak date
            "peak_price": df.loc[peak_idx, PRICE_COL],  # peak price
            "max_bsadf": sub["bsadf"].max(),  # maximum bsadf
        })  # store event
    return pd.DataFrame(events)  # return event table

def plot_results(df):
    plt.rcParams.update({
        "font.family": "Arial",  # set font
        "font.size": 18,  # set base font size
        "axes.labelsize": 18,  # set label size
        "axes.titlesize": 18,  # set title size
        "xtick.labelsize": 18,  # set x tick size
        "ytick.labelsize": 18,  # set y tick size
        "legend.fontsize": 18,  # set legend size
    })  # update plot style
    fig, ax = plt.subplots(figsize=(12, 6))  # create figure
    ax.plot(
        df[DATE_COL],
        df[PRICE_COL],
        color="#0000ce",
        linewidth=1.2,
        label="Price"
    )  # plot price series
    active = False  # track active bubble shading
    start = None  # store bubble start
    for i in range(len(df)):  # loop through rows
        if df.loc[i, "bubble_label"] == 1 and not active:  # start bubble region
            active = True  # activate shading
            start = df.loc[i, DATE_COL]  # store start date
        if active and (df.loc[i, "bubble_label"] == 0 or i == len(df) - 1):  # end bubble region
            end = df.loc[i, DATE_COL]  # store end date
            ax.axvspan(start, end, color="red", alpha=0.18)  # shade bubble region
            active = False  # reset shading
    ax.set_xlabel("Date")  # set x label
    ax.set_ylabel("Price")  # set y label
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # set two-year ticks
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # format years
    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.8)  # add grid
    ax.legend(frameon=False, loc="upper left")  # add legend
    plt.tight_layout()  # tidy layout
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")  # save plot
    plt.show()  # show plot

df = load_data()  # load input data
y = df["log_price"].values  # extract log prices
T = len(y)  # get sample length
min_window = int(np.floor(MIN_WINDOW_FRAC * T))  # calculate minimum window
min_window = max(min_window, 24 if USE_MONTHLY else 125)  # enforce minimum window

print("Observations:", T)  # print observation count
print("Monthly data:", USE_MONTHLY)  # print frequency setting
print("Minimum window:", min_window)  # print minimum window
print("Critical value level:", CV_LEVEL)  # print critical value level
print("Computing BSADF...")  # print progress
df["bsadf"] = bsadf_series(y, min_window, lags=ADF_LAGS)  # compute bsadf series

print("Bootstrapping critical values...")  # print progress
df["bsadf_cv"] = bootstrap_cv(
    y=y,
    min_window=min_window,
    nboot=NBOOT,
    cv_level=CV_LEVEL,
    lags=ADF_LAGS
)  # compute bootstrap critical values

df["raw_bubble_label"] = (df["bsadf"] > df["bsadf_cv"]).astype(int)  # label raw bubbles
df["bubble_label"] = clean_labels(
    df["raw_bubble_label"],
    min_len=MIN_BUBBLE_PERIODS,
    merge_gap=MERGE_GAP_PERIODS
)  # clean bubble labels

events = extract_events(df)  # extract bubble events
df.to_csv(OUTPUT_DAILY, index=False)  # save labelled data
events.to_csv(OUTPUT_EVENTS, index=False)  # save event table
plot_results(df)  # plot labelled bubbles

print("\nGSADF statistic:", np.nanmax(df["bsadf"]))  # print max statistic
print("Max BSADF critical value:", np.nanmax(df["bsadf_cv"]))  # print max critical value
print("\nDetected events:")  # print event header
print(events)  # print events
print("\nSaved:", OUTPUT_DAILY)  # print saved daily file
print("Saved:", OUTPUT_EVENTS)  # print saved event file
print("Saved:", OUTPUT_PNG)  # print saved plot