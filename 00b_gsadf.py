import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# ============================================================
# SETTINGS
# ============================================================

PRICE_CSV = "eurostoxx600_prices.csv"

DATE_COL = "Date"
PRICE_COL = "Close"

USE_MONTHLY = True          # strongly recommended for GSADF/BSADF
NBOOT = 499                 # use 999 for final
CV_LEVEL = 0.90             # 0.90 detects more; 0.95 stricter

MIN_WINDOW_FRAC = 0.10      # lower = more labels; try 0.10–0.20
ADF_LAGS = 0                # keep 0 first; try 1 if noisy

MIN_BUBBLE_PERIODS = 2      # monthly periods if USE_MONTHLY=True
MERGE_GAP_PERIODS = 1

OUTPUT_DAILY = "eurostoxx600_bsadf_daily_labels.csv"
OUTPUT_EVENTS = "eurostoxx600_bsadf_events.csv"
OUTPUT_PNG = "eurostoxx600_bsadf_labels.png"

np.random.seed(42)


# ============================================================
# DATA
# ============================================================

def load_data():
    df = pd.read_csv(PRICE_CSV)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, PRICE_COL]).sort_values(DATE_COL)

    if USE_MONTHLY:
        df = (
            df.set_index(DATE_COL)
              .resample("ME")
              .last()
              .dropna()
              .reset_index()
        )

    df["log_price"] = np.log(df[PRICE_COL])
    return df.reset_index(drop=True)


# ============================================================
# ADF RIGHT-TAILED TEST
# Δy_t = α + β y_{t-1} + lagged Δy terms + ε_t
# ============================================================

def adf_tstat(y, lags=0):
    y = np.asarray(y)

    dy = np.diff(y)
    y_lag = y[:-1]

    if lags > 0:
        rows = []
        target = []

        for t in range(lags, len(dy)):
            row = [y_lag[t]]
            row += [dy[t - j] for j in range(1, lags + 1)]
            rows.append(row)
            target.append(dy[t])

        X = np.asarray(rows)
        Y = np.asarray(target)
    else:
        X = y_lag.reshape(-1, 1)
        Y = dy

    if len(Y) < 8:
        return np.nan

    X = add_constant(X)
    model = OLS(Y, X).fit()

    return model.tvalues[1]   # t-stat on y_{t-1}


def bsadf_series(y, min_window, lags=0):
    T = len(y)
    out = np.full(T, np.nan)

    for r2 in range(min_window - 1, T):
        stats = []

        for r1 in range(0, r2 - min_window + 2):
            window = y[r1:r2 + 1]
            stats.append(adf_tstat(window, lags=lags))

        out[r2] = np.nanmax(stats)

    return out


# ============================================================
# BOOTSTRAP CRITICAL VALUES
# Random walk with drift calibrated from data
# ============================================================

def bootstrap_cv(y, min_window, nboot, cv_level, lags=0):
    T = len(y)
    dy = np.diff(y)

    mu = np.nanmean(dy)
    sigma = np.nanstd(dy, ddof=1)

    boot_stats = np.full((nboot, T), np.nan)

    for b in range(nboot):
        eps = np.random.normal(mu, sigma, T - 1)
        yb = np.r_[y[0], y[0] + np.cumsum(eps)]

        boot_stats[b, :] = bsadf_series(yb, min_window, lags=lags)

        if (b + 1) % 50 == 0:
            print(f"Bootstrap {b + 1}/{nboot}")

    return np.nanquantile(boot_stats, cv_level, axis=0)


# ============================================================
# LABEL CLEANING
# ============================================================

def clean_labels(raw, min_len=2, merge_gap=1):
    raw = np.asarray(raw).astype(bool)
    idx = np.where(raw)[0]

    if len(idx) == 0:
        return raw.astype(int)

    groups = []
    s = idx[0]
    p = idx[0]

    for i in idx[1:]:
        if i - p <= merge_gap + 1:
            p = i
        else:
            groups.append((s, p))
            s = i
            p = i

    groups.append((s, p))

    clean = np.zeros_like(raw, dtype=bool)

    for s, e in groups:
        if e - s + 1 >= min_len:
            clean[s:e + 1] = True

    return clean.astype(int)


def extract_events(df):
    events = []
    label = df["bubble_label"].values.astype(bool)
    idx = np.where(label)[0]

    if len(idx) == 0:
        return pd.DataFrame()

    groups = []
    s = idx[0]
    p = idx[0]

    for i in idx[1:]:
        if i == p + 1:
            p = i
        else:
            groups.append((s, p))
            s = i
            p = i

    groups.append((s, p))

    for s, e in groups:
        sub = df.iloc[s:e + 1]
        peak_idx = sub[PRICE_COL].idxmax()

        events.append({
            "bubble_start": df.loc[s, DATE_COL],
            "bubble_end": df.loc[e, DATE_COL],
            "duration_periods": e - s + 1,
            "peak_date": df.loc[peak_idx, DATE_COL],
            "peak_price": df.loc[peak_idx, PRICE_COL],
            "max_bsadf": sub["bsadf"].max(),
        })

    return pd.DataFrame(events)


# ============================================================
# PLOT
# ============================================================

def plot_results(df):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        df[DATE_COL],
        df[PRICE_COL],
        color="#0000ce",
        linewidth=1.2,
        label="Price"
    )

    active = False
    start = None

    for i in range(len(df)):
        if df.loc[i, "bubble_label"] == 1 and not active:
            active = True
            start = df.loc[i, DATE_COL]

        if active and (df.loc[i, "bubble_label"] == 0 or i == len(df) - 1):
            end = df.loc[i, DATE_COL]
            ax.axvspan(start, end, color="red", alpha=0.18)
            active = False

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.grid(True, which="major", axis="both", alpha=0.3, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================

df = load_data()
y = df["log_price"].values
T = len(y)

min_window = int(np.floor(MIN_WINDOW_FRAC * T))
min_window = max(min_window, 24 if USE_MONTHLY else 125)

print("Observations:", T)
print("Monthly data:", USE_MONTHLY)
print("Minimum window:", min_window)
print("Critical value level:", CV_LEVEL)

print("Computing BSADF...")
df["bsadf"] = bsadf_series(y, min_window, lags=ADF_LAGS)

print("Bootstrapping critical values...")
df["bsadf_cv"] = bootstrap_cv(
    y=y,
    min_window=min_window,
    nboot=NBOOT,
    cv_level=CV_LEVEL,
    lags=ADF_LAGS
)

df["raw_bubble_label"] = (df["bsadf"] > df["bsadf_cv"]).astype(int)

df["bubble_label"] = clean_labels(
    df["raw_bubble_label"],
    min_len=MIN_BUBBLE_PERIODS,
    merge_gap=MERGE_GAP_PERIODS
)

events = extract_events(df)

df.to_csv(OUTPUT_DAILY, index=False)
events.to_csv(OUTPUT_EVENTS, index=False)

plot_results(df)

print("\nGSADF statistic:", np.nanmax(df["bsadf"]))
print("Max BSADF critical value:", np.nanmax(df["bsadf_cv"]))
print("\nDetected events:")
print(events)
print("\nSaved:", OUTPUT_DAILY)
print("Saved:", OUTPUT_EVENTS)
print("Saved:", OUTPUT_PNG)