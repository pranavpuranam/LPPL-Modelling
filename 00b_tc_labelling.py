import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# SETTINGS
# ============================================================

PRICE_CSV = "eurostoxx600_prices.csv"

DATE_COL = "Date"
PRICE_COL = "Close"

LOCAL_WINDOW = 20          # peak must be max within +/- 20 trading days
CONFIRM_HORIZON = 250      # drawdown must occur within next 250 trading days

PRICE_COLOR = "#0000ce"
PEAK_COLOR = "red"

FIGSIZE = (12, 8)


# ============================================================
# LOAD DATA
# ============================================================

def load_prices():
    df = pd.read_csv(PRICE_CSV)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[PRICE_COL] = pd.to_numeric(df[PRICE_COL], errors="coerce")

    df = (
        df.dropna(subset=[DATE_COL, PRICE_COL])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    return df


# ============================================================
# FIND LOCAL MAXIMA FOLLOWED BY CONFIRMED DRAWDOWN
# ============================================================

def find_algorithmic_bubble_peaks(prices, drawdown_threshold):
    """
    A bubble peak is defined as a local maximum followed by a drawdown
    of at least drawdown_threshold within CONFIRM_HORIZON trading days.

    drawdown_threshold should be decimal:
        0.10 = 10%
        0.15 = 15%
        0.20 = 20%
    """

    df = prices.copy()
    peaks = []

    for i in range(LOCAL_WINDOW, len(df) - CONFIRM_HORIZON):
        price_i = df.loc[i, PRICE_COL]

        local_start = i - LOCAL_WINDOW
        local_end = i + LOCAL_WINDOW

        local_max = df.loc[local_start:local_end, PRICE_COL].max()

        if price_i != local_max:
            continue

        future_start = i + 1
        future_end = i + CONFIRM_HORIZON

        future_prices = df.loc[future_start:future_end, PRICE_COL]
        trough_idx = future_prices.idxmin()
        trough_price = df.loc[trough_idx, PRICE_COL]

        realised_drawdown = (price_i - trough_price) / price_i

        if realised_drawdown >= drawdown_threshold:
            peaks.append({
                "peak_date": df.loc[i, DATE_COL],
                "peak_price": price_i,
                "trough_date": df.loc[trough_idx, DATE_COL],
                "trough_price": trough_price,
                "drawdown": realised_drawdown,
                "drawdown_pct": realised_drawdown * 100,
                "days_peak_to_trough": trough_idx - i,
            })

    peaks = pd.DataFrame(peaks)

    if peaks.empty:
        return peaks

    # Remove clustered duplicate peaks: keep the highest peak within overlapping local windows
    peaks = peaks.sort_values("peak_date").reset_index(drop=True)
    cleaned = []

    for _, row in peaks.iterrows():
        if not cleaned:
            cleaned.append(row)
            continue

        last = cleaned[-1]
        days_apart = abs((row["peak_date"] - last["peak_date"]).days)

        if days_apart <= LOCAL_WINDOW * 2:
            if row["peak_price"] > last["peak_price"]:
                cleaned[-1] = row
        else:
            cleaned.append(row)

    return pd.DataFrame(cleaned)


# ============================================================
# PLOT
# ============================================================

def plot_bubble_peaks(prices, peaks, drawdown_threshold):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        prices[DATE_COL],
        prices[PRICE_COL],
        color=PRICE_COLOR,
        linewidth=1.2,
        label="Price",
    )

    if not peaks.empty:
        ax.scatter(
            peaks["peak_date"],
            peaks["peak_price"],
            color=PEAK_COLOR,
            s=45,
            zorder=5,
            label=f"Confirmed peaks: {int(drawdown_threshold * 100)}% drawdown",
        )

        for _, row in peaks.iterrows():
            ax.annotate(
                row["peak_date"].strftime("%Y-%m-%d"),
                xy=(row["peak_date"], row["peak_price"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
                rotation=45,
            )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(False)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.margins(x=0)
    ax.legend(loc="upper left", frameon=False)

    plt.tight_layout()

    output_png = (
        f"eurostoxx600_algorithmic_bubble_peaks_"
        f"{int(drawdown_threshold * 100)}pct_drawdown.png"
    )

    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved plot to:", output_png)


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_bubble_peak_analysis(drawdown_threshold):
    """
    Main function.

    Example:
        run_bubble_peak_analysis(0.10)
        run_bubble_peak_analysis(0.15)
        run_bubble_peak_analysis(0.20)
    """

    prices = load_prices()

    peaks = find_algorithmic_bubble_peaks(
        prices=prices,
        drawdown_threshold=drawdown_threshold,
    )

    output_csv = (
        f"eurostoxx600_algorithmic_bubble_peaks_"
        f"{int(drawdown_threshold * 100)}pct_drawdown.csv"
    )

    peaks.to_csv(output_csv, index=False)

    print("Drawdown threshold:", f"{drawdown_threshold:.0%}")
    print("Number of labelled peaks:", len(peaks))
    print("Saved peaks to:", output_csv)

    if not peaks.empty:
        print(peaks)

    plot_bubble_peaks(prices, peaks, drawdown_threshold)

    return peaks


# ============================================================
# RUN EXAMPLE
# ============================================================

if __name__ == "__main__":
    run_bubble_peak_analysis(0.30)