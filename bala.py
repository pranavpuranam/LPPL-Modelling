import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


PRICE_CSV = "eurostoxx600_prices.csv"
CONFIDENCE_CSV = "eurostoxx600_lppls_confidence.csv"
OUTPUT_PNG = "eurostoxx600_price_confidence.png"


def load_data():
    prices = pd.read_csv(PRICE_CSV)
    confidence = pd.read_csv(CONFIDENCE_CSV)

    prices["Date"] = pd.to_datetime(prices["Date"])
    confidence["t2"] = pd.to_datetime(confidence["t2"])

    prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
    confidence["bubble_confidence"] = pd.to_numeric(
        confidence["bubble_confidence"], errors="coerce"
    )

    prices = prices.dropna(subset=["Date", "Close"]).sort_values("Date")
    confidence = confidence.dropna(subset=["t2", "bubble_confidence"]).sort_values("t2")

    return prices, confidence


def interpolate_confidence_to_price_dates(prices, confidence):
    """
    Linearly interpolates sparse CI(t2) points onto the full price date grid.
    """
    price_dates = prices[["Date"]].copy()

    conf_interp = (
        confidence[["t2", "bubble_confidence"]]
        .rename(columns={"t2": "Date"})
        .merge(price_dates, on="Date", how="outer")
        .sort_values("Date")
    )

    conf_interp["bubble_confidence"] = (
        conf_interp["bubble_confidence"]
        .interpolate(method="linear", limit_area="inside")
    )

    conf_interp = conf_interp[conf_interp["Date"].isin(prices["Date"])]

    return conf_interp


def plot_price_confidence(prices, conf_interp):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.plot(
        prices["Date"],
        prices["Close"],
        color="black",
        linewidth=1.5,
        label="STOXX Europe 600 Price",
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()

    ax2.plot(
        conf_interp["Date"],
        conf_interp["bubble_confidence"],
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="LPPLS Confidence Score",
    )

    ax2.set_ylabel("Confidence Score")
    ax2.set_ylim(0, 1)

    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        frameon=False,
    )

    plt.title("STOXX Europe 600 Price and LPPLS Confidence Score")
    plt.tight_layout()

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    prices, confidence = load_data()
    conf_interp = interpolate_confidence_to_price_dates(prices, confidence)
    plot_price_confidence(prices, conf_interp)