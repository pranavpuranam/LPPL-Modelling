import pandas as pd
import matplotlib.pyplot as plt


def plot_eurostoxx_history(
    csv_file="eurostoxx600_prices.csv",
    output_file="eurostoxx600_price_history.png",
    resample_freq="W",
    line_color="#0000ce"
):
    df = pd.read_csv(csv_file)

    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = (
        df.dropna(subset=["Date", "Close"])
          .sort_values("Date")
          .set_index("Date")
    )

    plot_df = (
        df["Close"].resample(resample_freq).last().dropna()
        if resample_freq is not None
        else df["Close"]
    )

    critical_dates = pd.to_datetime([
        "2007-06-01",
        "2015-04-15",
        "2022-01-05"
    ])

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 14

    plt.figure(figsize=(8, 5))  # 3:2 aspect ratio

    plt.plot(
        plot_df.index,
        plot_df.values,
        color=line_color,
        linestyle=(0, (1.5, 2.5)),
        linewidth=1.6,
        label="EUROSTOXX 600 Close"
    )

    for date in critical_dates:
        nearest_date = df.index[df.index.get_indexer([date], method="nearest")[0]]
        plt.scatter(
            nearest_date,
            df.loc[nearest_date, "Close"],
            color="black",
            s=55,
            zorder=5
        )

    plt.xlabel("Date")
    plt.ylabel("Close Price")

    plt.grid(False)
    plt.minorticks_off()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved as: {output_file}")


plot_eurostoxx_history(line_color="#0000ce")