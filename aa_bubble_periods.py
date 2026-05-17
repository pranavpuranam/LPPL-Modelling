import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14

df = pd.read_csv("eurostoxx600_prices.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

bubble_dates = {
    "GFC": "2007-06-01",
    "China QE": "2015-04-15",
    "Post-COVID stimulus": "2022-01-05",
}

event_points = []

for event, date_str in bubble_dates.items():
    target_date = pd.Timestamp(date_str)
    valid_dates = df.index[df.index <= target_date]

    if len(valid_dates) == 0:
        continue

    actual_date = valid_dates[-1]
    actual_price = df.loc[actual_date, "Close"]

    event_points.append({
        "event": event,
        "actual_date": actual_date,
        "price": actual_price
    })

event_df = pd.DataFrame(event_points)

plt.figure(figsize=(9, 6))

plt.plot(
    df.index,
    df["Close"],
    color="darkgrey",
    linestyle=":",
    linewidth=1.8
)

plt.scatter(
    event_df["actual_date"],
    event_df["price"],
    color="black",
    s=60,
    zorder=5
)

plt.xlabel("Date", fontsize=14)
plt.ylabel("Price", fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(False, which="both")
plt.minorticks_off()
plt.tight_layout()

plt.savefig("eurostoxx600_selected_bubbles.png", dpi=300, bbox_inches="tight")
plt.show()

print(event_df)