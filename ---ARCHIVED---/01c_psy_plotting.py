import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------
# 1) Import cleaned Shiller data
# -----------------------

shiller_classified = pd.read_csv("shiller_data_clean.csv", parse_dates=["date"])

# -----------------------
# 2) Add bubble indicator
# -----------------------

bubble_dates = pd.read_csv(
    "label_dates.csv",
    parse_dates=["start", "end"]
)

# initialise as NA
shiller_classified["psy_bubble"] = np.nan

# only label inside the target window
window_mask = (
    (shiller_classified["date"] >= "1990-01-31") &
    (shiller_classified["date"] <= "2025-12-31")
)

shiller_classified.loc[window_mask, "psy_bubble"] = 0

# apply bubble labels
for _, row in bubble_dates.iterrows():
    mask = (
        window_mask &
        (shiller_classified["date"] >= row["start"]) &
        (shiller_classified["date"] <= row["end"])
    )
    shiller_classified.loc[mask, "psy_bubble"] = 1

shiller_classified.to_csv("master.csv", index=False)

# -----------------------
# 3) Plots
# -----------------------

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18
})

mask = (shiller_classified["date"] >= "1990-01-01") & \
       (shiller_classified["date"] <= "2025-12-31")

df_plot = shiller_classified.loc[mask]

fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xlim(pd.Timestamp("1990-01-01"),
            pd.Timestamp("2025-12-31"))

ax.plot(
    df_plot["date"],
    df_plot["sp_comp_p"],
    color="#ff0000",
    label="S&P 500 Price"
)

ax.fill_between(
    df_plot["date"],
    0, 1,
    where=df_plot["psy_bubble"] == 1,
    transform=ax.get_xaxis_transform(),
    color="#ababab",
    alpha=0.4,
    label="PSY Bubble"
)

ax.set_xlabel("Year")
ax.set_ylabel("Price")

plt.tight_layout()
plt.savefig("sp500_price.pdf", bbox_inches="tight")
plt.show()