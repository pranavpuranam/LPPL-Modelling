# 01c_apply_psy_labels.py

import pandas as pd  # handle dataframes
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plot charts
import matplotlib.dates as mdates  # format date axis

shiller_classified = pd.read_csv("shiller_data_clean.csv", parse_dates=["date"])  # load cleaned shiller data

bubble_dates = pd.read_csv(
    "label_dates.csv",
    parse_dates=["start", "end"]
)  # load bubble start and end dates

shiller_classified["psy_bubble"] = np.nan  # create empty bubble label column

window_mask = (
    (shiller_classified["date"] >= "1990-01-31") &
    (shiller_classified["date"] <= "2025-12-31")
)  # define labelling window

shiller_classified.loc[window_mask, "psy_bubble"] = 0  # set default label to no bubble

for _, row in bubble_dates.iterrows():  # loop through bubble date ranges
    mask = (
        window_mask &
        (shiller_classified["date"] >= row["start"]) &
        (shiller_classified["date"] <= row["end"])
    )  # find rows inside this bubble range
    shiller_classified.loc[mask, "psy_bubble"] = 1  # mark bubble periods

shiller_classified.to_csv("master.csv", index=False)  # save labelled master dataset

plt.rcParams.update({
    "font.family": "Arial",  # set font
    "font.size": 18  # set font size
})  # update plot style

mask = (shiller_classified["date"] >= "1990-01-01") & \
       (shiller_classified["date"] <= "2025-12-31")  # define plotting window

df_plot = shiller_classified.loc[mask]  # filter data for plot

fig, ax = plt.subplots(figsize=(8, 6))  # create figure

ax.set_xlim(
    pd.Timestamp("1990-01-01"),
    pd.Timestamp("2025-12-31")
)  # set x axis limits

ax.plot(
    df_plot["date"],
    df_plot["sp_comp_p"],
    color="#ff0000",
    label="S&P 500 Price"
)  # plot s&p 500 price

ax.fill_between(
    df_plot["date"],
    0, 1,
    where=df_plot["psy_bubble"] == 1,
    transform=ax.get_xaxis_transform(),
    color="#ababab",
    alpha=0.4,
    label="PSY Bubble"
)  # shade labelled bubble periods

ax.set_xlabel("Year")  # set x label
ax.set_ylabel("Price")  # set y label

plt.tight_layout()  # tidy layout
plt.savefig("sp500_price.pdf", bbox_inches="tight")  # save plot
plt.show()  # show plot