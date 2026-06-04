# 03a_count_plot_example.py

import pandas as pd  # handle dataframes
import matplotlib.pyplot as plt  # plot charts

build = pd.read_csv(
    "C:/Users/Pranav/OneDrive/Desktop/Final-Year-Project/build.csv",
    parse_dates=["date"]
)  # load final build dataset

var = "gold_spot_price"  # choose variable to compare
x_label = var.upper()  # create x axis label

no_bubble = build.loc[build["psy_bubble"] == 0, var].dropna()  # get values outside bubbles
bubble = build.loc[build["psy_bubble"] == 1, var].dropna()  # get values inside bubbles

plt.rcParams.update({
    "font.family": "Arial",  # set font
    "font.size": 18  # set font size
})  # update plot style

plt.figure(figsize=(8, 5))  # create figure

plt.hist(
    no_bubble,
    bins=30,
    alpha=0.6,
    label="No bubble (psy_bubble = 0)",
    color="#ff0000"
)  # plot non-bubble histogram

plt.hist(
    bubble,
    bins=30,
    alpha=0.6,
    label="Bubble (psy_bubble = 1)",
    color="#ababab"
)  # plot bubble histogram

plt.xlabel(x_label)  # set x label
plt.ylabel("Count")  # set y label
plt.legend()  # add legend

plt.tight_layout()  # tidy layout
plt.savefig("count_plot_example.pdf", bbox_inches="tight")  # save plot
plt.show()  # show plot