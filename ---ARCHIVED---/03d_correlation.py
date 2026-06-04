# 03d_correlation_heatmap.py

import pandas as pd  # handle dataframes
import matplotlib.pyplot as plt  # plot charts
import numpy as np  # numerical operations

build = pd.read_csv(
    "C:/Users/Pranav/OneDrive/Desktop/Final-Year-Project/build.csv",
    parse_dates=["date"]
)  # load final build dataset

cols = build.columns.difference(["date", "psy_bubble"])  # select feature columns
data = build[cols].select_dtypes(include="number")  # keep numeric columns only

corr = data.corr()  # calculate correlation matrix

plt.rcParams["font.family"] = "Arial"  # set font

fig, ax = plt.subplots(figsize=(12, 10))  # create figure
cax = ax.imshow(corr, cmap="bwr", vmin=-1, vmax=1)  # plot correlation heatmap

ax.set_xticks(np.arange(len(corr.columns)))  # set x tick positions
ax.set_yticks(np.arange(len(corr.columns)))  # set y tick positions
ax.set_xticklabels(corr.columns, fontsize=9, rotation=90)  # set x tick labels
ax.set_yticklabels(corr.columns, fontsize=9)  # set y tick labels

cbar = fig.colorbar(cax, ax=ax)  # add colour bar
cbar.ax.tick_params(labelsize=9)  # set colour bar tick size

plt.tight_layout()  # tidy layout
plt.savefig("correlation.pdf", bbox_inches="tight")  # save heatmap
plt.show()  # show plot