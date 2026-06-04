# 02c_summary_and_michigan_plot.py

import pandas as pd  # handle dataframes
import matplotlib.pyplot as plt  # plot charts

build = pd.read_csv(
    "C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/build.csv",
    parse_dates=["date"]
)  # load final build dataset

cols_exclude = ["date", "psy_bubble"]  # columns not used in summary stats
summary = build.drop(columns=cols_exclude).describe().T  # calculate summary statistics

fig, ax = plt.subplots(figsize=(14, 0.4 * len(summary)))  # create table figure
ax.axis("off")  # hide axes

table = ax.table(
    cellText=summary.round(4).values,
    colLabels=summary.columns,
    rowLabels=summary.index,
    loc="center"
)  # create summary statistics table

table.auto_set_font_size(False)  # disable automatic font sizing
table.set_fontsize(8)  # set table font size
table.scale(1, 1.2)  # scale table spacing

plt.savefig(
    "summary_statistics.pdf",
    bbox_inches="tight"
)  # save summary statistics table

plt.close()  # close table figure

plt.rcParams.update({
    "font.family": "Arial",  # set font
    "font.size": 18  # set font size
})  # update plot style

mask = (build["date"] >= "1990-01-01") & \
       (build["date"] <= "2025-12-31")  # define plotting window

df_plot = build.loc[mask]  # filter data for plot

fig, ax = plt.subplots(figsize=(8, 6))  # create figure

ax.set_xlim(
    pd.Timestamp("1990-01-01"),
    pd.Timestamp("2025-12-31")
)  # set x axis limits

ax.plot(
    df_plot["date"],
    df_plot["michigan_sentiment"],
    color="#ff0000",
    label="Michigan Sentiment"
)  # plot michigan sentiment

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
ax.set_ylabel("Michigan Sentiment")  # set y label

plt.tight_layout()  # tidy layout
plt.savefig("michigan_sentiment.pdf", bbox_inches="tight")  # save plot
plt.show()  # show plot