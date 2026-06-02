import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# 1) Import
# -----------------------

build = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/build.csv", parse_dates=["date"])

# -----------------------
# 2) Summary Statistics
# -----------------------

cols_exclude = ["date", "psy_bubble"]
summary = build.drop(columns=cols_exclude).describe().T

fig, ax = plt.subplots(figsize=(14, 0.4 * len(summary)))
ax.axis("off")

table = ax.table(
    cellText=summary.round(4).values,
    colLabels=summary.columns,
    rowLabels=summary.index,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.2)

plt.savefig(
    "summary_statistics.pdf",
    bbox_inches="tight"
)

plt.close()

# -----------------------
# 3) Plot
# -----------------------

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18
})

mask = (build["date"] >= "1990-01-01") & \
       (build["date"] <= "2025-12-31")

df_plot = build.loc[mask]

fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xlim(
    pd.Timestamp("1990-01-01"),
    pd.Timestamp("2025-12-31")
)

ax.plot(
    df_plot["date"],
    df_plot["michigan_sentiment"],
    color="#ff0000",
    label="Gold Spot Price"
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
ax.set_ylabel("Michigan Sentiment")

plt.tight_layout()
plt.savefig("michigan_sentiment.pdf", bbox_inches="tight")
plt.show()