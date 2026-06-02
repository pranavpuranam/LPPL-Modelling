import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# 1) Import
# -----------------------
build = pd.read_csv(
    "C:/Users/Pranav/OneDrive/Desktop/Final-Year-Project/build.csv",
    parse_dates=["date"]
)

# -----------------------
# 2) Plot
# -----------------------

cols = build.columns.difference(["date", "psy_bubble"])
data = build[cols].select_dtypes(include="number")

corr = data.corr()

plt.rcParams["font.family"] = "Arial"

fig, ax = plt.subplots(figsize=(12, 10))
cax = ax.imshow(corr, cmap="bwr", vmin=-1, vmax=1)

# Ticks and labels
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns, fontsize=9, rotation=90)
ax.set_yticklabels(corr.columns, fontsize=9)

# Colorbar
cbar = fig.colorbar(cax, ax=ax)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("correlation.pdf", bbox_inches="tight")
plt.show()