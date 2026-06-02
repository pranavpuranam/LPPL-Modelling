import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# 1) Import
# -----------------------

build = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/Final-Year-Project/build.csv", parse_dates=["date"])

# -----------------------
# 2) Plot
# -----------------------

var = "gold_spot_price"          # change this to any continuous column name
x_label = var.upper()

no_bubble = build.loc[build["psy_bubble"] == 0, var].dropna()
bubble = build.loc[build["psy_bubble"] == 1, var].dropna()

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 18
})

plt.figure(figsize=(8, 5))

plt.hist(no_bubble, bins=30, alpha=0.6, label="No bubble (psy_bubble = 0)", color = "#ff0000")
plt.hist(bubble, bins=30, alpha=0.6, label="Bubble (psy_bubble = 1)", color = "#ababab")

plt.xlabel(x_label)
plt.ylabel("Count")
plt.legend()

plt.tight_layout()
plt.savefig("count_plot_example.pdf", bbox_inches="tight")
plt.show()