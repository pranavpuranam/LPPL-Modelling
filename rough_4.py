import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


CSV_FILE = "lppl_results_gfc_2007.csv"

df = pd.read_csv(CSV_FILE)

df["tc_predicted"] = pd.to_datetime(df["tc_predicted"])
df["tc_true"] = pd.to_datetime(df["tc_true"])

# Keep only valid LPPL fits
df = df[df["check_pass"] == True].copy()

if df.empty:
    raise ValueError("No valid rows where check_pass == True.")

tc_true = df["tc_true"].iloc[0]

# Convert dates to numeric format for KDE
tc_numeric = df["tc_predicted"].map(pd.Timestamp.toordinal).to_numpy()
tc_true_numeric = tc_true.toordinal()

# KDE
kde = gaussian_kde(tc_numeric)

x_min = tc_numeric.min() - 30
x_max = tc_numeric.max() + 30
x_grid = np.linspace(x_min, x_max, 500)

density = kde(x_grid)

# Convert x-axis back to dates
x_dates = [pd.Timestamp.fromordinal(int(x)) for x in x_grid]

plt.figure(figsize=(10, 6))

plt.plot(
    x_dates,
    density,
    color="steelblue",
    linewidth=2,
    label="KDE of predicted $t_c$"
)

plt.fill_between(
    x_dates,
    density,
    color="steelblue",
    alpha=0.25
)

plt.axvline(
    tc_true,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"True $t_c$ = {tc_true.date()}"
)

plt.xlabel("Predicted critical time $t_c$")
plt.ylabel("Density")
plt.title("KDE of LPPL predicted critical times")
plt.legend()
plt.tight_layout()
plt.show()