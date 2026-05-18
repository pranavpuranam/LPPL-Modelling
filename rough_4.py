import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm


CSV_FILE = "lppl_results_gfc_2007.csv"

df = pd.read_csv(CSV_FILE)

df["tc_predicted"] = pd.to_datetime(df["tc_predicted"])
df["tc_true"] = pd.to_datetime(df["tc_true"])

df = df[df["check_pass"] == True].copy()

if df.empty:
    raise ValueError("No valid rows where check_pass == True.")

tc_true = df["tc_true"].iloc[0]

tc_numeric = df["tc_predicted"].map(pd.Timestamp.toordinal).to_numpy()

# KDE
kde = gaussian_kde(tc_numeric)

# MLE normal fit: mean and std with ddof=0
mu = np.mean(tc_numeric)
sigma = np.std(tc_numeric, ddof=0)

if sigma == 0:
    raise ValueError("All predicted tc values are identical; normal fit is degenerate.")

# Wider automatic plotting range
data_min = tc_numeric.min()
data_max = tc_numeric.max()
data_range = data_max - data_min

padding = max(60, 0.35 * data_range)

x_min = data_min - padding
x_max = data_max + padding

# Also make sure tc_true is visible
x_min = min(x_min, tc_true.toordinal() - padding)
x_max = max(x_max, tc_true.toordinal() + padding)

x_grid = np.linspace(x_min, x_max, 1000)

kde_density = kde(x_grid)
normal_density = norm.pdf(x_grid, loc=mu, scale=sigma)

x_dates = [pd.Timestamp.fromordinal(int(x)) for x in x_grid]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11

plt.figure(figsize=(9, 6))

plt.plot(
    x_dates,
    kde_density,
    color="black",
    linewidth=2.2,
    label="KDE of predicted $t_c$"
)

plt.fill_between(
    x_dates,
    kde_density,
    color="darkgrey",
    alpha=0.65
)

plt.plot(
    x_dates,
    normal_density,
    color="red",
    linestyle="-",
    linewidth=1.6,
    label="Normal fit"
)

plt.axvline(
    tc_true,
    color="red",
    linestyle="--",
    linewidth=1.2,
    label=f"True $t_c$ = {tc_true.date()}"
)

plt.xlabel("Predicted Critical Time $t_c$", fontsize=11, fontname="Arial")
plt.ylabel("Density", fontsize=11, fontname="Arial")

plt.grid(False)
plt.minorticks_off()

plt.xticks(fontsize=11, fontname="Arial")
plt.yticks(fontsize=11, fontname="Arial")

plt.legend(prop={"family": "Arial", "size": 11})
plt.tight_layout()

plt.savefig("lppl_tc_kde_normal_gfc_2007.png", dpi=300)
plt.show()

print(f"Normal MLE mean date: {pd.Timestamp.fromordinal(int(round(mu))).date()}")
print(f"Normal MLE sigma:     {sigma:.2f} days")