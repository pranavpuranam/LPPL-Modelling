import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "lppl_results_gfc_2007.csv"

df = pd.read_csv(CSV_FILE)

df["t1"] = pd.to_datetime(df["t1"])
df["t2"] = pd.to_datetime(df["t2"])
df["tc_predicted"] = pd.to_datetime(df["tc_predicted"])
df["tc_true"] = pd.to_datetime(df["tc_true"])

df = df[df["check_pass"] == True].copy()

if df.empty:
    raise ValueError("No valid rows where check_pass == True.")

tc_true = df["tc_true"].iloc[0]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11

plt.figure(figsize=(9, 6))

plt.scatter(
    df["t2"],
    df["tc_predicted"],
    s=25,
    alpha=0.7,
    color="steelblue",
    label="Predicted $t_c$"
)

plt.axhline(
    tc_true,
    color="red",
    linestyle="--",
    linewidth=1.2,
    label=f"True $t_c$ = {tc_true.date()}"
)

plt.xlabel("Fitting Window End Date $t_2$", fontsize=11, fontname="Arial")
plt.ylabel("Predicted Critical Time $t_c$", fontsize=11, fontname="Arial")

plt.grid(False)
plt.minorticks_off()

plt.xticks(fontsize=11, fontname="Arial")
plt.yticks(fontsize=11, fontname="Arial")

plt.legend(prop={"family": "Arial", "size": 11})
plt.tight_layout()

plt.savefig("lppl_tc_plot_gfc_2007.png", dpi=300)
plt.show()