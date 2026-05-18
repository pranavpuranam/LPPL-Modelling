import pandas as pd
import matplotlib.pyplot as plt


CSV_FILE = "lppl_results_gfc_2007.csv"

df = pd.read_csv(CSV_FILE)

df["t1"] = pd.to_datetime(df["t1"])
df["t2"] = pd.to_datetime(df["t2"])
df["tc_predicted"] = pd.to_datetime(df["tc_predicted"])
df["tc_true"] = pd.to_datetime(df["tc_true"])

# Keep only successful / valid rows
df = df[df["check_pass"] == True].copy()

tc_true = df["tc_true"].iloc[0]

plt.figure(figsize=(10, 6))

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
    linewidth=2,
    label=f"True $t_c$ = {tc_true.date()}"
)

plt.xlabel("Fitting window end date $t_2$")
plt.ylabel("Predicted critical time $t_c$")
plt.title("LPPL predicted critical times")

plt.legend()
plt.tight_layout()
plt.show()