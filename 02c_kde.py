import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_tc_kde(csv_file, output_file=None, valid_only=True):
    df = pd.read_csv(csv_file)

    df["tc_predicted"] = pd.to_datetime(df["tc_predicted"])
    df["tc_true"] = pd.to_datetime(df["tc_true"])

    if valid_only:
        df = df[df["check_pass"] == True].copy()

    if df.empty:
        raise ValueError("No rows available after filtering.")

    tc_true = df["tc_true"].iloc[0]
    tc_numeric = df["tc_predicted"].map(pd.Timestamp.toordinal).to_numpy()

    kde = gaussian_kde(tc_numeric)

    data_min = tc_numeric.min()
    data_max = tc_numeric.max()
    data_range = data_max - data_min
    padding = max(60, 0.35 * data_range)

    x_min = min(data_min - padding, tc_true.toordinal() - padding)
    x_max = max(data_max + padding, tc_true.toordinal() + padding)

    x_grid = np.linspace(x_min, x_max, 1000)
    kde_density = kde(x_grid)
    x_dates = [pd.Timestamp.fromordinal(int(x)) for x in x_grid]

    if output_file is None:
        output_file = csv_file.replace(".csv", "_kde.png")

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 11

    plt.figure(figsize=(9, 6))

    plt.plot(x_dates, kde_density, color="black", linewidth=2.2,
             label="KDE of predicted $t_c$")

    plt.fill_between(x_dates, kde_density, color="darkgrey", alpha=0.65)

    plt.axvline(tc_true, color="red", linestyle="--", linewidth=1.2,
                label=f"True $t_c$ = {tc_true.date()}")

    plt.xlabel("Predicted Critical Time $t_c$")
    plt.ylabel("Density")

    plt.grid(False)
    plt.minorticks_off()
    plt.legend(prop={"family": "Arial", "size": 11})
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figure saved as: {output_file}")


# Example use
plot_tc_kde("baseline_gfc.csv")