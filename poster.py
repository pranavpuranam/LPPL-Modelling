# ============================================================
# Single-window LPPL/JLS fit and plot
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.optimize import differential_evolution, minimize


# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "eurostoxx600_prices.csv"
INDEX_NAME = "STOXX Europe 600"

END_DATE = "2007-01-01"
WINDOW_SIZE = 450

CONTEXT_DAYS = 50

TC_MIN_DAYS = 1
TC_MAX_DAYS = 180

M_BOUNDS = (0.1, 0.9)
OMEGA_BOUNDS = (6.0, 13.0)

SEED = 42
MAXITER = 5000

LPPL_COLOR = "red"
TC_COLOR = "red"

TC_LABEL_X_OFFSET_DAYS = -105
TC_LABEL_Y_FRAC_FROM_BOTTOM = 0.04

FIG_WIDTH = 17.76
FIG_HEIGHT = 9.95
DPI = 300


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_FILE)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

if "log_price" not in df.columns:
    raise ValueError("CSV file must contain a column named 'log_price'.")

df = df.dropna(subset=["log_price"]).reset_index(drop=True)

end_date = pd.to_datetime(END_DATE)
df_until_end = df[df["Date"] <= end_date].copy()

if len(df_until_end) < WINDOW_SIZE:
    raise ValueError(
        f"Not enough data before {END_DATE}. "
        f"Requested {WINDOW_SIZE}, available {len(df_until_end)}."
    )

window = df_until_end.iloc[-WINDOW_SIZE:].copy()

start_date = window["Date"].iloc[0]
actual_end_date = window["Date"].iloc[-1]

t0 = start_date
window["t"] = (window["Date"] - t0).dt.days

t = window["t"].to_numpy(dtype=float)
y = window["log_price"].to_numpy(dtype=float)

t_last = t.max()


# ============================================================
# LPPL FUNCTIONS
# ============================================================

def solve_linear_params(t, y, tc, m, omega):
    dt = tc - t

    if np.any(dt <= 0):
        return None, np.inf

    f = dt ** m
    g = f * np.cos(omega * np.log(dt))
    h = f * np.sin(omega * np.log(dt))

    X = np.column_stack([
        np.ones_like(t),
        f,
        g,
        h
    ])

    try:
        params, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ params
        sse = np.sum((y - y_hat) ** 2)
        return params, sse

    except np.linalg.LinAlgError:
        return None, np.inf


def objective(theta):
    tc, m, omega = theta

    if not (t_last + TC_MIN_DAYS <= tc <= t_last + TC_MAX_DAYS):
        return 1e50

    if not (M_BOUNDS[0] <= m <= M_BOUNDS[1]):
        return 1e50

    if not (OMEGA_BOUNDS[0] <= omega <= OMEGA_BOUNDS[1]):
        return 1e50

    _, sse = solve_linear_params(t, y, tc, m, omega)
    return sse


# ============================================================
# OPTIMISATION
# ============================================================

bounds = [
    (t_last + TC_MIN_DAYS, t_last + TC_MAX_DAYS),
    M_BOUNDS,
    OMEGA_BOUNDS
]

result_de = differential_evolution(
    objective,
    bounds=bounds,
    seed=SEED,
    polish=False,
    workers=1
)

result = minimize(
    objective,
    result_de.x,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": MAXITER}
)

tc, m, omega = result.x

linear_params, sse = solve_linear_params(t, y, tc, m, omega)

if linear_params is None:
    raise RuntimeError("LPPL fit failed.")

A, B, C1, C2 = linear_params

C = np.sqrt(C1 ** 2 + C2 ** 2)
phi = np.arctan2(C2, C1)

tc_date = t0 + pd.Timedelta(days=float(tc))


# ============================================================
# FITTED CURVE
# ============================================================

dt = tc - t

y_fit = (
    A
    + B * dt ** m
    + C1 * dt ** m * np.cos(omega * np.log(dt))
    + C2 * dt ** m * np.sin(omega * np.log(dt))
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(f"\nLPPL/JLS fit: {INDEX_NAME}")
print("------------------------------------------")
print(f"Fit window:       {start_date.date()} to {actual_end_date.date()}")
print(f"Window size:      {WINDOW_SIZE} trading observations")
print(f"Estimated tc:     {tc_date.date()}")
print(f"Days after end:   {(tc_date - actual_end_date).days}")
print(f"SSE:              {sse:.6f}")

print("\nParameters:")
print(f"A      = {A:.6f}")
print(f"B      = {B:.6f}")
print(f"C1     = {C1:.6f}")
print(f"C2     = {C2:.6f}")
print(f"C      = {C:.6f}")
print(f"m      = {m:.6f}")
print(f"omega  = {omega:.6f}")
print(f"phi    = {phi:.6f}")


# ============================================================
# PLOT DATA
# ============================================================

plot_start_date = start_date - pd.Timedelta(days=CONTEXT_DAYS)
plot_end_date = tc_date + pd.Timedelta(days=CONTEXT_DAYS)

plot_df = df[
    (df["Date"] >= plot_start_date) &
    (df["Date"] <= plot_end_date)
].copy()

before_window = plot_df[plot_df["Date"] < start_date]

inside_window = plot_df[
    (plot_df["Date"] >= start_date) &
    (plot_df["Date"] <= actual_end_date)
]

after_window = plot_df[plot_df["Date"] > actual_end_date]


# ============================================================
# PLOT
# ============================================================

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 34

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

# Thicker stock-price lines
ax.plot(
    before_window["Date"],
    before_window["log_price"],
    linestyle=":",
    linewidth=3.2,
    color="darkgrey"
)

ax.plot(
    inside_window["Date"],
    inside_window["log_price"],
    linestyle=":",
    linewidth=3.2,
    color="black"
)

ax.plot(
    after_window["Date"],
    after_window["log_price"],
    linestyle=":",
    linewidth=3.2,
    color="darkgrey"
)

# Thicker LPPL fit line
ax.plot(
    window["Date"],
    y_fit,
    color=LPPL_COLOR,
    linewidth=4.2,
    label="LPPL fit"
)

# Thicker tc line
ax.axvline(
    tc_date,
    color=TC_COLOR,
    linestyle="-",
    linewidth=3.2,
    label=r"Predicted $t_c$"
)

ax.set_xlabel("Date", fontsize=34, fontname="Arial")
ax.set_ylabel("Log Price", fontsize=34, fontname="Arial")

ax.minorticks_off()
ax.tick_params(axis="both", labelsize=34)

ax.set_xlim(plot_df["Date"].min(), plot_df["Date"].max())
ax.margins(x=0)

y_all = pd.concat([
    before_window["log_price"],
    inside_window["log_price"],
    after_window["log_price"],
    pd.Series(y_fit)
], ignore_index=True)

ymin = y_all.min()
ymax = y_all.max()
yrange = ymax - ymin

if yrange == 0:
    yrange = 1

ymin_plot = ymin - 0.05 * yrange
ymax_plot = ymax + 0.05 * yrange
ax.set_ylim(ymin_plot, ymax_plot)

# Back to automatic axis labeling / old style
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.grid(True, which="major", linestyle="-", linewidth=1.2, alpha=0.5)

label_x = tc_date + pd.Timedelta(days=TC_LABEL_X_OFFSET_DAYS)
label_y = ymin + TC_LABEL_Y_FRAC_FROM_BOTTOM * yrange

ax.text(
    label_x,
    label_y,
    f"$t_c$ = {tc_date.date()}",
    color=TC_COLOR,
    ha="center",
    va="bottom",
    fontsize=34,
    fontname="Arial"
)

ax.legend(
    prop={"family": "Arial", "size": 34},
    frameon=False
)

plt.tight_layout()

safe_name = INDEX_NAME.lower().replace(" ", "_").replace("&", "and")
output_name = f"{safe_name}_lppl_fit_{actual_end_date.date()}_{WINDOW_SIZE}obs.png"

# Keep final PNG dimensions as requested
plt.savefig(output_name, dpi=DPI)
plt.show()

print(f"\nSaved chart as: {output_name}")