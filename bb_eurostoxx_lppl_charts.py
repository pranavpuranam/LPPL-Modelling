# pip install pandas numpy scipy matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

# ============================================================
# USER INPUTS
# ============================================================
CSV_FILE = "eurostoxx600_prices.csv"
INDEX_NAME = "STOXX Europe 600"

END_DATE = "2007-01-01"     # yyyy-mm-dd
WINDOW_SIZE = 450           # number of trading observations, must be >= 30

CONTEXT_DAYS = 200           # calendar days shown before window and after estimated tc

TC_MIN_DAYS = 1             # minimum tc after END_DATE, calendar days
TC_MAX_DAYS = 180           # maximum tc after END_DATE, calendar days

# LPPL parameter bounds
M_BOUNDS = (0.1, 0.9)
OMEGA_BOUNDS = (6.0, 13.0)

# ============================================================
# Checks
# ============================================================
if not isinstance(WINDOW_SIZE, int) or WINDOW_SIZE < 30:
    raise ValueError("WINDOW_SIZE must be an integer >= 30.")

if CONTEXT_DAYS < 0:
    raise ValueError("CONTEXT_DAYS must be non-negative.")

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(CSV_FILE)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

if "log_price" not in df.columns:
    raise ValueError("CSV file must contain a column named 'log_price'.")

df = df.dropna(subset=["log_price"]).reset_index(drop=True)

end_date = pd.to_datetime(END_DATE)

# Keep data up to chosen end date for fitting
df_until_end = df[df["Date"] <= end_date].copy()

if len(df_until_end) < WINDOW_SIZE:
    raise ValueError(
        f"Not enough data before {END_DATE}. "
        f"Requested {WINDOW_SIZE} observations, available {len(df_until_end)}."
    )

# Select fitting window: last WINDOW_SIZE observations up to END_DATE
window = df_until_end.iloc[-WINDOW_SIZE:].copy()

start_date = window["Date"].iloc[0]
actual_end_date = window["Date"].iloc[-1]

# Time variable in calendar days from start of fitting window
t0 = start_date
window["t"] = (window["Date"] - t0).dt.days

t = window["t"].values.astype(float)
y = window["log_price"].values.astype(float)
t_last = t.max()

# ============================================================
# LPPL helper functions
# ============================================================
def solve_linear_params(t, y, tc, m, omega):
    """
    For fixed nonlinear parameters tc, m, omega,
    solve linear parameters A, B, C1, C2 by least squares.
    """
    dt = tc - t

    if np.any(dt <= 0):
        return None, np.inf

    f = dt ** m
    g = f * np.cos(omega * np.log(dt))
    h = f * np.sin(omega * np.log(dt))

    X = np.column_stack([np.ones_like(t), f, g, h])

    try:
        params, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ params
        sse = np.sum((y - y_hat) ** 2)
        return params, sse
    except np.linalg.LinAlgError:
        return None, np.inf


def objective(theta):
    tc, m, omega = theta

    # Safety penalty in case optimiser tests invalid values
    if not (t_last + TC_MIN_DAYS <= tc <= t_last + TC_MAX_DAYS):
        return 1e50
    if not (M_BOUNDS[0] <= m <= M_BOUNDS[1]):
        return 1e50
    if not (OMEGA_BOUNDS[0] <= omega <= OMEGA_BOUNDS[1]):
        return 1e50

    _, sse = solve_linear_params(t, y, tc, m, omega)
    return sse


# ============================================================
# Bounds
# ============================================================
bounds = [
    (t_last + TC_MIN_DAYS, t_last + TC_MAX_DAYS),  # tc
    M_BOUNDS,                                      # m
    OMEGA_BOUNDS                                   # omega
]

# ============================================================
# Optimisation
# ============================================================
result_de = differential_evolution(
    objective,
    bounds=bounds,
    seed=42,
    polish=False,
    workers=1
)

result = minimize(
    objective,
    result_de.x,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 5000}
)

tc, m, omega = result.x
linear_params, sse = solve_linear_params(t, y, tc, m, omega)

if linear_params is None:
    raise RuntimeError("LPPL fit failed.")

if not (t_last + TC_MIN_DAYS <= tc <= t_last + TC_MAX_DAYS):
    raise RuntimeError(f"Invalid tc found: {tc}")

A, B, C1, C2 = linear_params

C = np.sqrt(C1**2 + C2**2)
phi = np.arctan2(C2, C1)

tc_date = t0 + pd.Timedelta(days=float(tc))

# ============================================================
# Fitted LPPL curve over fitting window
# ============================================================
dt = tc - t

y_fit = (
    A
    + B * dt**m
    + C1 * dt**m * np.cos(omega * np.log(dt))
    + C2 * dt**m * np.sin(omega * np.log(dt))
)

# ============================================================
# Concise output
# ============================================================
print(f"\nLPPL / First-order JLS fit: {INDEX_NAME}")
print("------------------------------------------")
print(f"Fit window:       {start_date.date()} to {actual_end_date.date()}")
print(f"Window size:      {WINDOW_SIZE} trading observations")
print(f"Estimated tc:     {tc_date.date()}")
print(f"Days after end:   {(tc_date - actual_end_date).days}")
print(f"SSE:              {sse:.6f}")

print("\nModel:")
print(
    "ln p(t) = A + B(tc-t)^m "
    "+ C1(tc-t)^m cos(omega ln(tc-t)) "
    "+ C2(tc-t)^m sin(omega ln(tc-t))"
)

print("\nParameters:")
print(f"A      = {A:.6f}")
print(f"B      = {B:.6f}")
print(f"C1     = {C1:.6f}")
print(f"C2     = {C2:.6f}")
print(f"C      = {C:.6f}")
print(f"m      = {m:.6f}")
print(f"omega  = {omega:.6f}")
print(f"phi    = {phi:.6f}")

print("\nValidity checks:")
print(f"m in [{M_BOUNDS[0]}, {M_BOUNDS[1]}]:       {M_BOUNDS[0] <= m <= M_BOUNDS[1]}")
print(f"omega in [{OMEGA_BOUNDS[0]}, {OMEGA_BOUNDS[1]}]: {OMEGA_BOUNDS[0] <= omega <= OMEGA_BOUNDS[1]}")
print(f"B < 0:                    {B < 0}")
print(f"tc after window end:       {tc_date > actual_end_date}")

# ============================================================
# Dynamic context data for plotting
# ============================================================
plot_start_date = start_date - pd.Timedelta(days=CONTEXT_DAYS)
plot_end_date = tc_date + pd.Timedelta(days=CONTEXT_DAYS)

plot_df = df[
    (df["Date"] >= plot_start_date) &
    (df["Date"] <= plot_end_date)
].copy()

# ============================================================
# Plot
# ============================================================
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 11

plt.figure(figsize=(9, 6))

before_window = plot_df[plot_df["Date"] < start_date]
inside_window = plot_df[
    (plot_df["Date"] >= start_date) & (plot_df["Date"] <= actual_end_date)
]
after_window = plot_df[plot_df["Date"] > actual_end_date]

# Observed log price before fitting window
plt.plot(
    before_window["Date"],
    before_window["log_price"],
    linestyle=":",
    linewidth=1.8,
    color="darkgrey",
    label="_nolegend_"
)

# Observed log price inside fitting window
plt.plot(
    inside_window["Date"],
    inside_window["log_price"],
    linestyle=":",
    linewidth=1.8,
    color="black",
    label="Observed Log Price"
)

# Observed log price after fitting window
plt.plot(
    after_window["Date"],
    after_window["log_price"],
    linestyle=":",
    linewidth=1.8,
    color="darkgrey",
    label="_nolegend_"
)

# JLS fitted line only over fitting window
plt.plot(
    window["Date"],
    y_fit,
    color="red",
    linestyle="-",
    linewidth=2.2,
    label="JLS Fit"
)

# Predicted tc line
plt.axvline(
    tc_date,
    color="red",
    linestyle="--",
    linewidth=1.0,
    label=f"Predicted $t_c$ = {tc_date.date()}"
)

plt.xlabel("Date", fontsize=11, fontname="Arial")
plt.ylabel("Log Price", fontsize=11, fontname="Arial")

plt.grid(False)
plt.minorticks_off()

plt.xticks(fontsize=11, fontname="Arial")
plt.yticks(fontsize=11, fontname="Arial")

plt.legend(prop={"family": "Arial", "size": 11})
plt.tight_layout()

safe_name = INDEX_NAME.lower().replace(" ", "_").replace("&", "and")
output_name = f"{safe_name}_lppl_fit_{actual_end_date.date()}_{WINDOW_SIZE}obs.png"

plt.savefig(output_name, dpi=300)
plt.show()

print(f"\nSaved chart as: {output_name}")