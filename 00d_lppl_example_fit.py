# 00d: plot example lppl fit

import pandas as pd  # handle dataframes
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plot charts
import matplotlib.dates as mdates  # format date axis
from scipy.optimize import differential_evolution, minimize  # optimise lppl fit

CSV_FILE = "eurostoxx600_prices.csv"  # input price data
INDEX_NAME = "STOXX Europe 600"  # index label
END_DATE = "2007-01-01"  # fit endpoint
WINDOW_SIZE = 450  # fitting window length
CONTEXT_DAYS = 50  # extra plot context
TC_MIN_DAYS = 1  # minimum tc after t2
TC_MAX_DAYS = 180  # maximum tc after t2
M_BOUNDS = (0.1, 0.9)  # m bounds
OMEGA_BOUNDS = (6.0, 13.0)  # omega bounds
SEED = 42  # random seed
MAXITER = 5000  # local optimiser iterations
LPPL_COLOR = "red"  # lppl line colour
TC_COLOR = "red"  # tc line colour
EXTRAPOLATION_POINTS = 300  # dashed curve points
TC_LABEL_X_OFFSET_DAYS = -105  # tc label x offset
TC_LABEL_Y_FRAC_FROM_BOTTOM = 0.04  # tc label y offset

df = pd.read_csv(CSV_FILE)  # load price data
df["Date"] = pd.to_datetime(df["Date"])  # parse dates
df = df.sort_values("Date").reset_index(drop=True)  # sort chronologically

if "log_price" not in df.columns:  # check log price exists
    raise ValueError("CSV file must contain a column named 'log_price'.")  # stop if missing

df = df.dropna(subset=["log_price"]).reset_index(drop=True)  # remove missing log prices
end_date = pd.to_datetime(END_DATE)  # parse endpoint
df_until_end = df[df["Date"] <= end_date].copy()  # keep data up to endpoint

if len(df_until_end) < WINDOW_SIZE:  # check enough data
    raise ValueError(
        f"Not enough data before {END_DATE}. "
        f"Requested {WINDOW_SIZE}, available {len(df_until_end)}."
    )  # stop if window too long

window = df_until_end.iloc[-WINDOW_SIZE:].copy()  # select fitting window
start_date = window["Date"].iloc[0]  # get t1 date
actual_end_date = window["Date"].iloc[-1]  # get t2 date
t0 = start_date  # set time origin
window["t"] = (window["Date"] - t0).dt.days  # convert dates to days
t = window["t"].to_numpy(dtype=float)  # get time array
y = window["log_price"].to_numpy(dtype=float)  # get log price array
t_last = t.max()  # get final time

def solve_linear_params(t, y, tc, m, omega):
    dt = tc - t  # time to critical time
    if np.any(dt <= 0):  # check valid domain
        return None, np.inf  # reject invalid tc
    f = dt ** m  # power-law term
    g = f * np.cos(omega * np.log(dt))  # cosine term
    h = f * np.sin(omega * np.log(dt))  # sine term
    X = np.column_stack([
        np.ones_like(t),
        f,
        g,
        h
    ])  # build linear design matrix
    try:
        params, *_ = np.linalg.lstsq(X, y, rcond=None)  # estimate linear parameters
        y_hat = X @ params  # fitted values
        sse = np.sum((y - y_hat) ** 2)  # sum squared error
        return params, sse  # return fit
    except np.linalg.LinAlgError:
        return None, np.inf  # reject failed fit

def objective(theta):
    tc, m, omega = theta  # unpack nonlinear parameters
    if not (t_last + TC_MIN_DAYS <= tc <= t_last + TC_MAX_DAYS):  # check tc bounds
        return 1e50  # penalise invalid tc
    if not (M_BOUNDS[0] <= m <= M_BOUNDS[1]):  # check m bounds
        return 1e50  # penalise invalid m
    if not (OMEGA_BOUNDS[0] <= omega <= OMEGA_BOUNDS[1]):  # check omega bounds
        return 1e50  # penalise invalid omega
    _, sse = solve_linear_params(t, y, tc, m, omega)  # solve linear parameters
    return sse  # minimise sse

def lppl_value(t_values, A, B, C1, C2, tc, m, omega):
    t_values = np.asarray(t_values, dtype=float)  # convert to array
    dt = tc - t_values  # time to critical time
    if np.any(dt <= 0):  # check valid domain
        raise ValueError("LPPL cannot be evaluated at or after tc because log(tc - t) is undefined.")  # stop if invalid
    return (
        A
        + B * dt ** m
        + C1 * dt ** m * np.cos(omega * np.log(dt))
        + C2 * dt ** m * np.sin(omega * np.log(dt))
    )  # evaluate lppl function

bounds = [
    (t_last + TC_MIN_DAYS, t_last + TC_MAX_DAYS),
    M_BOUNDS,
    OMEGA_BOUNDS
]  # nonlinear parameter bounds

result_de = differential_evolution(
    objective,
    bounds=bounds,
    seed=SEED,
    polish=False,
    workers=1
)  # run global optimisation

result = minimize(
    objective,
    result_de.x,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": MAXITER}
)  # refine with local optimisation

tc, m, omega = result.x  # get nonlinear parameters
linear_params, sse = solve_linear_params(t, y, tc, m, omega)  # estimate linear parameters

if linear_params is None:  # check fit success
    raise RuntimeError("LPPL fit failed.")  # stop if failed

A, B, C1, C2 = linear_params  # unpack linear parameters
C = np.sqrt(C1 ** 2 + C2 ** 2)  # recover amplitude
phi = np.arctan2(C2, C1)  # recover phase
tc_date = t0 + pd.Timedelta(days=float(tc))  # convert tc to date

y_fit_window = lppl_value(
    t_values=t,
    A=A,
    B=B,
    C1=C1,
    C2=C2,
    tc=tc,
    m=m,
    omega=omega
)  # fitted curve over window

if tc > t_last:  # check extrapolation possible
    t_extrap = np.linspace(
        t_last,
        tc - 1e-6,
        EXTRAPOLATION_POINTS
    )  # create extrapolated times
    y_fit_extrap = lppl_value(
        t_values=t_extrap,
        A=A,
        B=B,
        C1=C1,
        C2=C2,
        tc=tc,
        m=m,
        omega=omega
    )  # evaluate extrapolated curve
    extrap_dates = [
        t0 + pd.Timedelta(days=float(tt))
        for tt in t_extrap
    ]  # convert extrapolation to dates
else:
    t_extrap = np.array([])  # empty extrapolated times
    y_fit_extrap = np.array([])  # empty extrapolated fit
    extrap_dates = []  # empty extrapolated dates

print(f"\nLPPL/JLS fit: {INDEX_NAME}")  # print fit title
print("------------------------------------------")  # print divider
print(f"Fit window:       {start_date.date()} to {actual_end_date.date()}")  # print window dates
print(f"Window size:      {WINDOW_SIZE} trading observations")  # print window size
print(f"Estimated tc:     {tc_date.date()}")  # print critical time
print(f"Days after end:   {(tc_date - actual_end_date).days}")  # print tc offset
print(f"SSE:              {sse:.6f}")  # print error

print("\nParameters:")  # print parameter header
print(f"A      = {A:.6f}")  # print A
print(f"B      = {B:.6f}")  # print B
print(f"C1     = {C1:.6f}")  # print C1
print(f"C2     = {C2:.6f}")  # print C2
print(f"C      = {C:.6f}")  # print C
print(f"m      = {m:.6f}")  # print m
print(f"omega  = {omega:.6f}")  # print omega
print(f"phi    = {phi:.6f}")  # print phi

plot_start_date = start_date - pd.Timedelta(days=CONTEXT_DAYS)  # set plot start
plot_end_date = tc_date + pd.Timedelta(days=CONTEXT_DAYS)  # set plot end
plot_df = df[
    (df["Date"] >= plot_start_date) &
    (df["Date"] <= plot_end_date)
].copy()  # slice plot data
before_window = plot_df[plot_df["Date"] < start_date]  # data before window
inside_window = plot_df[
    (plot_df["Date"] >= start_date) &
    (plot_df["Date"] <= actual_end_date)
]  # data inside window
after_window = plot_df[plot_df["Date"] > actual_end_date]  # data after window

plt.rcParams["font.family"] = "Arial"  # set font
plt.rcParams["font.size"] = 24  # set font size
fig, ax = plt.subplots(figsize=(12, 8))  # create figure

ax.plot(
    before_window["Date"],
    before_window["log_price"],
    linestyle=":",
    linewidth=1.8,
    color="darkgrey"
)  # plot context before window

ax.plot(
    inside_window["Date"],
    inside_window["log_price"],
    linestyle=":",
    linewidth=1.8,
    color="black"
)  # plot fitted window data

ax.plot(
    after_window["Date"],
    after_window["log_price"],
    linestyle=":",
    linewidth=1.8,
    color="darkgrey"
)  # plot context after window

ax.plot(
    window["Date"],
    y_fit_window,
    color=LPPL_COLOR,
    linestyle="-",
    linewidth=2.2,
    label="LPPL fit"
)  # plot solid fitted curve

if len(extrap_dates) > 0:  # check extrapolated curve exists
    ax.plot(
        extrap_dates,
        y_fit_extrap,
        color=LPPL_COLOR,
        linestyle="--",
        linewidth=2.2,
        label="LPPL extrapolation"
    )  # plot dashed extrapolation

ax.axvline(
    tc_date,
    color=TC_COLOR,
    linestyle="-",
    linewidth=1.0,
    label=r"Predicted $t_c$"
)  # plot critical time line

ax.set_xlabel("Date", fontsize=24, fontname="Arial")  # set x label
ax.set_ylabel("Log Price", fontsize=24, fontname="Arial")  # set y label
ax.minorticks_off()  # remove minor ticks
ax.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.5)  # add grid
ax.xaxis.set_major_locator(mdates.YearLocator())  # set yearly ticks
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # format year labels
ax.tick_params(axis="both", labelsize=24)  # set tick size
ax.set_xlim(plot_df["Date"].min(), plot_df["Date"].max())  # set x limits
ax.margins(x=0)  # remove x padding

y_components = [
    before_window["log_price"],
    inside_window["log_price"],
    after_window["log_price"],
    pd.Series(y_fit_window)
]  # collect y values

if len(y_fit_extrap) > 0:  # check extrapolated y values
    y_components.append(pd.Series(y_fit_extrap))  # add extrapolated y values

y_all = pd.concat(y_components, ignore_index=True)  # combine y values
ymin = y_all.min()  # get y min
ymax = y_all.max()  # get y max
yrange = ymax - ymin  # get y range
ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.05 * yrange)  # set y limits

label_x = tc_date + pd.Timedelta(days=TC_LABEL_X_OFFSET_DAYS)  # set label x
label_y = ymin + TC_LABEL_Y_FRAC_FROM_BOTTOM * yrange  # set label y

ax.text(
    label_x,
    label_y,
    f"$t_c$ = {tc_date.date()}",
    color=TC_COLOR,
    ha="center",
    va="bottom",
    fontsize=24
)  # add critical time label

ax.legend(
    prop={"family": "Arial", "size": 24},
    frameon=False
)  # add legend

plt.tight_layout()  # tidy layout
safe_name = INDEX_NAME.lower().replace(" ", "_").replace("&", "and")  # create safe filename
output_name = f"{safe_name}_lppl_fit_{actual_end_date.date()}_{WINDOW_SIZE}obs.png"  # create output name
plt.savefig(output_name, dpi=300, bbox_inches="tight")  # save chart
plt.show()  # show chart
print(f"\nSaved chart as: {output_name}")  # print save path