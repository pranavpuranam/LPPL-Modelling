import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize


# ============================================================
# SETTINGS — DAILY POSITIVE-BUBBLE LPPLS CONFIDENCE
# ============================================================

PRICE_CSV = "eurostoxx600_prices.csv"

OUTPUT_CSV = "eurostoxx600_daily_positive_lppls_results.csv"
CONFIDENCE_CSV = "eurostoxx600_daily_positive_lppls_confidence.csv"

DATE_COL = "Date"
Y_COL = "log_price"

# Original parameter ranges
M_BOUNDS = (0.1, 0.9)
OMEGA_BOUNDS = (6.0, 13.0)

# Original rolling window setup
DT_MIN = 125
DT_MAX = 750
DT_STEP = 25

# Daily endpoint t2
T2_STEP = 1

TC_MIN_DAYS = 1
TC_MAX_DAYS = 500

SEED = 42
MAXITER = 5000

REQUIRE_FULL_SCALE = True


# ============================================================
# DATA
# ============================================================

def load_prices(csv_file=PRICE_CSV):
    df = pd.read_csv(csv_file)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["log_price"] = pd.to_numeric(df["log_price"], errors="coerce")

    return (
        df.dropna(subset=[DATE_COL, "Close", "log_price"])
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )


# ============================================================
# WINDOW GENERATION
# ============================================================

def generate_rolling_lppl_windows(
    df,
    date_col=DATE_COL,
    t2_step=T2_STEP,
    dt_min=DT_MIN,
    dt_max=DT_MAX,
    dt_step=DT_STEP,
    require_full_scale=REQUIRE_FULL_SCALE,
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    rows = []

    t2_start_idx = dt_max if require_full_scale else dt_min

    for t2_idx in range(t2_start_idx, len(df), t2_step):
        t2_date = df.loc[t2_idx, date_col]

        for dt in range(dt_min, dt_max + 1, dt_step):
            t1_idx = t2_idx - dt

            if t1_idx < 0:
                continue

            rows.append({
                "t1": df.loc[t1_idx, date_col].strftime("%Y-%m-%d"),
                "t2": t2_date.strftime("%Y-%m-%d"),
                "window_dt_requested": dt,
            })

    return pd.DataFrame(rows)


# ============================================================
# LPPLS MODEL FITTING
# ============================================================

def solve_linear_params(t, y, tc, m, omega):
    tau = tc - t

    if np.any(tau <= 0):
        return None, np.inf, None

    f = tau ** m
    g = f * np.cos(omega * np.log(tau))
    h = f * np.sin(omega * np.log(tau))

    X = np.column_stack([np.ones_like(t), f, g, h])

    try:
        params, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ params
        residuals = y - y_hat

        sse = np.sum(residuals ** 2)
        rmse = np.sqrt(np.mean(residuals ** 2))

        return params, sse, rmse

    except np.linalg.LinAlgError:
        return None, np.inf, None


def fit_lppl_row(
    df,
    t1,
    t2,
    global_tc_upper_date,
    date_col=DATE_COL,
    y_col=Y_COL,
    tc_min_days=TC_MIN_DAYS,
    seed=SEED,
    maxiter=MAXITER,
):
    t1 = pd.to_datetime(t1)
    t2 = pd.to_datetime(t2)
    global_tc_upper_date = pd.to_datetime(global_tc_upper_date)

    window = df[
        (df[date_col] >= t1) &
        (df[date_col] <= t2)
    ].copy()

    if len(window) < 30:
        raise ValueError(f"Window has only {len(window)} observations.")

    window = window.sort_values(date_col).reset_index(drop=True)

    actual_t1 = window[date_col].iloc[0]
    actual_t2 = window[date_col].iloc[-1]

    window["t"] = (window[date_col] - actual_t1).dt.days.astype(float)

    t = window["t"].to_numpy(dtype=float)
    y = window[y_col].to_numpy(dtype=float)

    t_last = float(t.max())

    tc_lower = t_last + tc_min_days
    tc_upper = float((global_tc_upper_date - actual_t1).days)

    if tc_upper <= tc_lower:
        raise ValueError("Invalid tc bounds.")

    def objective(theta):
        tc, m, omega = theta
        _, sse, _ = solve_linear_params(t, y, tc, m, omega)
        return sse

    bounds = [
        (tc_lower, tc_upper),
        M_BOUNDS,
        OMEGA_BOUNDS,
    ]

    result_de = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        polish=False,
        workers=1,
        updating="immediate",
    )

    result = minimize(
        objective,
        result_de.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter},
    )

    tc, m, omega = result.x

    linear_params, sse, rmse = solve_linear_params(t, y, tc, m, omega)

    if linear_params is None:
        raise RuntimeError("LPPLS fit failed.")

    A, B, C1, C2 = linear_params

    C = float(np.sqrt(C1 ** 2 + C2 ** 2))
    phi = float(np.arctan2(C2, C1))
    tc_date = actual_t1 + pd.Timedelta(days=float(tc))

    damping = (m * abs(B)) / (omega * abs(C)) if abs(C) > 0 else np.inf

    positive_lppls_valid = (
        (B < 0) and
        (abs(C) < 1) and
        (M_BOUNDS[0] <= m <= M_BOUNDS[1]) and
        (OMEGA_BOUNDS[0] <= omega <= OMEGA_BOUNDS[1]) and
        (damping >= 1) and
        ((tc_date - actual_t2).days > 0)
    )

    return pd.DataFrame([{
        "t1": actual_t1.strftime("%Y-%m-%d"),
        "t2": actual_t2.strftime("%Y-%m-%d"),
        "observations": int(len(window)),

        "tc_predicted": tc_date.strftime("%Y-%m-%d"),
        "tc_days_after_t2": int((tc_date - actual_t2).days),
        "global_tc_upper_date": global_tc_upper_date.strftime("%Y-%m-%d"),

        "positive_lppls_valid": bool(positive_lppls_valid),

        "RMSE": float(rmse),
        "SSE": float(sse),

        "A": float(A),
        "B": float(B),
        "C1": float(C1),
        "C2": float(C2),
        "C": float(C),
        "m": float(m),
        "omega": float(omega),
        "phi": float(phi),
        "damping": float(damping),

        "optimizer_success": bool(result.success),
    }])


# ============================================================
# DAILY ROLLING RUNNER
# ============================================================

def run_daily_positive_lppls(
    df,
    output_csv=OUTPUT_CSV,
    date_col=DATE_COL,
    y_col=Y_COL,
    t2_step=T2_STEP,
    dt_min=DT_MIN,
    dt_max=DT_MAX,
    dt_step=DT_STEP,
    tc_min_days=TC_MIN_DAYS,
    tc_max_days=TC_MAX_DAYS,
    seed=SEED,
    require_full_scale=REQUIRE_FULL_SCALE,
):
    windows = generate_rolling_lppl_windows(
        df=df,
        date_col=date_col,
        t2_step=t2_step,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_step=dt_step,
        require_full_scale=require_full_scale,
    )

    if windows.empty:
        raise ValueError("No valid rolling windows generated.")

    if os.path.exists(output_csv):
        os.remove(output_csv)

    for i, row in windows.iterrows():
        try:
            t2 = pd.to_datetime(row["t2"])
            global_tc_upper_date = t2 + pd.Timedelta(days=tc_max_days)

            fit_row = fit_lppl_row(
                df=df,
                t1=row["t1"],
                t2=row["t2"],
                global_tc_upper_date=global_tc_upper_date,
                date_col=date_col,
                y_col=y_col,
                tc_min_days=tc_min_days,
                seed=seed,
            )

            fit_row["window_dt_requested"] = row["window_dt_requested"]

            write_header = not os.path.exists(output_csv)

            fit_row.to_csv(
                output_csv,
                mode="a",
                header=write_header,
                index=False,
            )

            print(
                f"{i + 1}/{len(windows)} done | "
                f"t2={row['t2']} | "
                f"dt={row['window_dt_requested']} | "
                f"valid={fit_row['positive_lppls_valid'].iloc[0]} | "
                f"tc={fit_row['tc_predicted'].iloc[0]}"
            )

        except Exception as e:
            error_row = pd.DataFrame([{
                "t1": row["t1"],
                "t2": row["t2"],
                "window_dt_requested": row["window_dt_requested"],
                "error": str(e),
            }])

            write_header = not os.path.exists(output_csv)

            error_row.to_csv(
                output_csv,
                mode="a",
                header=write_header,
                index=False,
            )

            print(
                f"{i + 1}/{len(windows)} failed | "
                f"t2={row['t2']} | "
                f"dt={row['window_dt_requested']} | "
                f"{e}"
            )

    return pd.read_csv(output_csv)


# ============================================================
# POSITIVE CONFIDENCE INDICATOR
# ============================================================

def make_daily_positive_confidence(results):
    res = results.copy()

    res["t2"] = pd.to_datetime(res["t2"], errors="coerce")
    res["tc_predicted"] = pd.to_datetime(res["tc_predicted"], errors="coerce")

    res["positive_lppls_valid"] = (
        res["positive_lppls_valid"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    res = res.dropna(subset=["t2"])

    confidence = (
        res.groupby("t2")
        .agg(
            total_fits=("positive_lppls_valid", "size"),
            valid_fits=("positive_lppls_valid", "sum"),
            mean_tc=("tc_predicted", "mean"),
            median_tc=("tc_predicted", "median"),
            mean_rmse=("RMSE", "mean"),
        )
        .reset_index()
    )

    confidence["positive_bubble_confidence"] = (
        confidence["valid_fits"] / confidence["total_fits"]
    )

    return confidence


# ============================================================
# PLOTTING
# ============================================================

def plot_price_and_daily_positive_confidence(
    prices,
    confidence,
    title="STOXX Europe 600: Daily Positive LPPLS Bubble Confidence",
):
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(
        prices[DATE_COL],
        prices["Close"],
        color="black",
        linewidth=1.4,
        label="STOXX Europe 600 Price",
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()

    ax2.plot(
        confidence["t2"],
        confidence["positive_bubble_confidence"],
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="Positive LPPLS Confidence Score",
    )

    ax2.set_ylabel("Confidence Score")
    ax2.set_ylim(0, 1)

    ax1.set_title(title)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig("eurostoxx600_daily_positive_lppls_confidence_plot.png", dpi=300)
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    prices = load_prices(PRICE_CSV)

    expected_windows = generate_rolling_lppl_windows(prices)
    print("Rows in price file:", len(prices))
    print("Expected LPPLS fits:", len(expected_windows))
    print("Output raw fits file:", OUTPUT_CSV)
    print("Output confidence file:", CONFIDENCE_CSV)

    results = run_daily_positive_lppls(
        df=prices,
        output_csv=OUTPUT_CSV,
        t2_step=T2_STEP,
        dt_min=DT_MIN,
        dt_max=DT_MAX,
        dt_step=DT_STEP,
        tc_min_days=TC_MIN_DAYS,
        tc_max_days=TC_MAX_DAYS,
        seed=SEED,
        require_full_scale=REQUIRE_FULL_SCALE,
    )

    confidence = make_daily_positive_confidence(results)
    confidence.to_csv(CONFIDENCE_CSV, index=False)

    plot_price_and_daily_positive_confidence(prices, confidence)

    print("Saved raw daily positive LPPLS fits to:", OUTPUT_CSV)
    print("Saved daily positive confidence scores to:", CONFIDENCE_CSV)
    print(confidence.head())
    print(confidence.tail())