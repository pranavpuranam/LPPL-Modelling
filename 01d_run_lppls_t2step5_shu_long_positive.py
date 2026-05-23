import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize


# ============================================================
# SETTINGS — SHU-LONG POSITIVE LPPLS, COARSE T2 STEP 5
# ============================================================

PRICE_CSV = "eurostoxx600_prices.csv"

OUTPUT_CSV = "eurostoxx600_lppls_t2step5_shu_long_positive_fits.csv"
CONFIDENCE_CSV = "eurostoxx600_lppls_t2step5_shu_long_positive_confidence.csv"
OUTPUT_PNG = "eurostoxx600_lppls_t2step5_shu_long_positive_price_confidence.png"

DATE_COL = "Date"
Y_COL = "log_price"

M_BOUNDS = (0.1, 0.9)
OMEGA_BOUNDS = (6.0, 13.0)

# Shu-style long-window range, coarsened to keep runtime manageable
DT_MIN = 205
DT_MAX = 650
DT_STEP = 20

# Every 5th trading-day endpoint
T2_STEP = 5

# Shu-style tc search/filter logic
TC_MIN_DAYS = 1
TC_SEARCH_FRACTION = 1 / 3
TC_VALID_FRACTION = 1 / 5

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

def generate_rolling_lppl_windows(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    rows = []

    t2_start_idx = DT_MAX if REQUIRE_FULL_SCALE else DT_MIN

    for t2_idx in range(t2_start_idx, len(df), T2_STEP):
        t2_date = df.loc[t2_idx, DATE_COL]

        for dt in range(DT_MIN, DT_MAX + 1, DT_STEP):
            t1_idx = t2_idx - dt

            if t1_idx < 0:
                continue

            rows.append({
                "t1": df.loc[t1_idx, DATE_COL].strftime("%Y-%m-%d"),
                "t2": t2_date.strftime("%Y-%m-%d"),
                "window_dt_requested": dt,
            })

    return pd.DataFrame(rows)


# ============================================================
# LPPLS FITTING
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


def fit_lppl_row(df, t1, t2, window_dt_requested):
    t1 = pd.to_datetime(t1)
    t2 = pd.to_datetime(t2)

    window = df[
        (df[DATE_COL] >= t1) &
        (df[DATE_COL] <= t2)
    ].copy()

    if len(window) < DT_MIN:
        raise ValueError(f"Window has only {len(window)} observations.")

    window = window.sort_values(DATE_COL).reset_index(drop=True)

    actual_t1 = window[DATE_COL].iloc[0]
    actual_t2 = window[DATE_COL].iloc[-1]

    window["t"] = (window[DATE_COL] - actual_t1).dt.days.astype(float)

    t = window["t"].to_numpy(dtype=float)
    y = window[Y_COL].to_numpy(dtype=float)

    t_last = float(t.max())
    window_calendar_days = max(t_last, 1.0)

    tc_lower = t_last + TC_MIN_DAYS
    tc_upper = t_last + TC_SEARCH_FRACTION * window_calendar_days

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
        seed=SEED,
        polish=False,
        workers=1,
        updating="immediate",
    )

    result = minimize(
        objective,
        result_de.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": MAXITER},
    )

    tc, m, omega = result.x

    linear_params, sse, rmse = solve_linear_params(t, y, tc, m, omega)

    if linear_params is None:
        raise RuntimeError("LPPLS fit failed.")

    A, B, C1, C2 = linear_params

    C = float(np.sqrt(C1 ** 2 + C2 ** 2))
    phi = float(np.arctan2(C2, C1))
    tc_date = actual_t1 + pd.Timedelta(days=float(tc))

    tc_days_after_t2 = int((tc_date - actual_t2).days)

    damping = (m * abs(B)) / (omega * abs(C)) if abs(C) > 0 else np.inf

    tc_search_upper_days_after_t2 = float(tc_upper - t_last)
    tc_valid_upper_days_after_t2 = float(TC_VALID_FRACTION * window_calendar_days)

    positive_lppls_valid = (
        (B < 0) and
        (abs(C) < 1) and
        (M_BOUNDS[0] <= m <= M_BOUNDS[1]) and
        (OMEGA_BOUNDS[0] <= omega <= OMEGA_BOUNDS[1]) and
        (damping >= 1) and
        (tc_days_after_t2 > 0) and
        (tc_days_after_t2 <= tc_valid_upper_days_after_t2)
    )

    return pd.DataFrame([{
        "t1": actual_t1.strftime("%Y-%m-%d"),
        "t2": actual_t2.strftime("%Y-%m-%d"),
        "window_dt_requested": int(window_dt_requested),
        "observations": int(len(window)),
        "window_calendar_days": float(window_calendar_days),

        "tc_predicted": tc_date.strftime("%Y-%m-%d"),
        "tc_days_after_t2": int(tc_days_after_t2),
        "tc_search_upper_days_after_t2": tc_search_upper_days_after_t2,
        "tc_valid_upper_days_after_t2": tc_valid_upper_days_after_t2,

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
# RUNNER
# ============================================================

def run_lppls(df):
    windows = generate_rolling_lppl_windows(df)

    if windows.empty:
        raise ValueError("No valid rolling windows generated.")

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    print("Rows in price file:", len(df))
    print("Expected LPPLS fits:", len(windows))
    print("Windows per t2:", len(range(DT_MIN, DT_MAX + 1, DT_STEP)))
    print("Output raw fits file:", OUTPUT_CSV)
    print("Output confidence file:", CONFIDENCE_CSV)

    for i, row in windows.iterrows():
        try:
            fit_row = fit_lppl_row(
                df=df,
                t1=row["t1"],
                t2=row["t2"],
                window_dt_requested=row["window_dt_requested"],
            )

            write_header = not os.path.exists(OUTPUT_CSV)

            fit_row.to_csv(
                OUTPUT_CSV,
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

            write_header = not os.path.exists(OUTPUT_CSV)

            error_row.to_csv(
                OUTPUT_CSV,
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

    return pd.read_csv(OUTPUT_CSV)


# ============================================================
# CONFIDENCE INDICATOR
# ============================================================

def make_confidence(results):
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
# QUICK PLOT
# ============================================================

def plot_price_and_confidence(prices, confidence):
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
        label="LPPLS Shu-Long Positive Confidence",
    )

    ax2.set_ylabel("Confidence Score")
    ax2.set_ylim(0, 1)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper left",
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    prices = load_prices(PRICE_CSV)

    results = run_lppls(prices)

    confidence = make_confidence(results)
    confidence.to_csv(CONFIDENCE_CSV, index=False)

    plot_price_and_confidence(prices, confidence)

    print("Saved raw fits to:", OUTPUT_CSV)
    print("Saved confidence scores to:", CONFIDENCE_CSV)
    print("Saved plot to:", OUTPUT_PNG)

    print(confidence.head())
    print(confidence.tail())