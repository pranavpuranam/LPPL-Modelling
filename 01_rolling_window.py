import os
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution, minimize


M_BOUNDS = (0.1, 0.9)
OMEGA_BOUNDS = (6.0, 13.0)


def load_prices(csv_file="eurostoxx600_prices.csv"):
    df = pd.read_csv(csv_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["log_price"] = pd.to_numeric(df["log_price"], errors="coerce")

    return (
        df
        .dropna(subset=["Date", "Close", "log_price"])
        .sort_values("Date")
        .reset_index(drop=True)
    )


def generate_lppl_windows(
    df,
    tc_true,
    date_col="Date",
    t2_start_before_tc=180,
    t2_end_before_tc=5,
    t2_step=10,
    dt_min=125,
    dt_max=750,
    dt_step=25,
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    tc_true = pd.to_datetime(tc_true)
    tc_idx = df[df[date_col] <= tc_true].index.max()

    if pd.isna(tc_idx):
        raise ValueError("tc_true is before the start of the dataset.")

    rows = []

    for t2_idx in range(
        tc_idx - t2_start_before_tc,
        tc_idx - t2_end_before_tc + 1,
        t2_step,
    ):
        if t2_idx < 0:
            continue

        t2_date = df.loc[t2_idx, date_col]

        for dt in range(dt_min, dt_max + 1, dt_step):
            t1_idx = t2_idx - dt

            if t1_idx < 0:
                continue

            t1_date = df.loc[t1_idx, date_col]

            rows.append({
                "t1": t1_date.strftime("%Y-%m-%d"),
                "t2": t2_date.strftime("%Y-%m-%d"),
                "dt": dt,
            })

    return pd.DataFrame(rows)


def solve_linear_params(t, y, tc, m, omega):
    dt = tc - t

    if np.any(dt <= 0):
        return None, np.inf, None

    f = dt ** m
    g = f * np.cos(omega * np.log(dt))
    h = f * np.sin(omega * np.log(dt))

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
    date_col="Date",
    y_col="log_price",
    tc_min_days=1,
    seed=42,
    maxiter=5000,
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
        raise ValueError(
            f"Invalid tc bounds: lower={tc_lower}, upper={tc_upper}. "
            "Global upper date is not after this window's t2."
        )

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
        raise RuntimeError("LPPL fit failed.")

    A, B, C1, C2 = linear_params

    C = float(np.sqrt(C1**2 + C2**2))
    phi = float(np.arctan2(C2, C1))
    tc_date = actual_t1 + pd.Timedelta(days=float(tc))

    check_pass = (B < 0) and (abs(C) < 1)

    return pd.DataFrame([{
        "t1": actual_t1.strftime("%Y-%m-%d"),
        "t2": actual_t2.strftime("%Y-%m-%d"),
        "observations": int(len(window)),
        "tc_predicted": tc_date.strftime("%Y-%m-%d"),
        "tc_days_after_t2": int((tc_date - actual_t2).days),
        "global_tc_upper_date": global_tc_upper_date.strftime("%Y-%m-%d"),
        "check_pass": bool(check_pass),
        "RMSE": float(rmse),
        "A": float(A),
        "B": float(B),
        "C1": float(C1),
        "C2": float(C2),
        "C": float(C),
        "m": float(m),
        "omega": float(omega),
        "phi": float(phi),
        "optimizer_success": bool(result.success),
    }])


def run_lppl_event(
    df,
    tc_true,
    output_csv,
    date_col="Date",
    y_col="log_price",
    t2_start_before_tc=180,
    t2_end_before_tc=5,
    t2_step=10,
    dt_min=125,
    dt_max=750,
    dt_step=25,
    tc_min_days=1,
    tc_max_days=365,
    seed=42,
):
    windows = generate_lppl_windows(
        df=df,
        tc_true=tc_true,
        date_col=date_col,
        t2_start_before_tc=t2_start_before_tc,
        t2_end_before_tc=t2_end_before_tc,
        t2_step=t2_step,
        dt_min=dt_min,
        dt_max=dt_max,
        dt_step=dt_step,
    )

    if windows.empty:
        raise ValueError("No valid windows generated.")

    latest_t2 = pd.to_datetime(windows["t2"]).max()
    global_tc_upper_date = latest_t2 + pd.Timedelta(days=tc_max_days)

    if os.path.exists(output_csv):
        os.remove(output_csv)

    for i, row in windows.iterrows():
        try:
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

            fit_row.insert(0, "tc_true", pd.to_datetime(tc_true).strftime("%Y-%m-%d"))
            fit_row.insert(3, "dt", row["dt"])

            write_header = not os.path.exists(output_csv)

            fit_row.to_csv(
                output_csv,
                mode="a",
                header=write_header,
                index=False,
            )

            print(
                f"{i + 1}/{len(windows)} done | "
                f"t1={row['t1']} t2={row['t2']} dt={row['dt']} | "
                f"pass={fit_row['check_pass'].iloc[0]} | "
                f"tc={fit_row['tc_predicted'].iloc[0]}"
            )

        except Exception as e:
            error_row = pd.DataFrame([{
                "tc_true": pd.to_datetime(tc_true).strftime("%Y-%m-%d"),
                "t1": row["t1"],
                "t2": row["t2"],
                "dt": row["dt"],
                "global_tc_upper_date": global_tc_upper_date.strftime("%Y-%m-%d"),
                "error": str(e),
            }])

            write_header = not os.path.exists(output_csv)

            error_row.to_csv(
                output_csv,
                mode="a",
                header=write_header,
                index=False,
            )

            print(f"{i + 1}/{len(windows)} failed | t1={row['t1']} t2={row['t2']} | {e}")

    return pd.read_csv(output_csv)


prices = load_prices("eurostoxx600_prices.csv")

results = run_lppl_event(
    df=prices,
    tc_true="2007-06-01",
    output_csv="lppl_results_gfc_2007.csv",
    t2_step=5,
    dt_step=10,
    tc_min_days=1,
    tc_max_days=500,
)

print(results.head())