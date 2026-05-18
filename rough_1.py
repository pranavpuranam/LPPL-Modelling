# ============================================================
# Imports
# ============================================================
import pandas as pd
import numpy as np

from scipy.optimize import differential_evolution, minimize


# ============================================================
# Load EuroStoxx 600 price data
# Expected columns: Date, Close, log_price
# ============================================================
CSV_FILE = "eurostoxx600_prices.csv"

prices = pd.read_csv(CSV_FILE)

prices["Date"] = pd.to_datetime(prices["Date"])
prices["Close"] = pd.to_numeric(prices["Close"], errors="coerce")
prices["log_price"] = pd.to_numeric(prices["log_price"], errors="coerce")

prices = (
    prices
    .dropna(subset=["Date", "Close", "log_price"])
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# Hard-coded admissible LPPL nonlinear bounds
# ============================================================
M_BOUNDS = (0.1, 0.9)
OMEGA_BOUNDS = (6.0, 13.0)


# ============================================================
# Linear LPPL solver
# ============================================================
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


# ============================================================
# Main LPPL fitting function
# ============================================================
def fit_lppl_window(
    df,
    t1,
    t2,
    date_col="Date",
    y_col="log_price",
    tc_min_days=1,
    tc_max_days=365,
    seed=42,
    maxiter=5000,
    verbose=True,
):
    """
    Fit LPPL over date window [t1, t2].

    Hard-coded optimizer bounds:
        0.1 <= m <= 0.9
        6 <= omega <= 13

    tc search range:
        t2 + tc_min_days <= tc <= t2 + tc_max_days

    Post-fit filters:
        B < 0
        abs(C) < 1
    """

    t1 = pd.to_datetime(t1)
    t2 = pd.to_datetime(t2)

    window = df[
        (df[date_col] >= t1) &
        (df[date_col] <= t2)
    ].copy()

    if len(window) < 30:
        raise ValueError(
            f"Window has only {len(window)} observations. Need at least 30."
        )

    window = window.sort_values(date_col).reset_index(drop=True)

    actual_t1 = window[date_col].iloc[0]
    actual_t2 = window[date_col].iloc[-1]

    window["t"] = (window[date_col] - actual_t1).dt.days.astype(float)

    t = window["t"].to_numpy(dtype=float)
    y = window[y_col].to_numpy(dtype=float)

    t_last = float(t.max())

    tc_lower = t_last + tc_min_days
    tc_upper = t_last + tc_max_days

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

    filter_results = {
        "hardcoded 0.1 <= m <= 0.9": bool(0.1 <= m <= 0.9),
        "hardcoded 6 <= omega <= 13": bool(6 <= omega <= 13),
        "post-fit B < 0": bool(B < 0),
        "post-fit abs(C) < 1": bool(abs(C) < 1),
    }

    passes_filters = all(filter_results.values())

    output = {
        "fit_start": actual_t1.strftime("%Y-%m-%d"),
        "fit_end": actual_t2.strftime("%Y-%m-%d"),
        "n_obs": int(len(window)),
        "tc_search_start": (
            actual_t2 + pd.Timedelta(days=tc_min_days)
        ).strftime("%Y-%m-%d"),
        "tc_search_end": (
            actual_t2 + pd.Timedelta(days=tc_max_days)
        ).strftime("%Y-%m-%d"),
        "tc_date": tc_date.strftime("%Y-%m-%d"),
        "tc_days_after_t2": int((tc_date - actual_t2).days),
        "A": float(A),
        "B": float(B),
        "C1": float(C1),
        "C2": float(C2),
        "C": float(C),
        "m": float(m),
        "omega": float(omega),
        "phi": float(phi),
        "sse": float(sse),
        "rmse": float(rmse),
        "optimizer_success": bool(result.success),
        "passes_filters": bool(passes_filters),
        "filter_results": filter_results,
    }

    if verbose:
        print("\nLPPL fit result")
        print("----------------")
        print(f"Fit window:       {output['fit_start']} to {output['fit_end']}")
        print(f"Observations:     {output['n_obs']}")
        print(f"tc search range:  {output['tc_search_start']} to {output['tc_search_end']}")
        print(f"Predicted tc:     {output['tc_date']}")
        print(f"Days after t2:    {output['tc_days_after_t2']}")
        print(f"SSE:              {output['sse']:.6f}")
        print(f"RMSE:             {output['rmse']:.6f}")
        print(f"Passes filters:   {output['passes_filters']}")

        print("\nParameters:")
        for key in ["A", "B", "C1", "C2", "C", "m", "omega", "phi"]:
            print(f"{key:8s} = {output[key]:.6f}")

        print("\nFilter checks:")
        for key, value in output["filter_results"].items():
            print(f"{key:30s}: {value}")

    return output


# ============================================================
# Example run
# ============================================================
fit = fit_lppl_window(
    prices,
    t1="2001-04-01",
    t2="2007-01-01",
    tc_min_days=1,
    tc_max_days=365,
)