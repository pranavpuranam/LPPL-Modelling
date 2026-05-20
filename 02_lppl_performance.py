import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


def calculate_core_tc_metrics(
    csv_path,
    tc_true,
    output_csv="core_tc_metrics.csv",
    model_name=None,
    event_name=None,
    date_col="tc_predicted",
    rmse_col="RMSE",
    check_col="check_pass",
    optimizer_col="optimizer_success",
    hit_windows=(30, 60, 90),
):
    df = pd.read_csv(csv_path)

    if date_col not in df.columns:
        raise ValueError(f"CSV must contain column: {date_col}")

    tc_true_dt = pd.to_datetime(tc_true)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    if df.empty:
        raise ValueError("No valid tc predictions found.")

    errors = (df[date_col] - tc_true_dt).dt.days.to_numpy()
    n = len(errors)

    metrics = {
        "model": model_name,
        "event": event_name,
        "input_csv": csv_path,
        "tc_true": tc_true,
        "n_fits": n,

        "mean_tc_error_days": float(np.mean(errors)),
        "median_tc_error_days": float(np.median(errors)),
        "mae_tc_error_days": float(np.mean(np.abs(errors))),
        "std_tc_error_days": float(np.std(errors, ddof=1)) if n > 1 else np.nan,
    }

    for w in hit_windows:
        metrics[f"hit_rate_within_{w}_days_pct"] = float(
            np.mean(np.abs(errors) <= w) * 100
        )

    if check_col in df.columns:
        metrics["check_pass_pct"] = float(df[check_col].astype(bool).mean() * 100)

    if optimizer_col in df.columns:
        metrics["optimizer_success_pct"] = float(
            df[optimizer_col].astype(bool).mean() * 100
        )

    if check_col in df.columns and optimizer_col in df.columns:
        valid = df[check_col].astype(bool) & df[optimizer_col].astype(bool)
        metrics["valid_and_optimizer_success_pct"] = float(valid.mean() * 100)

    if rmse_col in df.columns:
        rmse = pd.to_numeric(df[rmse_col], errors="coerce").dropna()

        if len(rmse) > 0:
            metrics["rmse_median"] = float(rmse.median())
            metrics["rmse_mean"] = float(rmse.mean())

    if n >= 2 and np.std(errors) > 0:
        kde = gaussian_kde(errors)
        grid = np.linspace(errors.min(), errors.max(), 1000)
        density = kde(grid)

        kde_peak_error = float(grid[np.argmax(density)])
        kde_peak_date = tc_true_dt + pd.to_timedelta(kde_peak_error, unit="D")

        metrics["kde_peak_error_days"] = kde_peak_error
        metrics["kde_peak_tc_date"] = kde_peak_date.strftime("%Y-%m-%d")
    else:
        metrics["kde_peak_error_days"] = np.nan
        metrics["kde_peak_tc_date"] = None

    result = pd.DataFrame([metrics])
    result.to_csv(output_csv, index=False)

    print(f"Saved core metrics to: {output_csv}")
    return result


# Example usage
metrics = calculate_core_tc_metrics(
    csv_path="baseline_qe.csv",
    tc_true="2015-04-15",
    output_csv="baseline_qe_metrics.csv",
    model_name="LPPL",
    event_name="QE"
)

print(metrics.to_string(index=False))