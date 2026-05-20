import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde


def calculate_benchmark_tc_metrics(
    csv_path,
    output_csv="benchmark_metrics.csv",
    model_name="Random Benchmark",
    event_name=None,
    guess_col="tc_guesses",
    true_col="tc_true",
    hit_windows=(30, 60, 90),
):
    df = pd.read_csv(csv_path)

    if guess_col not in df.columns:
        raise ValueError(f"CSV must contain column: {guess_col}")

    if true_col not in df.columns:
        raise ValueError(f"CSV must contain column: {true_col}")

    df[guess_col] = pd.to_datetime(df[guess_col], errors="coerce")
    df[true_col] = pd.to_datetime(df[true_col], errors="coerce")

    df = df.dropna(subset=[guess_col, true_col]).copy()

    if df.empty:
        raise ValueError("No valid benchmark guesses found.")

    errors = (df[guess_col] - df[true_col]).dt.days.to_numpy()
    n = len(errors)

    metrics = {
        "model": model_name,
        "event": event_name,
        "input_csv": csv_path,
        "tc_true": df[true_col].iloc[0].strftime("%Y-%m-%d"),
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

    if "benchmark_type" in df.columns:
        metrics["benchmark_type"] = df["benchmark_type"].iloc[0]

    if "lower_bound" in df.columns:
        metrics["lower_bound"] = df["lower_bound"].iloc[0]

    if "upper_bound" in df.columns:
        metrics["upper_bound"] = df["upper_bound"].iloc[0]

    if n >= 2 and np.std(errors) > 0:
        kde = gaussian_kde(errors)
        grid = np.linspace(errors.min(), errors.max(), 1000)
        density = kde(grid)

        kde_peak_error = float(grid[np.argmax(density)])
        kde_peak_date = df[true_col].iloc[0] + pd.to_timedelta(kde_peak_error, unit="D")

        metrics["kde_peak_error_days"] = kde_peak_error
        metrics["kde_peak_tc_date"] = kde_peak_date.strftime("%Y-%m-%d")
    else:
        metrics["kde_peak_error_days"] = np.nan
        metrics["kde_peak_tc_date"] = None

    result = pd.DataFrame([metrics])
    result.to_csv(output_csv, index=False)

    print(f"Saved benchmark metrics to: {output_csv}")
    return result


# Example usage
benchmark_metrics = calculate_benchmark_tc_metrics(
    csv_path="benchmark_qe.csv",
    output_csv="benchmark_qe_metrics.csv",
    model_name="Random Benchmark",
    event_name="QE"
)

print(benchmark_metrics.to_string(index=False))