import pandas as pd
import numpy as np


def calculate_lppl_significance(
    lppl_csv,
    output_csv,
    model_name="LPPL",
    event_name=None,
    date_col="tc_predicted",
    true_col="tc_true",
    tc_true=None,
    window_days=90,
    n_simulations=10000,
    seed=42,
):
    # =========================
    # LOAD LPPL DATA
    # =========================

    df = pd.read_csv(lppl_csv)

    if date_col not in df.columns:
        raise ValueError(f"Missing required column: {date_col}")

    if tc_true is None:
        if true_col not in df.columns:
            raise ValueError("Either pass tc_true or include tc_true column in CSV.")
        tc_true_dt = pd.to_datetime(df[true_col].dropna().iloc[0])
    else:
        tc_true_dt = pd.to_datetime(tc_true)

    preds = pd.to_datetime(df[date_col], errors="coerce").dropna()

    if preds.empty:
        raise ValueError("No valid tc predictions found.")

    errors = (preds - tc_true_dt).dt.days.to_numpy()
    n = len(errors)

    # =========================
    # OBSERVED LPPL STATS
    # =========================

    lppl_h90 = np.mean(np.abs(errors) <= window_days) * 100
    lppl_mae = np.mean(np.abs(errors))
    lppl_mean_error = np.mean(errors)
    lppl_median_error = np.median(errors)

    # =========================
    # ORIGINAL PERMUTATION METHOD
    # =========================
    # This matches your original script:
    # Keep LPPL predictions fixed.
    # Randomly choose fake tc_true dates between min/max predicted tc.
    # Recalculate hit rate and MAE against each fake tc_true.

    rng = np.random.default_rng(seed)

    min_date = preds.min()
    max_date = preds.max()
    span_days = (max_date - min_date).days

    random_h90_values = np.empty(n_simulations)
    random_mae_values = np.empty(n_simulations)

    for i in range(n_simulations):
        fake_tc = min_date + pd.to_timedelta(
            rng.integers(0, span_days + 1),
            unit="D"
        )

        fake_errors = (preds - fake_tc).dt.days.to_numpy()

        random_h90_values[i] = np.mean(np.abs(fake_errors) <= window_days) * 100
        random_mae_values[i] = np.mean(np.abs(fake_errors))

    random_h90_mean = random_h90_values.mean()
    random_mae_mean = random_mae_values.mean()

    p_h90 = np.mean(random_h90_values >= lppl_h90)
    p_mae = np.mean(random_mae_values <= lppl_mae)

    # =========================
    # BOOTSTRAP BIAS CIs
    # =========================
    # Uses LPPL errors only.

    bootstrap_means = np.empty(n_simulations)
    bootstrap_medians = np.empty(n_simulations)

    for i in range(n_simulations):
        sample = rng.choice(errors, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)
        bootstrap_medians[i] = np.median(sample)

    mean_ci_low, mean_ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    median_ci_low, median_ci_high = np.percentile(bootstrap_medians, [2.5, 97.5])

    # =========================
    # OUTPUT TABLE ROW
    # =========================

    result = pd.DataFrame([{
        "Model": model_name,
        "Event": event_name,
        "N": n,
        "tc_true": tc_true_dt.strftime("%Y-%m-%d"),

        f"H{window_days}_pct": lppl_h90,
        f"Rand_H{window_days}_pct": random_h90_mean,
        f"p_H{window_days}": p_h90,

        "MAE_days": lppl_mae,
        "Rand_MAE_days": random_mae_mean,
        "p_MAE": p_mae,

        "Mean_Err_days": lppl_mean_error,
        "Mean_95_CI": f"[{mean_ci_low:.2f}, {mean_ci_high:.2f}]",

        "Md_Err_days": lppl_median_error,
        "Md_95_CI": f"[{median_ci_low:.2f}, {median_ci_high:.2f}]",
    }])

    result.to_csv(output_csv, index=False)

    print(f"Saved significance results to: {output_csv}")
    return result


# =========================
# EXAMPLE USAGE
# =========================

sig = calculate_lppl_significance(
    lppl_csv="baseline_covid.csv",
    output_csv="baseline_covid_significance.csv",
    model_name="LPPL",
    event_name="COVID",
    tc_true="2022-01-05",
    window_days=90,
    n_simulations=20000,
    seed=42
)

print(sig.to_string(index=False))