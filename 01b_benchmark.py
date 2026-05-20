import pandas as pd
import numpy as np


def generate_random_tc_benchmark(
    tc_true,
    n_samples,
    output_csv,
    t2_start_before_tc=180,
    t2_end_before_tc=5,
    tc_min_days=1,
    tc_max_days=365,
    seed=42,
):
    """
    Generate random benchmark tc guesses using the same broad timing logic
    as the LPPL fitting setup.

    Random tc guesses are sampled between:
        earliest possible forecast date = tc_true - t2_start_before_tc + tc_min_days
        latest possible forecast date   = tc_true - t2_end_before_tc + tc_max_days

    Parameters
    ----------
    tc_true : str
        True event date, e.g. "2015-04-15".
    n_samples : int
        Number of random benchmark guesses to generate.
    output_csv : str
        Output CSV filename.
    """

    rng = np.random.default_rng(seed)

    tc_true_dt = pd.to_datetime(tc_true)

    lower_date = tc_true_dt - pd.Timedelta(days=t2_start_before_tc) + pd.Timedelta(days=tc_min_days)
    upper_date = tc_true_dt - pd.Timedelta(days=t2_end_before_tc) + pd.Timedelta(days=tc_max_days)

    span_days = (upper_date - lower_date).days

    random_offsets = rng.integers(
        low=0,
        high=span_days + 1,
        size=n_samples
    )

    tc_guesses = lower_date + pd.to_timedelta(random_offsets, unit="D")

    benchmark = pd.DataFrame({
        "tc_true": tc_true_dt.strftime("%Y-%m-%d"),
        "tc_guesses": tc_guesses.strftime("%Y-%m-%d"),
        "benchmark_type": "random_uniform",
        "lower_bound": lower_date.strftime("%Y-%m-%d"),
        "upper_bound": upper_date.strftime("%Y-%m-%d"),
    })

    benchmark.to_csv(output_csv, index=False)

    print(f"Saved benchmark CSV to: {output_csv}")
    print(f"N samples: {n_samples}")
    print(f"Random tc range: {lower_date.date()} to {upper_date.date()}")

    return benchmark


# Example for QE
benchmark_qe = generate_random_tc_benchmark(
    tc_true="2022-01-05",
    n_samples=2268,
    output_csv="benchmark_covid.csv",
    t2_start_before_tc=180,
    t2_end_before_tc=5,
    tc_min_days=1,
    tc_max_days=365,
    seed=42
)

print(benchmark_qe.head())