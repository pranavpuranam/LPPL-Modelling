import pandas as pd


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
    """
    Generate LPPL fitting windows.

    For each t2 before tc_true:
        t1 = t2 - dt

    All steps are in trading observations/rows, not calendar days.
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    tc_true = pd.to_datetime(tc_true)

    tc_idx = df[df[date_col] <= tc_true].index.max()

    if pd.isna(tc_idx):
        raise ValueError("tc_true is before the start of the dataset.")

    results = []

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

            results.append({
                "t1": t1_date.strftime("%Y-%m-%d"),
                "t2": t2_date.strftime("%Y-%m-%d"),
                "dt": dt,
            })

    return pd.DataFrame(results)


# ============================================================
# Example use
# ============================================================
CSV_FILE = "eurostoxx600_prices.csv"

prices = pd.read_csv(CSV_FILE)
prices["Date"] = pd.to_datetime(prices["Date"])
prices = prices.sort_values("Date").reset_index(drop=True)

windows_df = generate_lppl_windows(
    prices,
    tc_true="2007-06-01",
    t2_step=10,
    dt_step=25,
)

print(windows_df.to_string(index=False))