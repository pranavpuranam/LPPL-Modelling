import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
CONF_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"
PRICE_PATH = "eurostoxx600_prices.csv"

CONF_COL = "positive_bubble_confidence"
DATE_COL_CONF = "t2"
DATE_COL_PRICE = "Date"

EVENT_COL = "tc_literature"   # change to tc_gsadf or tc_drawdown if needed

WINDOWS = [3, 4, 5, 15, 30, 45, 50, 55, 60]
N_PERMUTATIONS = 10000
RANDOM_SEED = 42
EXCLUSION_WINDOW = 90

OUTPUT_PATH = "ci_pre_collapse_permutation_test.csv"


# =========================
# LOAD DATA
# =========================
conf = pd.read_csv(CONF_PATH)
prices = pd.read_csv(PRICE_PATH)

conf[DATE_COL_CONF] = pd.to_datetime(conf[DATE_COL_CONF])
prices[DATE_COL_PRICE] = pd.to_datetime(prices[DATE_COL_PRICE])

conf = conf.rename(columns={DATE_COL_CONF: "Date"})
prices = prices.rename(columns={DATE_COL_PRICE: "Date"})

df = prices.merge(
    conf[["Date", CONF_COL]],
    on="Date",
    how="left"
).sort_values("Date").reset_index(drop=True)

df[CONF_COL] = df[CONF_COL].fillna(0)

if EVENT_COL not in df.columns:
    raise ValueError(f"Could not find event column: {EVENT_COL}")

event_idx = df.index[df[EVENT_COL] == 1].to_numpy()

if len(event_idx) == 0:
    raise ValueError(f"No event dates found where {EVENT_COL} == 1")


# =========================
# HELPERS
# =========================
def mean_ci_before(index, window):
    start = index - window
    end = index

    if start < 0:
        return np.nan

    return df.loc[start:end - 1, CONF_COL].mean()


def valid_pseudo_event_indices(window):
    candidates = np.arange(window, len(df))
    mask = np.ones(len(candidates), dtype=bool)

    for tc in event_idx:
        mask &= np.abs(candidates - tc) > EXCLUSION_WINDOW

    return candidates[mask]


# =========================
# PERMUTATION TEST
# =========================
rng = np.random.default_rng(RANDOM_SEED)
results = []

for window in WINDOWS:
    usable_events = [idx for idx in event_idx if idx - window >= 0]

    real_values = np.array([
        mean_ci_before(idx, window)
        for idx in usable_events
    ])

    real_mean = np.nanmean(real_values)

    pseudo_candidates = valid_pseudo_event_indices(window)

    if len(pseudo_candidates) < len(usable_events):
        raise ValueError(
            f"Not enough pseudo-event candidates for window={window}. "
            f"Reduce EXCLUSION_WINDOW."
        )

    random_means = []

    for _ in range(N_PERMUTATIONS):
        sampled_idx = rng.choice(
            pseudo_candidates,
            size=len(usable_events),
            replace=False
        )

        sampled_values = np.array([
            mean_ci_before(idx, window)
            for idx in sampled_idx
        ])

        random_means.append(np.nanmean(sampled_values))

    random_means = np.array(random_means)

    random_mean = random_means.mean()
    diff = real_mean - random_mean

    p_value = (1 + np.sum(random_means >= real_mean)) / (1 + N_PERMUTATIONS)

    results.append({
        "window_days_before_tc": window,
        "n_real_events": len(usable_events),
        "real_mean_ci": real_mean,
        "random_mean_ci": random_mean,
        "difference": diff,
        "empirical_p_value": p_value,
        "random_ci_5pct": np.percentile(random_means, 5),
        "random_ci_95pct": np.percentile(random_means, 95),
    })


# =========================
# SAVE + PRINT
# =========================
results_df = pd.DataFrame(results)

print("\nPermutation test: Is CI elevated before true collapse dates?\n")
print(results_df.to_string(index=False))

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved results to: {OUTPUT_PATH}")