# 03a: test ci before collapse dates

import numpy as np  # numerical operations
import pandas as pd  # handle dataframes

CONF_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"  # confidence input
PRICE_PATH = "eurostoxx600_prices.csv"  # price input
CONF_COL = "positive_bubble_confidence"  # confidence column
DATE_COL_CONF = "t2"  # confidence date column
DATE_COL_PRICE = "Date"  # price date column
EVENT_COL = "tc_literature"  # event label column
WINDOWS = [3, 4, 5, 15, 30, 45, 50, 55, 60]  # lookback windows
N_PERMUTATIONS = 10000  # permutation count
RANDOM_SEED = 42  # random seed
EXCLUSION_WINDOW = 90  # event exclusion window
OUTPUT_PATH = "ci_pre_collapse_permutation_test.csv"  # output file

conf = pd.read_csv(CONF_PATH)  # load confidence data
prices = pd.read_csv(PRICE_PATH)  # load price data
conf[DATE_COL_CONF] = pd.to_datetime(conf[DATE_COL_CONF])  # parse confidence dates
prices[DATE_COL_PRICE] = pd.to_datetime(prices[DATE_COL_PRICE])  # parse price dates
conf = conf.rename(columns={DATE_COL_CONF: "Date"})  # standardise confidence date name
prices = prices.rename(columns={DATE_COL_PRICE: "Date"})  # standardise price date name

df = prices.merge(
    conf[["Date", CONF_COL]],
    on="Date",
    how="left"
).sort_values("Date").reset_index(drop=True)  # merge price and confidence data

df[CONF_COL] = df[CONF_COL].fillna(0)  # fill missing confidence with zero

if EVENT_COL not in df.columns:  # check event labels exist
    raise ValueError(f"Could not find event column: {EVENT_COL}")  # stop if missing

event_idx = df.index[df[EVENT_COL] == 1].to_numpy()  # get event indices

if len(event_idx) == 0:  # check events exist
    raise ValueError(f"No event dates found where {EVENT_COL} == 1")  # stop if none

def mean_ci_before(index, window):
    start = index - window  # set window start
    end = index  # set window end
    if start < 0:  # check enough history
        return np.nan  # return missing if not enough data
    return df.loc[start:end - 1, CONF_COL].mean()  # calculate pre-event mean ci

def valid_pseudo_event_indices(window):
    candidates = np.arange(window, len(df))  # create candidate indices
    mask = np.ones(len(candidates), dtype=bool)  # initialise valid mask
    for tc in event_idx:  # loop over true events
        mask &= np.abs(candidates - tc) > EXCLUSION_WINDOW  # exclude nearby dates
    return candidates[mask]  # return valid pseudo-events

rng = np.random.default_rng(RANDOM_SEED)  # initialise random generator
results = []  # store results

for window in WINDOWS:  # loop over lookback windows
    usable_events = [idx for idx in event_idx if idx - window >= 0]  # keep events with history
    real_values = np.array([
        mean_ci_before(idx, window)
        for idx in usable_events
    ])  # calculate real pre-event ci values
    real_mean = np.nanmean(real_values)  # calculate real mean ci
    pseudo_candidates = valid_pseudo_event_indices(window)  # get pseudo-event candidates
    if len(pseudo_candidates) < len(usable_events):  # check enough candidates
        raise ValueError(
            f"Not enough pseudo-event candidates for window={window}. "
            f"Reduce EXCLUSION_WINDOW."
        )  # stop if too few candidates
    random_means = []  # store random means
    for _ in range(N_PERMUTATIONS):  # loop over permutations
        sampled_idx = rng.choice(
            pseudo_candidates,
            size=len(usable_events),
            replace=False
        )  # sample pseudo-events
        sampled_values = np.array([
            mean_ci_before(idx, window)
            for idx in sampled_idx
        ])  # calculate pseudo-event ci values
        random_means.append(np.nanmean(sampled_values))  # store pseudo mean
    random_means = np.array(random_means)  # convert to array
    random_mean = random_means.mean()  # calculate random mean ci
    diff = real_mean - random_mean  # calculate difference
    p_value = (1 + np.sum(random_means >= real_mean)) / (1 + N_PERMUTATIONS)  # calculate empirical p-value
    results.append({
        "window_days_before_tc": window,
        "n_real_events": len(usable_events),
        "real_mean_ci": real_mean,
        "random_mean_ci": random_mean,
        "difference": diff,
        "empirical_p_value": p_value,
        "random_ci_5pct": np.percentile(random_means, 5),
        "random_ci_95pct": np.percentile(random_means, 95),
    })  # store window result

results_df = pd.DataFrame(results)  # create results table
print("\nPermutation test: Is CI elevated before true collapse dates?\n")  # print title
print(results_df.to_string(index=False))  # print results
results_df.to_csv(OUTPUT_PATH, index=False)  # save results
print(f"\nSaved results to: {OUTPUT_PATH}")  # print save path