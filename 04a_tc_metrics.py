import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
RESULTS_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"
PRICE_PATH = "eurostoxx600_prices.csv"

EVENT_COL = "tc_literature"   # change to tc_gsadf / tc_drawdown if needed
DATE_COL = "Date"

LOOKBACK_TRADING_DAYS = 250   # only use fits with t2 in this pre-event window
HIT_WINDOWS = [15, 30, 60]    # calendar-day error bands


# =========================
# LOAD DATA
# =========================
fits = pd.read_csv(RESULTS_PATH)
prices = pd.read_csv(PRICE_PATH)

prices[DATE_COL] = pd.to_datetime(prices[DATE_COL])

for col in ["t1", "t2", "tc_predicted"]:
    fits[col] = pd.to_datetime(fits[col], errors="coerce")

fits["positive_lppls_valid"] = (
    fits["positive_lppls_valid"]
    .astype(str)
    .str.lower()
    .eq("true")
)

prices = prices.sort_values(DATE_COL).reset_index(drop=True)

if EVENT_COL not in prices.columns:
    raise ValueError(f"Could not find event column: {EVENT_COL}")

events = prices.loc[prices[EVENT_COL] == 1, DATE_COL].sort_values().to_list()

if len(events) == 0:
    raise ValueError(f"No events found where {EVENT_COL} == 1")


# =========================
# EVALUATE EACH EVENT
# =========================
rows = []

for event_date in events:
    event_idx = prices.index[prices[DATE_COL] == event_date][0]

    start_idx = max(0, event_idx - LOOKBACK_TRADING_DAYS)
    start_date = prices.loc[start_idx, DATE_COL]

    event_fits_all = fits[
        (fits["t2"] >= start_date) &
        (fits["t2"] < event_date)
    ].copy()

    event_fits_valid = event_fits_all[
        (event_fits_all["positive_lppls_valid"]) &
        (event_fits_all["tc_predicted"].notna())
    ].copy()

    n_total = len(event_fits_all)
    n_valid = len(event_fits_valid)

    row = {
        "event_date": event_date.strftime("%Y-%m-%d"),
        "n_total_fits": n_total,
        "n_valid_fits": n_valid,
        "valid_fit_pct": n_valid / n_total if n_total > 0 else np.nan,
    }

    if n_valid == 0:
        for metric in [
            "mean_tc_error_days",
            "median_tc_error_days",
            "median_abs_tc_error_days",
            "iqr_tc_error_days",
        ]:
            row[metric] = np.nan

        for h in HIT_WINDOWS:
            row[f"hit_rate_pm_{h}d"] = np.nan

        rows.append(row)
        continue

    event_fits_valid["tc_error_days"] = (
        event_fits_valid["tc_predicted"] - event_date
    ).dt.days

    errors = event_fits_valid["tc_error_days"]
    abs_errors = errors.abs()

    q1 = errors.quantile(0.25)
    q3 = errors.quantile(0.75)

    row.update({
        "mean_tc_error_days": errors.mean(),
        "median_tc_error_days": errors.median(),
        "median_abs_tc_error_days": abs_errors.median(),
        "iqr_tc_error_days": q3 - q1,
    })

    for h in HIT_WINDOWS:
        row[f"hit_rate_pm_{h}d"] = (abs_errors <= h).mean()

    rows.append(row)


# =========================
# DISPLAY TABLE AS PNG
# =========================
metrics = pd.DataFrame(rows)

display_table = metrics.copy()

for col in display_table.columns:
    if col.startswith("valid_fit_pct") or col.startswith("hit_rate"):
        display_table[col] = (100 * display_table[col]).round(2)

numeric_cols = display_table.select_dtypes(include=[np.number]).columns
display_table[numeric_cols] = display_table[numeric_cols].round(2)

display_table = display_table.fillna("").astype(str)

fig_width = max(12, 1.2 * len(display_table.columns))
fig_height = max(2, 0.45 * len(display_table) + 1)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.axis("off")

table = ax.table(
    cellText=display_table.values,
    colLabels=display_table.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold")

plt.tight_layout()
plt.show()