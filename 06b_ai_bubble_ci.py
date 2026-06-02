import pandas as pd

# =========================
# CONFIG
# =========================

CSV_FILE = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"
DATE_COL = "t2"
CI_COL = "positive_bubble_confidence"

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(CSV_FILE)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df[CI_COL] = pd.to_numeric(df[CI_COL], errors="coerce")

df = (
    df.dropna(subset=[DATE_COL, CI_COL])
      .sort_values(DATE_COL)
      .reset_index(drop=True)
)

# =========================
# FULL-SAMPLE PEAK
# =========================

full_peak_idx = df[CI_COL].idxmax()
full_peak_val = df.loc[full_peak_idx, CI_COL]
full_peak_date = df.loc[full_peak_idx, DATE_COL]

# =========================
# LAST 6 MONTHS PEAK
# =========================

last_date = df[DATE_COL].max()
six_months_ago = last_date - pd.DateOffset(months=6)

recent_df = df[df[DATE_COL] >= six_months_ago].copy()

if recent_df.empty:
    raise ValueError("No observations found in the last 6 months of the sample.")

recent_peak_idx = recent_df[CI_COL].idxmax()
recent_peak_val = recent_df.loc[recent_peak_idx, CI_COL]
recent_peak_date = recent_df.loc[recent_peak_idx, DATE_COL]

# =========================
# PRINT RESULTS
# =========================

print("Confidence score summary")
print("------------------------")
print(f"Full sample peak CI:       {full_peak_val:.6f}")
print(f"Full sample peak date:     {full_peak_date.date()}")

print()
print(f"Last sample date:          {last_date.date()}")
print(f"6-month window starts:     {six_months_ago.date()}")
print(f"Last 6 months peak CI:     {recent_peak_val:.6f}")
print(f"Last 6 months peak date:   {recent_peak_date.date()}")