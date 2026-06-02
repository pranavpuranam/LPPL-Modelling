# 06b: print recent confidence peaks

import pandas as pd  # handle dataframes

CSV_FILE = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"  # confidence input
DATE_COL = "t2"  # date column
CI_COL = "positive_bubble_confidence"  # confidence column

df = pd.read_csv(CSV_FILE)  # load confidence data
df[DATE_COL] = pd.to_datetime(df[DATE_COL])  # parse dates
df[CI_COL] = pd.to_numeric(df[CI_COL], errors="coerce")  # force numeric confidence

df = (
    df.dropna(subset=[DATE_COL, CI_COL])
      .sort_values(DATE_COL)
      .reset_index(drop=True)
)  # clean and sort data

full_peak_idx = df[CI_COL].idxmax()  # find full-sample peak row
full_peak_val = df.loc[full_peak_idx, CI_COL]  # get full-sample peak value
full_peak_date = df.loc[full_peak_idx, DATE_COL]  # get full-sample peak date

last_date = df[DATE_COL].max()  # get latest date
six_months_ago = last_date - pd.DateOffset(months=6)  # set recent window start
recent_df = df[df[DATE_COL] >= six_months_ago].copy()  # keep last six months

if recent_df.empty:  # check recent data exists
    raise ValueError("No observations found in the last 6 months of the sample.")  # stop if empty

recent_peak_idx = recent_df[CI_COL].idxmax()  # find recent peak row
recent_peak_val = recent_df.loc[recent_peak_idx, CI_COL]  # get recent peak value
recent_peak_date = recent_df.loc[recent_peak_idx, DATE_COL]  # get recent peak date

print("Confidence score summary")  # print title
print("------------------------")  # print divider
print(f"Full sample peak CI:       {full_peak_val:.6f}")  # print full peak value
print(f"Full sample peak date:     {full_peak_date.date()}")  # print full peak date

print()  # print blank line
print(f"Last sample date:          {last_date.date()}")  # print latest date
print(f"6-month window starts:     {six_months_ago.date()}")  # print recent start date
print(f"Last 6 months peak CI:     {recent_peak_val:.6f}")  # print recent peak value
print(f"Last 6 months peak date:   {recent_peak_date.date()}")  # print recent peak date