import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -----------------------
# 1) Import and Sort
# -----------------------

main = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/shiller_build_labelled.csv", parse_dates=["date"])
gold = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/GOLD.csv", parse_dates=["date"])
credit = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/BANK_CREDIT.csv", parse_dates=["date"])
michigan = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/MICHIGAN_SENTIMENT.csv", parse_dates=["date"])
vix = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/VIX.csv", parse_dates=["date"])
wti = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/WTI.csv", parse_dates=["date"])

main = main.sort_values("date")
gold = gold.sort_values("date")
credit = credit.sort_values("date")
michigan = michigan.sort_values("date")
vix = vix.sort_values("date")
wti = wti.sort_values("date")

# -----------------------
# 2) Merge
# -----------------------

tol = pd.Timedelta(days=5)

def asof_merge(left, right, value_col, new_name):
    return pd.merge_asof(
        left,
        right[["date", value_col]].rename(columns={value_col: new_name}),
        on="date",
        direction="nearest",
        tolerance=tol
    )

# Merge sequentially
main = asof_merge(main, gold, "gold_avg_spot_price", "gold_spot_price")
main = asof_merge(main, credit, "TOTBKCR", "bank_credit")
main = asof_merge(main, michigan, "UMCSENT", "michigan_sentiment")
main = asof_merge(main, vix, "VIXCLS", "vix")
main = asof_merge(main, wti, "WTISPLC", "wti_price")


# -----------------------
# 2) Interpolate missing values and remove empty rows
# -----------------------

cols_to_interp = [
    "gold_spot_price",
    "bank_credit",
    "michigan_sentiment",
    "vix",
    "wti_price",
]

mask = (main["date"] >= "1990-01-01") & (main["date"] <= "2025-12-31")

main.loc[mask, cols_to_interp] = (
    main.loc[mask]
        .set_index("date")[cols_to_interp]
        .interpolate(method="time")
        .values
)

main = main.dropna().reset_index(drop=True)

print(main.isna().sum())

main.to_csv("build.csv", index=False)