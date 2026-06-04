# 02a_build_macro_features.py

import pandas as pd  # handle dataframes
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plot charts
import matplotlib.dates as mdates  # format date axis

main = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/shiller_build_labelled.csv", parse_dates=["date"])  # load labelled shiller dataset
gold = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/GOLD.csv", parse_dates=["date"])  # load gold price data
credit = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/BANK_CREDIT.csv", parse_dates=["date"])  # load bank credit data
michigan = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/MICHIGAN_SENTIMENT.csv", parse_dates=["date"])  # load consumer sentiment data
vix = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/VIX.csv", parse_dates=["date"])  # load vix data
wti = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/WTI.csv", parse_dates=["date"])  # load oil price data

main = main.sort_values("date")  # sort main data by date
gold = gold.sort_values("date")  # sort gold data by date
credit = credit.sort_values("date")  # sort credit data by date
michigan = michigan.sort_values("date")  # sort sentiment data by date
vix = vix.sort_values("date")  # sort vix data by date
wti = wti.sort_values("date")  # sort oil data by date

tol = pd.Timedelta(days=5)  # set merge tolerance

def asof_merge(left, right, value_col, new_name):
    return pd.merge_asof(
        left,
        right[["date", value_col]].rename(columns={value_col: new_name}),
        on="date",
        direction="nearest",
        tolerance=tol
    )  # merge nearest dated value into main data

main = asof_merge(main, gold, "gold_avg_spot_price", "gold_spot_price")  # add gold price
main = asof_merge(main, credit, "TOTBKCR", "bank_credit")  # add bank credit
main = asof_merge(main, michigan, "UMCSENT", "michigan_sentiment")  # add sentiment
main = asof_merge(main, vix, "VIXCLS", "vix")  # add vix
main = asof_merge(main, wti, "WTISPLC", "wti_price")  # add oil price

cols_to_interp = [
    "gold_spot_price",
    "bank_credit",
    "michigan_sentiment",
    "vix",
    "wti_price",
]  # columns to interpolate

mask = (main["date"] >= "1990-01-01") & (main["date"] <= "2025-12-31")  # define modelling window

main.loc[mask, cols_to_interp] = (
    main.loc[mask]
        .set_index("date")[cols_to_interp]
        .interpolate(method="time")
        .values
)  # interpolate missing values over time

main = main.dropna().reset_index(drop=True)  # drop remaining missing rows

print(main.isna().sum())  # check missing values

main.to_csv("build.csv", index=False)  # save final build dataset