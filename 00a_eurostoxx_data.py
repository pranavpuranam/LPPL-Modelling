# 00a: clean eurostoxx data

import yfinance as yf  # download market data
import pandas as pd  # handle dataframes
import numpy as np  # numerical operations

ticker = "^STOXX"  # stoxx europe 600 ticker

df = yf.download(
    ticker,
    start="1986-01-01",
    end=None,
    interval="1d",
    auto_adjust=False,
    progress=False
)  # download daily price data

if isinstance(df.columns, pd.MultiIndex):  # check for multiindex columns
    df.columns = df.columns.get_level_values(0)  # flatten columns

df = df.reset_index()  # move date from index to column

df = df[["Date", "Close"]].copy()  # keep required columns

df["Date"] = pd.to_datetime(df["Date"])  # parse dates
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")  # force numeric prices

print("Initial rows:", len(df))  # print raw row count
print("Missing Close values:", df["Close"].isna().sum())  # count missing prices
print("Duplicate dates:", df["Date"].duplicated().sum())  # count duplicate dates
print("Non-positive Close values:", (df["Close"] <= 0).sum())  # count invalid prices

df = df.dropna(subset=["Date", "Close"])  # remove missing dates/prices
df = df[df["Close"] > 0]  # remove non-positive prices
df = df.drop_duplicates(subset=["Date"])  # remove duplicate dates
df = df.sort_values("Date").reset_index(drop=True)  # sort chronologically

df["log_price"] = np.log(df["Close"])  # calculate log price

print("\nFinal rows:", len(df))  # print cleaned row count
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())  # print sample range
print("Missing values after cleaning:")  # print missing value header
print(df.isna().sum())  # count final missing values

df.to_csv("eurostoxx600_prices.csv", index=False)  # save cleaned dataset