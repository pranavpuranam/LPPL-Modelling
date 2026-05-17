# pip install yfinance pandas numpy

import yfinance as yf
import pandas as pd
import numpy as np

ticker = "^STOXX"  # EURO STOXX 50 Index

df = yf.download(
    ticker,
    start="1986-01-01",   # yfinance will return from earliest available date
    end=None,
    interval="1d",
    auto_adjust=False,
    progress=False
)

# Clean column names if yfinance returns multi-index columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# Keep only Date and Close
df = df[["Date", "Close"]].copy()

# Clean types
df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Basic data quality checks
print("Initial rows:", len(df))
print("Missing Close values:", df["Close"].isna().sum())
print("Duplicate dates:", df["Date"].duplicated().sum())
print("Non-positive Close values:", (df["Close"] <= 0).sum())

# Remove invalid rows
df = df.dropna(subset=["Date", "Close"])
df = df[df["Close"] > 0]
df = df.drop_duplicates(subset=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Add LPPL variable
df["log_price"] = np.log(df["Close"])

# Final checks
print("\nFinal rows:", len(df))
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
print("Missing values after cleaning:")
print(df.isna().sum())

# Save
df.to_csv("eurostoxx600_prices.csv", index=False)