# 02b_clean_gold_spot.py

import pandas as pd  # handle dataframes

file_path = "C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/Gold_price_averages_in_a range_of_currencies_since_1978.xlsx"  # raw gold data file

df = pd.read_excel(file_path)  # load workbook once

df = pd.read_excel(
    file_path,
    sheet_name="Monthly_Avg"
)  # load monthly average sheet

df = df.iloc[5:]  # drop header rows
df = df.iloc[:, [2, 3]]  # keep date and gold price columns
df = df.reset_index(drop=True)  # reset row index

df.columns = ["date", "gold_avg_spot_price"]  # rename columns

print(df.head())  # preview cleaned data

df.to_csv("gold_spot.csv", index=False)  # save cleaned gold price data