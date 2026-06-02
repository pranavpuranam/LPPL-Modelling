import pandas as pd

file_path = "C:/Users/Pranav/OneDrive/Desktop/GITHUB/Final-Year-Project/Gold_price_averages_in_a range_of_currencies_since_1978.xlsx"

df = pd.read_excel(file_path)

df = pd.read_excel(
    file_path,
    sheet_name="Monthly_Avg"
)

df = df.iloc[5:]
df = df.iloc[:, [2, 3]]
df = df.reset_index(drop=True)

df.columns = ["date", "gold_avg_spot_price"]

print(df.head())

df.to_csv("gold_spot.csv", index=False)