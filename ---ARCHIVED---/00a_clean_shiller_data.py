import pandas as pd
import numpy as np
import re
from pandas.tseries.offsets import MonthEnd

# -----------------------
# PARAMETERS
# -----------------------
FILE = "ie_data.xls"
SHEET = "Data"
HEADER_ROWS = 8      # rows 0–7 are header text
DATA_START = 8       # first data row

# -----------------------
# 1) READ HEADER BLOCK
# -----------------------
hdr = pd.read_excel(FILE, sheet_name=SHEET, header=None, nrows=HEADER_ROWS)

# Remove values in 2nd and 3rd row of 1st column
hdr.iat[1, 0] = np.nan
hdr.iat[2, 0] = np.nan

def clean_token(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

# Build column names by vertical concatenation
colnames = []
for j in range(hdr.shape[1]):
    parts = [clean_token(hdr.iat[i, j]) for i in range(HEADER_ROWS)]
    parts = [p for p in parts if p]
    name = " ".join(parts)
    name = re.sub(r"\s+", " ", name).strip()
    colnames.append(name)

keep = [i for i, c in enumerate(colnames) if c]
colnames = [colnames[i] for i in keep]

# -----------------------
# 2) READ DATA
# -----------------------
df = pd.read_excel(FILE, sheet_name=SHEET, header=None, skiprows=DATA_START)
df = df.iloc[:, keep]
df.columns = colnames

# -----------------------
# 3) DROP LAST ROW + LAST 3 COLUMNS
# -----------------------
df = df.iloc[:-2]        # drop final incomplete row
df = df.iloc[:, :-3]     # drop forward-looking 10y columns

# -----------------------
# 4) SNAKE_CASE COLUMN NAMES
# -----------------------
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r"[^\w\s]", "", regex=True)
      .str.replace(r"\s+", "_", regex=True)
)

# -----------------------
# 5) FIX SHILLER DATE (YYYY.MM)
# -----------------------
s = df["date"].astype(str).str.strip()
parts = s.str.split(".", n=1, expand=True)

year = parts[0].astype(int)
mstr = parts[1].fillna("").str.strip()

# Shiller quirk: ".1" = October
month = pd.to_numeric(
    mstr.where(mstr.str.len() != 1, "10"),
    errors="raise"
).astype(int)

df["date"] = (
    pd.to_datetime(dict(year=year, month=month, day=1))
    + MonthEnd(0)
)

# -----------------------
# 6) DROP date_fraction COLUMN
# -----------------------
df = df.drop(columns=["date_fraction"], errors="ignore")

# -----------------------
# 6b) DROP ROWS WITH ANY MISSING VALUES (AND LOG THEM)
# -----------------------

rows_with_na = df[df.isna().any(axis=1)]

if not rows_with_na.empty:
    print("Dropping rows with missing values on these dates:")
    for d in rows_with_na["date"]:
        print(f"  - {d.strftime('%Y-%m-%d')}")
else:
    print("No rows with missing values found.")

df = df.dropna(axis=0, how="any")

# -----------------------
# 7) OUTPUT
# -----------------------
df.to_csv("shiller_data_clean.csv", index=False)