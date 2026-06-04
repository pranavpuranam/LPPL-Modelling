# 00a_clean_shiller_data.py

import pandas as pd  # handle dataframes
import numpy as np  # numerical operations
import re  # clean text with regex
from pandas.tseries.offsets import MonthEnd  # move dates to month end

FILE = "ie_data.xls"  # raw shiller data file
SHEET = "Data"  # excel sheet name
HEADER_ROWS = 8  # number of header rows
DATA_START = 8  # first data row

hdr = pd.read_excel(FILE, sheet_name=SHEET, header=None, nrows=HEADER_ROWS)  # load header rows

hdr.iat[1, 0] = np.nan  # blank repeated header cell
hdr.iat[2, 0] = np.nan  # blank repeated header cell

def clean_token(x):
    if pd.isna(x):  # check missing value
        return ""  # return empty string
    s = str(x).strip()  # convert to clean string
    s = re.sub(r"\s+", " ", s)  # collapse spaces
    return s  # return cleaned token

colnames = []  # store column names
for j in range(hdr.shape[1]):  # loop over columns
    parts = [clean_token(hdr.iat[i, j]) for i in range(HEADER_ROWS)]  # collect header parts
    parts = [p for p in parts if p]  # remove empty parts
    name = " ".join(parts)  # combine header parts
    name = re.sub(r"\s+", " ", name).strip()  # clean combined name
    colnames.append(name)  # store column name

keep = [i for i, c in enumerate(colnames) if c]  # keep named columns only
colnames = [colnames[i] for i in keep]  # filter column names

df = pd.read_excel(FILE, sheet_name=SHEET, header=None, skiprows=DATA_START)  # load data rows
df = df.iloc[:, keep]  # keep named columns only
df.columns = colnames  # assign cleaned headers

df = df.iloc[:-2]  # drop footer rows
df = df.iloc[:, :-3]  # drop final unused columns

df.columns = (
    df.columns
      .str.strip()  # remove edge spaces
      .str.lower()  # lowercase column names
      .str.replace(r"[^\w\s]", "", regex=True)  # remove punctuation
      .str.replace(r"\s+", "_", regex=True)  # replace spaces with underscores
)

s = df["date"].astype(str).str.strip()  # convert date column to strings
parts = s.str.split(".", n=1, expand=True)  # split year and month code

year = parts[0].astype(int)  # extract year
mstr = parts[1].fillna("").str.strip()  # extract month string

month = pd.to_numeric(
    mstr.where(mstr.str.len() != 1, "10"),  # handle october formatting issue
    errors="raise"
).astype(int)  # convert month to integer

df["date"] = (
    pd.to_datetime(dict(year=year, month=month, day=1))  # create first-of-month dates
    + MonthEnd(0)  # move dates to month end
)

df = df.drop(columns=["date_fraction"], errors="ignore")  # remove unused date fraction column

rows_with_na = df[df.isna().any(axis=1)]  # find rows with missing values

if not rows_with_na.empty:  # check if missing rows exist
    print("Dropping rows with missing values on these dates:")  # print warning
    for d in rows_with_na["date"]:  # loop through missing rows
        print(f"  - {d.strftime('%Y-%m-%d')}")  # print dropped date
else:
    print("No rows with missing values found.")  # print clean data message

df = df.dropna(axis=0, how="any")  # drop rows with missing values

df.to_csv("shiller_data_clean.csv", index=False)  # save cleaned shiller data