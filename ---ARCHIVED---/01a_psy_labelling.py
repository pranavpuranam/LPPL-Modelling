# 01a_build_bubble_raw.py

import pandas as pd  # handle dataframes
import numpy as np  # numerical operations
import matplotlib.pyplot as plt  # plot charts

df = pd.read_csv("shiller_data_clean.csv", parse_dates=["date"])  # load cleaned shiller data

shiller_price = df[["date", "sp_comp_p"]].copy()  # keep date and s&p composite price
shiller_price["log_p"] = np.log(shiller_price["sp_comp_p"])  # calculate log price
shiller_price.to_csv("bubble_raw.csv", index=False)  # save raw bubble input data