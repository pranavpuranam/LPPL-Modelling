import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1) Import cleaned Shiller data
# -----------------------
df = pd.read_csv("shiller_data_clean.csv", parse_dates=["date"])

# -----------------------
# 2) Build shiller_price DataFrame
# -----------------------

shiller_price = df[["date", "sp_comp_p"]].copy()
shiller_price["log_p"] = np.log(shiller_price["sp_comp_p"])
shiller_price.to_csv("bubble_raw.csv", index=False)