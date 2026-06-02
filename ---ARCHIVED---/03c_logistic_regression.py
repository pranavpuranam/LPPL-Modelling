import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------
# 1) Import
# -----------------------

build = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/Final-Year-Project/build.csv", parse_dates=["date"])

# -----------------------
# 2) Clean
# -----------------------

"""
cols_exclude = ["date", "psy_bubble"]
cols_to_z = [c for c in build.columns if c not in cols_exclude]

z_build = build[cols_to_z] = (
    build[cols_to_z] - build[cols_to_z].mean()
) / build[cols_to_z].std()

z_build.to_csv("temp.csv", index=False)
"""

# -----------------------
# 3) Model 1: Just Price
# -----------------------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

X = build[["sp_comp_p"]]   # only price column
y = build["psy_bubble"]

split = int(0.8 * len(build))

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Model 1 (Price Only)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))


# -----------------------
# 3) Model 2: All
# -----------------------

X = build.drop(columns=["date", "psy_bubble"])
y = build["psy_bubble"]

split = int(0.8 * len(build))

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))