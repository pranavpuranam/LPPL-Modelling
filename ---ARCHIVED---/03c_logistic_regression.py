# 03c_logistic_regression_tests.py

import pandas as pd  # handle dataframes
from sklearn.linear_model import LogisticRegression  # fit logistic regression
from sklearn.metrics import accuracy_score, roc_auc_score  # calculate model metrics

build = pd.read_csv(
    "C:/Users/Pranav/OneDrive/Desktop/Final-Year-Project/build.csv",
    parse_dates=["date"]
)  # load final build dataset

"""
cols_exclude = ["date", "psy_bubble"]  # columns not standardised
cols_to_z = [c for c in build.columns if c not in cols_exclude]  # columns to standardise

z_build = build[cols_to_z] = (
    build[cols_to_z] - build[cols_to_z].mean()
) / build[cols_to_z].std()  # z-score selected columns

z_build.to_csv("temp.csv", index=False)  # save temporary standardised data
"""  # old standardisation test

from sklearn.linear_model import LogisticRegression  # fit logistic regression
from sklearn.model_selection import TimeSeriesSplit  # time-series split tool
from sklearn.metrics import accuracy_score, roc_auc_score  # calculate model metrics

X = build[["sp_comp_p"]]  # use price only as feature
y = build["psy_bubble"]  # set bubble label as target

split = int(0.8 * len(build))  # set 80 percent train split

X_train, X_test = X.iloc[:split], X.iloc[split:]  # split features by time
y_train, y_test = y.iloc[:split], y.iloc[split:]  # split labels by time

model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000
)  # define logistic regression model

model.fit(X_train, y_train)  # fit model on training data

y_pred = model.predict(X_test)  # predict test labels
y_prob = model.predict_proba(X_test)[:, 1]  # predict bubble probabilities

print("Model 1 (Price Only)")  # print model name
print("Accuracy:", accuracy_score(y_test, y_pred))  # print accuracy
print("ROC AUC:", roc_auc_score(y_test, y_prob))  # print roc auc

X = build.drop(columns=["date", "psy_bubble"])  # use all variables except date and label
y = build["psy_bubble"]  # set bubble label as target

split = int(0.8 * len(build))  # set 80 percent train split

X_train, X_test = X.iloc[:split], X.iloc[split:]  # split features by time
y_train, y_test = y.iloc[:split], y.iloc[split:]  # split labels by time

model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000
)  # define logistic regression model

model.fit(X_train, y_train)  # fit model on training data

y_pred = model.predict(X_test)  # predict test labels
y_prob = model.predict_proba(X_test)[:, 1]  # predict bubble probabilities

print("Accuracy:", accuracy_score(y_test, y_pred))  # print accuracy
print("ROC AUC:", roc_auc_score(y_test, y_prob))  # print roc auc