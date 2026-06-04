# 03b_naive_bubble_baseline.py

import pandas as pd  # handle dataframes
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)  # calculate classification metrics

build = pd.read_csv(
    "C:/Users/Pranav/OneDrive/Desktop/Final-Year-Project/build.csv",
    parse_dates=["date"]
)  # load final build dataset

y_true = build["psy_bubble"].iloc[1:]  # get true bubble labels from second row onward
y_pred = build["psy_bubble"].shift(1).iloc[1:]  # use previous period label as prediction

y_prob = y_pred.copy()  # use same lagged label as probability score

print("Accuracy:", accuracy_score(y_true, y_pred))  # print accuracy
print("ROC AUC:", roc_auc_score(y_true, y_prob))  # print roc auc
print("Precision:", precision_score(y_true, y_pred, zero_division=0))  # print precision
print("Recall:", recall_score(y_true, y_pred, zero_division=0))  # print recall
print("F1 Score:", f1_score(y_true, y_pred, zero_division=0))  # print f1 score
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))  # print confusion matrix