import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix)

# -----------------------
# 1) Import
# -----------------------

build = pd.read_csv("C:/Users/Pranav/OneDrive/Desktop/Final-Year-Project/build.csv", parse_dates=["date"])

# -----------------------
# 2) Naive Persistence Model
# -----------------------

y_true = build["psy_bubble"].iloc[1:]
y_pred = build["psy_bubble"].shift(1).iloc[1:]

y_prob = y_pred.copy()

# -----------------------
# 3) Results
# -----------------------

print("Accuracy:", accuracy_score(y_true, y_pred))
print("ROC AUC:", roc_auc_score(y_true, y_prob))
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_true, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))