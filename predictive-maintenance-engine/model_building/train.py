import pandas as pd
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Ensure model saving directory exists
os.makedirs("/content/model_building", exist_ok=True)

# Define evaluation function (copied from notebook for self-containment)
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred)
    }
    return results

# Load preprocessed data
train_df = pd.read_csv("data/train/train_data.csv")
test_df = pd.read_csv("data/test/test_data.csv")

# Separate features and target
X_train = train_df.drop("engine_condition", axis=1)
y_train = train_df["engine_condition"]
X_test = test_df.drop("engine_condition", axis=1)
y_test = test_df["engine_condition"]

# Define the XGBoost model with best parameters found in EDA
# These parameters were identified as {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100}
# in the previous GridSearch step.
xgb_best = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    learning_rate=0.01,
    max_depth=3,
    n_estimators=100
)

# Train the model
xgb_best.fit(X_train, y_train)

# Evaluate the model
xgb_results = evaluate_model(xgb_best, X_test, y_test)

print("XGBoost Model Training and Evaluation Complete.")
print("Evaluation Results:", xgb_results)

# Save the best model locally
joblib.dump(xgb_best, "/content/model_building/best_model.joblib")
print("Trained XGBoost model saved to /content/model_building/best_model.joblib")
