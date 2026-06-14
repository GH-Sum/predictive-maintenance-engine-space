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

# --- DYNAMIC PATH RESOLUTION ---
# Finds the directory where this train.py script is executing
script_dir = os.path.dirname(os.path.abspath(__file__))

# Resolves the 'data' directory relative to the script location (one level up)
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

# Target model directory matches where the script lives
model_dir = script_dir

# Ensure model saving directory exists
os.makedirs(model_dir, exist_ok=True)


# Define evaluation function
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

# Load preprocessed data using the dynamically resolved data paths
train_path = os.path.join(data_dir, "train", "train_data.csv")
test_path = os.path.join(data_dir, "test", "test_data.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Separate features and target
X_train = train_df.drop("engine_condition", axis=1)
y_train = train_df["engine_condition"]
X_test = test_df.drop("engine_condition", axis=1)
y_test = test_df["engine_condition"]

# Define the XGBoost model with best parameters found in EDA
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

# Save the best model locally using the dynamic path
model_output_path = os.path.join(model_dir, "best_model.joblib")
joblib.dump(xgb_best, model_output_path)
print(f"Trained XGBoost model saved to {model_output_path}")
