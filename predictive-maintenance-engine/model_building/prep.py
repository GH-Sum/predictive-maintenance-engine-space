import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- DYNAMIC PATH RESOLUTION ---
# Finds the directory where this prep.py script is executing
script_dir = os.path.dirname(os.path.abspath(__file__))

# Resolves the 'data' directory relative to the script location (one level up)
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))

# Create output subdirectories dynamically
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# --- LOCAL DATA LOADING ---
# Find your raw data file inside the local repository layout.
# Assuming you have a CSV data file committed or copied into your data folder:
# If your raw file has a specific name (e.g., 'engine_data.csv'), change it here.
raw_data_path = os.path.join(data_dir, "predictive_maintenance.csv") 

# FALLBACK CHECK: If your raw file sits directly inside the train/ folder as 'train_data.csv' before processing
if not os.path.exists(raw_data_path):
    raw_data_path = os.path.join(train_dir, "train_data.csv")

print(f"Loading raw dataset from local path: {raw_data_path}")
df_engine = pd.read_csv(raw_data_path)

# Standardize column names
df_engine.columns = (
    df_engine.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

# Separate features and target
X = df_engine.drop("engine_condition", axis=1)
y = df_engine["engine_condition"]

# Perform stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Recombine train/test data for saving
train_df = X_train.copy()
train_df["engine_condition"] = y_train

test_df = X_test.copy()
test_df["engine_condition"] = y_test

# Save processed data locally using dynamic paths
train_output_path = os.path.join(train_dir, "train_data.csv")
test_output_path = os.path.join(test_dir, "test_data.csv")

train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)

print(f"Data preparation complete. Train dataset saved to: {train_output_path}")
print(f"Test dataset saved to: {test_output_path}")
