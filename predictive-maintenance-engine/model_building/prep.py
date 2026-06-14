import pandas as pd
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Ensure data directory structure exists
os.makedirs("data/train", exist_ok=True)
os.makedirs("data/test", exist_ok=True)

# Load the dataset from Hugging Face
# Using HF_TOKEN from environment for potential private datasets
hf_token = os.getenv("HF_TOKEN")
dataset = load_dataset("HF-Sum/predictive-maintenance-engine-data", token=hf_token)

# Convert the 'train' split of the dataset to a pandas DataFrame
df_engine = dataset['train'].to_pandas()

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

# Save processed data locally
train_df.to_csv("data/train/train_data.csv", index=False)
test_df.to_csv("data/test/test_data.csv", index=False)

print("Data preparation complete. Train and test datasets saved to 'data/' directory.")
