from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "HF-Sum/predictive-maintenance-engine-data" 
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Dynamic Absolute Path Resolution:
# 1. Finds the directory where this script is running (model_building/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Navigates one level up to the project root and into the 'data' folder
resolved_data_path = os.path.abspath(os.path.join(script_dir, "..", "data"))

print(f"Uploading files from resolved path: {resolved_data_path}")

api.upload_folder(
    folder_path=resolved_data_path, 
    repo_id="HF-Sum/predictive-maintenance-engine-data",
    repo_type="dataset"
)

