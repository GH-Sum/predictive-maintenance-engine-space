from huggingface_hub import HfApi
import os
import shutil

# --- DYNAMIC PATH RESOLUTION ---
# Finds the directory where this hosting.py script lives (.../hosting)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Safely goes up one level to the project root directory (.../predictive-maintenance-engine)
project_root = os.path.abspath(os.path.join(script_dir, ".."))

# Goes up another level to find where 'deployment' should sit in the workspace root
workspace_root = os.path.abspath(os.path.join(project_root, ".."))

# Resolve source and destination paths dynamically
src_requirements = os.path.join(project_root, "requirements.txt")
src_model = os.path.join(project_root, "model_building", "best_model.joblib")
dest_folder = os.path.join(workspace_root, "deployment")

# Ensure the deployment folder exists
os.makedirs(dest_folder, exist_ok=True)

# Copy requirements.txt so Hugging Face can install 'dill'
if os.path.exists(src_requirements):
    shutil.copy(src_requirements, os.path.join(dest_folder, "requirements.txt"))
    print(f"Copied requirements.txt to {dest_folder}")
else:
    print(f"WARNING: requirements.txt not found at {src_requirements}")

# Copy the trained model asset
if os.path.exists(src_model):
    shutil.copy(src_model, os.path.join(dest_folder, "best_model.joblib"))
    print(f"Copied best_model.joblib to {dest_folder}")
else:
    print(f"WARNING: best_model.joblib not found at {src_model}")

# --- HUGGING FACE UPLOAD ---
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=dest_folder,    
    repo_id="HF-Sum/predictive-maintenance-engine-space",          
    repo_type="space",                      
    path_in_repo="",                          
)
print("Successfully uploaded deployment folder to Hugging Face Space!")
