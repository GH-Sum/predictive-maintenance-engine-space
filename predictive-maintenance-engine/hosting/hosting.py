from huggingface_hub import HfApi
import os
import shutil

# --- ARTIFACT COLLECTION STEPS ---
# Define paths relative to Colab root / workspace structure
src_requirements = "/content/predictive-maintenance-engine-space/predictive-maintenance-engine/requirements.txt"
src_model = "/content/predictive-maintenance-engine-space/predictive-maintenance-engine/model_building/best_model.joblib"
dest_folder = "/content/deployment"

# Ensure the deployment folder exists
os.makedirs(dest_folder, exist_ok=True)

# Copy requirements.txt so Hugging Face can see and install 'dill'
if os.path.exists(src_requirements):
    shutil.copy(src_requirements, os.path.join(dest_folder, "requirements.txt"))
    print("Copied requirements.txt to deployment directory.")

# Copy the trained model so the web application can access it
if os.path.exists(src_model):
    shutil.copy(src_model, os.path.join(dest_folder, "best_model.joblib"))
    print("Copied best_model.joblib to deployment directory.")

# --- HUGGING FACE UPLOAD ---
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path=dest_folder,    
    repo_id="HF-Sum/predictive-maintenance-engine-space",          
    repo_type="space",                      
    path_in_repo="",                          
)
print("Successfully uploaded deployment folder to Hugging Face Space!")
