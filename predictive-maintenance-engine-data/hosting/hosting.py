from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="predictive-maintenance-engine-data/deployment",    
    repo_id="HF-Sum/predictive-maintenance-engine-space",          
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
#api.upload_file(
#    path_or_fileobj="README.md",
#    path_in_repo="README_test.md",
#    repo_id="HF-Sum/predictive-maintenance-engine-space",
#    repo_type="space",
#)
