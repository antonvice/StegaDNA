from huggingface_hub import upload_file, upload_folder
import os

repo_id = "anton96vice/StegaDNA-V5-StegaStamp"

print("Uploading hf_readme.md to README.md...")
upload_file(
    path_or_fileobj="./hf_readme.md",
    path_in_repo="README.md",
    repo_id=repo_id
)

print("Uploading folder contents...")
upload_folder(
    folder_path=".",
    repo_id=repo_id,
    ignore_patterns=[
        ".venv/*", 
        ".git/*", 
        "wandb/*", 
        "__pycache__/*", 
        "hf_readme.md", 
        "README.md", 
        ".ruff_cache/*", 
        ".DS_Store*"
    ]
)
print("Done!")
