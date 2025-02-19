# Install the huggingface_hub package if you haven't already:
# pip install huggingface_hub

from huggingface_hub import snapshot_download
import os

# Replace with the actual repository ID for Llama3 8B.
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # e.g., "meta/Llama3-8B" or another valid ID

# Define the directory where the model will be stored.
local_dir = ":/media/data/parsafar/models/Llama3_8B_instruc"

# Download the repository snapshot locally.
print("Downloading Llama3 8B...")
snapshot_download(repo_id=repo_id, local_dir=local_dir)
print("Download complete!")
print("Model stored in:", os.path.abspath(local_dir))
