from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

repo_id = "Sandhya-2025/tourism-package-purchase"
repo_type = "dataset"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is not set")

api = HfApi(token=HF_TOKEN)

# Step 1: Ensure dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating it...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        token=HF_TOKEN,
        private=False,
        exist_ok=True
    )
    print(f"Dataset repo '{repo_id}' created.")
except HfHubHTTPError as e:
    raise RuntimeError(
        "Authentication failed. Check HF_TOKEN permissions (dataset + write)."
    ) from e

# Step 2: Upload dataset
api.upload_folder(
    folder_path="data",   # ✅ correct relative path
    repo_id=repo_id,
    repo_type=repo_type,
)
