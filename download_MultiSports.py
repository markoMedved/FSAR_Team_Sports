from huggingface_hub import snapshot_download
from huggingface_hub import login

# This will open a text box for your token
login()

snapshot_download(
    repo_id="MCG-NJU/MultiSports",
    repo_type="dataset",
    local_dir="./multisports_data"
)

