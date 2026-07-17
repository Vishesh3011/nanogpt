#!/bin/bash
set -e  # exit immediately on any error

echo "[entrypoint] starting nanogpt (prod)"

# Download story model checkpoint from HuggingFace if repo is configured
if [ -n "$HF_TOKEN" ] && [ -n "$HF_REPO_STORIES" ]; then
    echo "[entrypoint] downloading stories checkpoint from $HF_REPO_STORIES"
    python -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id=os.environ['HF_REPO_STORIES'],
    filename='ckpt.pt',
    token=os.environ['HF_TOKEN'],
    local_dir='checkpoints/stories',
)
print(f'[entrypoint] stories checkpoint saved to {path}')
"
else
    echo "[entrypoint] HF_REPO_STORIES not set — stories model will be unavailable"
fi

# Download code model checkpoint from HuggingFace if repo is configured
if [ -n "$HF_TOKEN" ] && [ -n "$HF_REPO_CODE" ]; then
    echo "[entrypoint] downloading code checkpoint from $HF_REPO_CODE"
    python -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(
    repo_id=os.environ['HF_REPO_CODE'],
    filename='ckpt.pt',
    token=os.environ['HF_TOKEN'],
    local_dir='checkpoints/code',
)
print(f'[entrypoint] code checkpoint saved to {path}')
"
else
    echo "[entrypoint] HF_REPO_CODE not set — code model will be unavailable"
fi

echo "[entrypoint] starting uvicorn on port 8000"
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info