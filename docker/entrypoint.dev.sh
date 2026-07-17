#!/bin/bash
set -e

echo "[entrypoint] starting nanogpt-rebuilt (dev)"

# Checkpoint paths are expected to be bind-mounted by docker-compose.
# Uncomment these checks once local checkpoints exist:
# if [ ! -f "checkpoints/stories/ckpt.pt" ]; then
#     echo "[entrypoint] WARNING: checkpoints/stories/ckpt.pt not found — stories model unavailable"
# fi
# if [ ! -f "checkpoints/code/ckpt.pt" ]; then
#     echo "[entrypoint] WARNING: checkpoints/code/ckpt.pt not found — code model unavailable"
# fi

echo "[entrypoint] starting uvicorn with --reload (dev mode)"
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --reload \
    --log-level debug