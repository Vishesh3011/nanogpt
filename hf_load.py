#!/usr/bin/env python3
"""
Upload or download model folders to/from Hugging Face Hub.

Examples:
  python3 hf_load.py upload \
    --local-dir out/my_model \
    --repo-id your-username/my-model \
    --token your_hf_token \
    --private

  python3 hf_load.py download \
    --repo-id your-username/my-model \
    --local-dir out/my_model_downloaded \
    --token your_hf_token

Auth token resolution order:
  1) --token
  2) HF_TOKEN
  3) HUGGINGFACE_HUB_TOKEN
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import create_repo, snapshot_download, upload_folder


def resolve_token(cli_token):
    token = cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    raise ValueError(
        "Missing Hugging Face token. Set --token or env var HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
    )


def upload_model_folder(
    local_dir,
    repo_id,
    token,
    private=False,
    exist_ok=True,
    commit_message="Upload model folder",
    revision="main",
    allow_patterns=None,
    ignore_patterns=None,
):
    local_dir_path = Path(local_dir)
    if not local_dir_path.exists() or not local_dir_path.is_dir():
        raise FileNotFoundError(f"Local model folder not found: {local_dir}")

    create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        repo_type="model",
        exist_ok=exist_ok,
    )

    return upload_folder(
        repo_id=repo_id,
        folder_path=str(local_dir_path),
        repo_type="model",
        token=token,
        commit_message=commit_message,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )


def download_model_folder(
    repo_id,
    local_dir,
    token,
    revision="main",
    allow_patterns=None,
    ignore_patterns=None,
    local_dir_use_symlinks=False,
):
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        revision=revision,
        local_dir=str(local_dir_path),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        local_dir_use_symlinks=local_dir_use_symlinks,
    )


def split_patterns(value):
    if not value:
        return None
    patterns = [x.strip() for x in value.split(",")]
    patterns = [x for x in patterns if x]
    return patterns or None


def build_parser():
    parser = argparse.ArgumentParser(
        description="Upload/download model folders with Hugging Face Hub."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload local folder to HF model repo.")
    upload_parser.add_argument("--local-dir", required=True, help="Local model directory.")
    upload_parser.add_argument("--repo-id", required=True, help="Model repo id, e.g. user/model.")
    upload_parser.add_argument(
        "--token",
        default=None,
        help="HF auth token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN).",
    )
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private if it does not exist.",
    )
    upload_parser.add_argument(
        "--no-exist-ok",
        action="store_true",
        help="Fail if the repo already exists.",
    )
    upload_parser.add_argument(
        "--commit-message",
        default="Upload model folder",
        help="Commit message for upload.",
    )
    upload_parser.add_argument("--revision", default="main", help="Target revision/branch.")
    upload_parser.add_argument(
        "--allow-patterns",
        default=None,
        help="Comma-separated allow patterns, e.g. '*.bin,config.json'.",
    )
    upload_parser.add_argument(
        "--ignore-patterns",
        default=None,
        help="Comma-separated ignore patterns, e.g. '*.tmp,*.log'.",
    )

    download_parser = subparsers.add_parser("download", help="Download HF model repo to local folder.")
    download_parser.add_argument("--repo-id", required=True, help="Model repo id, e.g. user/model.")
    download_parser.add_argument("--local-dir", required=True, help="Destination local directory.")
    download_parser.add_argument(
        "--token",
        default=None,
        help="HF auth token (or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN).",
    )
    download_parser.add_argument("--revision", default="main", help="Branch/tag/commit.")
    download_parser.add_argument(
        "--allow-patterns",
        default=None,
        help="Comma-separated allow patterns, e.g. '*.bin,config.json'.",
    )
    download_parser.add_argument(
        "--ignore-patterns",
        default=None,
        help="Comma-separated ignore patterns, e.g. '*.tmp,*.log'.",
    )
    download_parser.add_argument(
        "--symlinks",
        action="store_true",
        help="Use symlinks in local_dir when available.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    token = resolve_token(getattr(args, "token", None))

    allow_patterns = split_patterns(getattr(args, "allow_patterns", None))
    ignore_patterns = split_patterns(getattr(args, "ignore_patterns", None))

    if args.command == "upload":
        result = upload_model_folder(
            local_dir=args.local_dir,
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            exist_ok=not args.no_exist_ok,
            commit_message=args.commit_message,
            revision=args.revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        print("Upload complete.")
        print(f"Repo: {args.repo_id}")
        print(f"Result: {result}")
        return

    if args.command == "download":
        local_path = download_model_folder(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            token=token,
            revision=args.revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_dir_use_symlinks=args.symlinks,
        )
        print("Download complete.")
        print(f"Repo: {args.repo_id}")
        print(f"Local path: {local_path}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
