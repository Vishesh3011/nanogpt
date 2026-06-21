#!/usr/bin/env python3
"""
Upload or download model checkpoint folders to/from the Hugging Face Hub.

Auth token resolution order:
  1) --token CLI argument
  2) HF_TOKEN environment variable
  3) HUGGINGFACE_HUB_TOKEN environment variable

Examples:
    python -m scripts.hf_sync upload \
        --local-dir checkpoints/stories \
        --repo-id your-username/nanogpt-stories \
        --private

    python -m scripts.hf_sync download \
        --repo-id your-username/nanogpt-stories \
        --local-dir checkpoints/stories
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

from huggingface_hub import create_repo, snapshot_download, upload_folder


def resolve_token(cli_token: Optional[str]) -> str:
    """Resolve the HF auth token from CLI arg or environment variables."""
    token = cli_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    raise ValueError(
        "Missing Hugging Face token. Set --token or env var HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
    )


def upload_model_folder(
    local_dir: str,
    repo_id: str,
    token: str,
    private: bool = False,
    exist_ok: bool = True,
    commit_message: str = "Upload model folder",
    revision: str = "main",
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> str:
    """Create (if needed) a model repo and upload `local_dir` to it."""
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
    repo_id: str,
    local_dir: str,
    token: str,
    revision: str = "main",
    allow_patterns: Optional[List[str]] = None,
    ignore_patterns: Optional[List[str]] = None,
) -> str:
    """Download a model repo into `local_dir`."""
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
    )


def _split_patterns(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    patterns = [x.strip() for x in value.split(",") if x.strip()]
    return patterns or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload/download checkpoints with the Hugging Face Hub.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload_parser = subparsers.add_parser("upload", help="Upload a local folder to an HF model repo.")
    upload_parser.add_argument("--local-dir", required=True)
    upload_parser.add_argument("--repo-id", required=True)
    upload_parser.add_argument("--token", default=None)
    upload_parser.add_argument("--private", action="store_true")
    upload_parser.add_argument("--no-exist-ok", action="store_true")
    upload_parser.add_argument("--commit-message", default="Upload model folder")
    upload_parser.add_argument("--revision", default="main")
    upload_parser.add_argument("--allow-patterns", default=None)
    upload_parser.add_argument("--ignore-patterns", default=None)

    download_parser = subparsers.add_parser("download", help="Download an HF model repo to a local folder.")
    download_parser.add_argument("--repo-id", required=True)
    download_parser.add_argument("--local-dir", required=True)
    download_parser.add_argument("--token", default=None)
    download_parser.add_argument("--revision", default="main")
    download_parser.add_argument("--allow-patterns", default=None)
    download_parser.add_argument("--ignore-patterns", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    token = resolve_token(getattr(args, "token", None))

    allow_patterns = _split_patterns(getattr(args, "allow_patterns", None))
    ignore_patterns = _split_patterns(getattr(args, "ignore_patterns", None))

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
        print(f"Upload complete. Repo: {args.repo_id}. Result: {result}")
        return

    if args.command == "download":
        local_path = download_model_folder(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            token=token,
            revision=args.revision,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        print(f"Download complete. Repo: {args.repo_id}. Local path: {local_path}")
        return


if __name__ == "__main__":
    main()
