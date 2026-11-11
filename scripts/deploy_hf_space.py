#!/usr/bin/env python3
"""Push hf_space/ to a Hugging Face Space.

Uploads only what the Space needs, so a 6.5 MB Lean development and a repo of
run logs do not travel with a Gradio app.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="user/space-name")
    ap.add_argument("--folder", default="hf_space")
    a = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set; nothing pushed", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.space, repo_type="space", space_sdk="gradio",
                    exist_ok=True)
    api.upload_folder(folder_path=a.folder, repo_id=a.space, repo_type="space",
                      commit_message="Deploy from CI")
    print(f"https://huggingface.co/spaces/{a.space}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
