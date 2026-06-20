#!/usr/bin/env python3
"""
Inspect a few rows from the uploaded Hugging Face dataset.

Usage:
    python preprocessing/utils/inspect_uploaded_dataset.py
    python preprocessing/utils/inspect_uploaded_dataset.py --rows 5
    python preprocessing/utils/inspect_uploaded_dataset.py --split train

Reads configuration from preprocessing/utils/.env:
    HF_TOKEN
    HF_DATASET_REPO_ID
    HF_DATASET_SPLIT
    INSPECT_NUM_ROWS
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
except ImportError:
    pass

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    default_repo_id = os.environ.get("HF_DATASET_REPO_ID", "").strip()
    default_split = os.environ.get("HF_DATASET_SPLIT", "train").strip() or "train"
    default_rows = int(os.environ.get("INSPECT_NUM_ROWS", "3").strip() or "3")

    parser = argparse.ArgumentParser(description="Inspect a few rows from an uploaded HF dataset")
    parser.add_argument("--repo-id", type=str, default=default_repo_id, help="HF dataset repo id")
    parser.add_argument("--split", type=str, default=default_split, help="Dataset split to inspect")
    parser.add_argument("--rows", type=int, default=default_rows, help="Number of rows to print")
    return parser.parse_args()


def shorten(text: str, limit: int = 160) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def describe_patch_tokens(value: Any) -> str:
    if value is None:
        return "missing"

    if isinstance(value, list):
        top_len = len(value)
        if top_len == 0:
            return "empty list"
        first = value[0]
        if isinstance(first, list):
            inner_len = len(first)
            return f"nested list [{top_len}, {inner_len}]"
        preview = ", ".join(f"{float(x):.4f}" for x in value[:5])
        return f"flat list len={top_len}, head=[{preview}]"

    return f"type={type(value).__name__}"


def iter_rows(repo_id: str, split: str, token: str, rows: int) -> Iterable[dict[str, Any]]:
    ds = load_dataset(repo_id, split=split, streaming=True, token=token)
    for idx, row in enumerate(ds):
        if idx >= rows:
            break
        yield row


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "").strip()

    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable is required")
    if not args.repo_id:
        raise ValueError("HF_DATASET_REPO_ID or --repo-id is required")
    if args.rows <= 0:
        raise ValueError("--rows must be >= 1")

    print(f"Inspecting dataset: {args.repo_id}")
    print(f"Split: {args.split}")
    print(f"Rows requested: {args.rows}")
    print()

    found = 0
    for found, row in enumerate(iter_rows(args.repo_id, args.split, hf_token, args.rows), start=1):
        caption = str(row.get("caption", ""))
        sample_id = str(row.get("sample_id", ""))
        patch_tokens = row.get("patch_tokens")

        print(f"Row {found}")
        print(f"  sample_id: {sample_id or '<missing>'}")
        print(f"  caption: {shorten(caption)}")
        print(f"  patch_tokens: {describe_patch_tokens(patch_tokens)}")
        print()

    if found == 0:
        print("No rows were returned. Check repo id, split, or whether upload has completed.")


if __name__ == "__main__":
    main()
