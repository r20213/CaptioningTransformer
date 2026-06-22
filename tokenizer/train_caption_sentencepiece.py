#!/usr/bin/env python3
"""
Train a SentencePiece Unigram tokenizer for captioning.

Pipeline:
  1) Authenticate to Hugging Face Hub.
  2) Materialize dataset locally with streaming=False and text-column selection.
  3) Deterministically split train/val/test and train on train split.
  4) Train Google SentencePiece (unigram) and export .model/.vocab.
  5) Run sanity checks for special tokens and numeric behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, login
from huggingface_hub.errors import RepositoryNotFoundError
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

import sentencepiece as spm

# Load tokenizer-specific .env first, then shared preprocessing .env for common vars like HF_TOKEN.
try:
    from dotenv import load_dotenv

    script_dir = Path(__file__).parent
    load_dotenv(dotenv_path=script_dir / ".env", override=False)
    shared_env = script_dir.parent / "preprocessing" / ".env"
    if shared_env.exists():
        load_dotenv(dotenv_path=shared_env, override=False)
except ImportError:
    pass


SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[PAD]", "[UNK]", "[MASK]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentencePiece unigram tokenizer from captions")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=os.environ.get("TOKENIZER_DATASET_ID", "").strip(),
        help="HF dataset repo id (env: TOKENIZER_DATASET_ID)",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default=os.environ.get("TOKENIZER_SOURCE_SPLIT", "train").strip() or "train",
        help="Source split to load from dataset repo",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=os.environ.get("TOKENIZER_TEXT_COLUMN", "caption").strip() or "caption",
        help="Caption/text column name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("TOKENIZER_OUTPUT_DIR", "tokenizer").strip() or "tokenizer",
        help="Directory to write tokenizer artifacts",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=int(os.environ.get("TOKENIZER_VOCAB_SIZE", "16000").strip() or "16000"),
        help="SentencePiece vocabulary size",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=int(os.environ.get("TOKENIZER_NUM_THREADS", "8").strip() or "8"),
        help="SentencePiece training threads",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("TOKENIZER_SEED", "42").strip() or "42"),
        help="Random seed for deterministic split",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=int(os.environ.get("TOKENIZER_MAX_ROWS", "10000").strip() or "10000"),
        help="Rows to process from source split (0 = all rows)",
    )
    parser.add_argument(
        "--text-dataset-repo-id",
        type=str,
        default=(
            os.environ.get("TOKENIZER_TEXT_DATASET_REPO_ID", "").strip()
            or os.environ.get("HF_DATASET_REPO_ID", "").strip()
        ),
        help="Hub dataset repo to store/reuse text-only split dataset",
    )
    parser.add_argument(
        "--upload-shard-rows",
        type=int,
        default=int(os.environ.get("TOKENIZER_UPLOAD_SHARD_ROWS", "50000").strip() or "50000"),
        help="Rows per local parquet shard before upload",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_id:
        raise ValueError("Dataset id is required. Set TOKENIZER_DATASET_ID or pass --dataset-id.")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be > 0")
    if args.num_threads <= 0:
        raise ValueError("--num-threads must be > 0")
    if args.seed < 0:
        raise ValueError("--seed must be >= 0")
    if args.max_rows < 0:
        raise ValueError("--max-rows must be >= 0")
    if not args.text_dataset_repo_id:
        raise ValueError("--text-dataset-repo-id is required (or set TOKENIZER_TEXT_DATASET_REPO_ID)")
    if args.upload_shard_rows <= 0:
        raise ValueError("--upload-shard-rows must be > 0")


def authenticate_hub(hf_token: str) -> None:
    if not hf_token:
        raise ValueError("HF_TOKEN is required in environment/.env")
    print("Authenticating with Hugging Face Hub...")
    login(token=hf_token, add_to_git_credential=False)


def _hash_split(text: str) -> str:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def _flush_shard(base_dir: Path, split_name: str, shard_idx: int, text_column: str, rows: list[str]) -> None:
    if not rows:
        return
    split_dir = base_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_path = split_dir / f"part-{shard_idx:06d}.parquet"
    table = pa.Table.from_pydict({text_column: rows})
    pq.write_table(table, shard_path, compression="zstd")


def text_split_dataset_exists(repo_id: str, hf_token: str) -> bool:
    try:
        probe = load_dataset(repo_id, split="train", streaming=True, token=hf_token)
        iterator = iter(probe)
        first_row = next(iterator, None)
        return first_row is not None
    except Exception:
        return False


def build_and_push_text_split_dataset(args: argparse.Namespace, hf_token: str, output_dir: Path) -> None:
    print(
        "Building text-only split dataset from source stream "
        f"and pushing to Hub repo {args.text_dataset_repo_id}..."
    )

    stream = load_dataset(
        args.dataset_id,
        split=args.source_split,
        streaming=True,
        token=hf_token,
    ).select_columns([args.text_column])

    upload_root = output_dir / "text_splits_upload"
    if upload_root.exists():
        shutil.rmtree(upload_root)
    upload_root.mkdir(parents=True, exist_ok=True)

    buffers = {"train": [], "validation": [], "test": []}
    shard_counts = {"train": 0, "validation": 0, "test": 0}
    row_counts = {"train": 0, "validation": 0, "test": 0}

    processed = 0
    progress_total = args.max_rows if args.max_rows > 0 else None
    with tqdm(total=progress_total, desc="Streaming source rows") as pbar:
        for row in stream:
            text_raw = row.get(args.text_column, "")
            if text_raw is None:
                continue
            text = str(text_raw)
            if not text:
                continue

            split_name = _hash_split(text)
            buffers[split_name].append(text)
            row_counts[split_name] += 1
            processed += 1
            pbar.update(1)

            if len(buffers[split_name]) >= args.upload_shard_rows:
                _flush_shard(
                    base_dir=upload_root,
                    split_name=split_name,
                    shard_idx=shard_counts[split_name],
                    text_column=args.text_column,
                    rows=buffers[split_name],
                )
                shard_counts[split_name] += 1
                buffers[split_name] = []

            if args.max_rows > 0 and processed >= args.max_rows:
                break

    for split_name in ["train", "validation", "test"]:
        if buffers[split_name]:
            _flush_shard(
                base_dir=upload_root,
                split_name=split_name,
                shard_idx=shard_counts[split_name],
                text_column=args.text_column,
                rows=buffers[split_name],
            )
            shard_counts[split_name] += 1
            buffers[split_name] = []

    print(
        "Prepared split dataset rows: "
        f"train={row_counts['train']} validation={row_counts['validation']} test={row_counts['test']}"
    )

    api = HfApi(token=hf_token)
    try:
        api.repo_info(repo_id=args.text_dataset_repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=args.text_dataset_repo_id, repo_type="dataset", exist_ok=True)

    api.upload_folder(
        folder_path=str(upload_root),
        repo_id=args.text_dataset_repo_id,
        repo_type="dataset",
        path_in_repo="",
    )

    print(f"Uploaded text-only split dataset to {args.text_dataset_repo_id}")


def write_training_corpus(train_stream, text_column: str, corpus_path: Path) -> int:
    rows_written = 0
    with corpus_path.open("w", encoding="utf-8") as f:
        for row in tqdm(train_stream, desc="Writing train corpus"):
            text_raw = row.get(text_column, "")
            if text_raw is None:
                continue
            text = str(text_raw)
            if not text:
                continue
            # Keep exact text content; SentencePiece handles unicode/newlines.
            f.write(text)
            f.write("\n")
            rows_written += 1
    return rows_written


def build_numeric_symbols() -> list[str]:
    """Return first 100 natural numbers as user-defined symbols."""
    return [str(i) for i in range(1, 101)]


def train_sentencepiece(
    corpus_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    numeric_symbols: list[str],
) -> Path:
    model_prefix = output_dir / "spm_unigram"

    user_symbols = ["[MASK]"] + numeric_symbols
    if args.vocab_size <= 4 + len(user_symbols):
        raise ValueError(
            "vocab_size is too small for special/user symbols. "
            f"Need > {4 + len(user_symbols)}, got {args.vocab_size}."
        )

    # Keep numeric spans as intact as possible.
    # - split_digits=False avoids forced per-digit segmentation.
    # - split_by_number=False avoids explicit number boundary splitting.
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=args.vocab_size,
        num_threads=args.num_threads,
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="[PAD]",
        unk_piece="[UNK]",
        bos_piece="[BOS]",
        eos_piece="[EOS]",
        user_defined_symbols=user_symbols,
        split_by_number=False,
        split_digits=False,
        byte_fallback=False,
    )

    return model_prefix.with_suffix(".model")


def run_sanity_check(model_path: Path) -> None:
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    sample = "A caption with number 87 and 42 apples."

    pieces = processor.encode(sample, out_type=str)
    decoded = processor.decode(pieces)

    print("\nSanity check")
    print("sample:", sample)
    print("pieces:", pieces)
    print("decoded:", decoded)

    required = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
    for tok in required:
        if processor.piece_to_id(tok) < 0:
            raise RuntimeError(f"Missing special token in trained model: {tok}")

    expected_atomic_numbers = {"87", "42"}
    observed_atomic_numbers = {piece for piece in pieces if piece.isdigit()}
    if not expected_atomic_numbers.issubset(observed_atomic_numbers):
        raise RuntimeError(
            "Numeric chunking check failed: expected atomic numeric pieces for 42 and 87; "
            f"got pieces={pieces}"
        )

    if decoded != sample:
        raise RuntimeError("Round-trip check failed: decoded text does not match original")


def convert_to_fast_tokenizer(model_path: Path, output_dir: Path) -> Path:
    """
    Convert SentencePiece .model to a fast tokenizer with tokenizer.json.
    Uses the tokenizers library to create Rust-serialized fast artifacts.
    """
    import json

    try:
        from tokenizers import Tokenizer
        from tokenizers.models import SentencePiece
    except ImportError as exc:
        raise ImportError(
            "Fast tokenizer conversion requires 'tokenizers' library. "
            "Install with: pip install tokenizers"
        ) from exc

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    vocab_size = processor.vocab_size()

    # Create a Rust-based fast tokenizer with SentencePiece backend.
    model = SentencePiece(str(model_path))
    tokenizer = Tokenizer(model)

    # Configure special tokens.
    special_token_map = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "[MASK]": 4,
    }
    for token, token_id in special_token_map.items():
        tokenizer.add_special_tokens([token])

    # Save as tokenizer.json (standard fast tokenizer format).
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    # Create tokenizer_config.json for transformers.
    tokenizer_config = {
        "tokenizer_class": "AutoTokenizer",
        "model_type": "sentencepiece",
        "vocab_size": vocab_size,
        "language": "en",
    }
    config_path = output_dir / "tokenizer_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create special_tokens_map.json.
    special_tokens = {
        "bos_token": {"content": "[BOS]", "lstrip": False, "rstrip": False, "single_word": False},
        "eos_token": {"content": "[EOS]", "lstrip": False, "rstrip": False, "single_word": False},
        "unk_token": {"content": "[UNK]", "lstrip": False, "rstrip": False, "single_word": False},
        "pad_token": {"content": "[PAD]", "lstrip": False, "rstrip": False, "single_word": False},
        "mask_token": {"content": "[MASK]", "lstrip": False, "rstrip": False, "single_word": False},
    }
    special_map_path = output_dir / "special_tokens_map.json"
    with special_map_path.open("w", encoding="utf-8") as f:
        json.dump(special_tokens, f, indent=2)

    print(f"\nCreated fast tokenizer artifacts:")
    print(f"- {tokenizer_json_path} (Rust-serialized fast tokenizer)")
    print(f"- {config_path}")
    print(f"- {special_map_path}")
    print(f"Tokenizer ready for fast loading with AutoTokenizer(use_fast=True)")
    return config_path


def run_test_cases(model_path: Path) -> None:
    """Run a suite of test cases to validate tokenizer behavior."""
    processor = spm.SentencePieceProcessor(model_file=str(model_path))

    test_cases = [
        "A cat sat on the mat.",
        "The numbers 1 and 100 are boundaries.",
        "Mixed: 42 apples, 87 oranges, 999 bananas.",
        "Special tokens: [BOS] text [EOS]",
        "Unicode: café, naïve, résumé",
        "Numbers: 1 2 3 4 5 10 20 50 100 101 200",
        "Long caption: The quick brown fox jumps over the lazy dog repeatedly.",
    ]

    print("\nRunning test cases...")
    for i, test in enumerate(test_cases, 1):
        pieces = processor.encode(test, out_type=str)
        decoded = processor.decode(pieces)
        match = "✓" if decoded == test else "✗"
        print(f"{match} Test {i}: {test[:50]}...")
        print(f"   → Pieces ({len(pieces)}): {pieces[:10]}{'...' if len(pieces) > 10 else ''}")
        if decoded != test:
            print(f"   ⚠ Decode mismatch: {decoded}")


def push_tokenizer_to_hub(model_path: Path, output_dir: Path, repo_id: str, hf_token: str) -> None:
    """Push tokenizer (model + configs) to Hugging Face Hub for fast loading."""
    api = HfApi(token=hf_token)

    # Ensure repo exists
    try:
        api.repo_info(repo_id, token=hf_token)
        print(f"Using existing tokenizer repo: {repo_id}")
    except RepositoryNotFoundError:
        print(f"Creating tokenizer repo: {repo_id}")
        api.create_repo(repo_id, private=False, token=hf_token)

    # Upload tokenizer files: SentencePiece model + transformers configs + fast tokenizer.json
    files_to_upload = [
        model_path,  # spm_unigram.model (SentencePiece binary)
        model_path.with_suffix(".vocab"),  # spm_unigram.vocab (vocabulary)
        output_dir / "tokenizer.json",  # Fast tokenizer (Rust-serialized)
        output_dir / "tokenizer_config.json",  # HF tokenizer config
        output_dir / "special_tokens_map.json",  # HF special tokens
    ]

    print(f"Uploading tokenizer files to {repo_id}...")
    for file_path in files_to_upload:
        if file_path.exists():
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=repo_id,
                token=hf_token,
            )
            print(f"  ✓ Uploaded {file_path.name}")

    print(f"✓ Tokenizer pushed to {repo_id} (ready for AutoTokenizer fast loading)")


def main() -> None:
    args = parse_args()
    validate_args(args)

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    authenticate_hub(hf_token)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if text_split_dataset_exists(args.text_dataset_repo_id, hf_token):
        print(f"Using existing text-only split dataset from Hub: {args.text_dataset_repo_id}")
    else:
        build_and_push_text_split_dataset(args, hf_token, output_dir)

    train_stream = load_dataset(
        args.text_dataset_repo_id,
        split="train",
        streaming=True,
        token=hf_token,
    ).select_columns([args.text_column])

    corpus_path = output_dir / "spm_train_corpus.txt"
    rows_written = write_training_corpus(train_stream, args.text_column, corpus_path)
    print(f"Training corpus rows written: {rows_written}")
    numeric_symbols = build_numeric_symbols()
    print(f"Registered numeric user symbols: {len(numeric_symbols)} (1-100)")

    print("Training SentencePiece unigram model...")
    model_path = train_sentencepiece(corpus_path, output_dir, args, numeric_symbols)
    vocab_path = model_path.with_suffix(".vocab")

    print("\nSaved tokenizer files:")
    print(f"- {model_path}")
    print(f"- {vocab_path}")

    run_sanity_check(model_path)
    print("\nSentencePiece training complete.")

    # Convert to fast tokenizer
    convert_to_fast_tokenizer(model_path, output_dir)

    # Run test cases
    run_test_cases(model_path)

    # Push to Hub
    tokenizer_repo = "LastTransformer/BLIP3o-SentencePiece-Unigram-Tokenizer"
    push_tokenizer_to_hub(model_path, output_dir, tokenizer_repo, hf_token)

    print("\n✓ All steps complete: trained, tested, and pushed tokenizer to Hub.")


if __name__ == "__main__":
    main()
