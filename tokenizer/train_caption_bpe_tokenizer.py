#!/usr/bin/env python3
"""
Train a fast BPE tokenizer for image captioning from a streaming Hugging Face dataset.

Features:
  - Streaming-only, no full dataset materialization
  - Selective column loading to skip heavy vision/embedding data
  - Deterministic shuffling with seed-based buffered streaming
  - Numeric-aware pre-tokenization with regex `\\d+`
  - 16,000 vocabulary cap with required special tokens
  - Memory-mapped iterator for low-overhead training
  - Exports tokenizer.json, vocab.json, merges.txt into tokenizer/
  - Sanity check for round-trip decode and number token integrity
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator

# Set Rust parallelism globally BEFORE importing tokenizers library.
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

from datasets import load_dataset
from tokenizers import Regex, Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[PAD]", "[UNK]", "[MASK]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 16k BPE tokenizer from streaming captions")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=os.environ.get("TOKENIZER_DATASET_ID", "").strip(),
        help="HF dataset repo id to stream (env: TOKENIZER_DATASET_ID)",
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
        help="BPE vocabulary size limit",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=int(os.environ.get("TOKENIZER_NUM_CPUS", "4").strip() or "4"),
        help="CPU threads for tokenizer training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("TOKENIZER_BATCH_SIZE", "5000").strip() or "5000"),
        help="Batch size for training iterator",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("TOKENIZER_SEED", "42").strip() or "42"),
        help="Random seed for deterministic train/val/test split",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_id:
        raise ValueError("Dataset id is required. Set TOKENIZER_DATASET_ID or pass --dataset-id.")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be > 0")
    if args.num_cpus <= 0:
        raise ValueError("--num-cpus must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.seed < 0:
        raise ValueError("--seed must be >= 0")





def train_caption_batch_iterator(
    stream_dataset,
    text_column: str,
    batch_size: int,
) -> Iterator[list[str]]:
    """
    Yield batches of text strings from the streaming dataset iterator.
    This minimizes Python-Rust context switches and saturates the Rust backend.
    The stream is already shuffled and selected; we just batch it.
    """
    batch = []
    for row in stream_dataset:
        # Direct column access on streaming rows
        text_raw = row.get(text_column) if isinstance(row, dict) else getattr(row, text_column, None)
        if text_raw is not None:
            text = str(text_raw).strip() if str(text_raw).strip() else None
            if text:
                batch.append(text)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(pattern=Regex(r"\d+"), behavior="isolated"),
            # ByteLevel + ByteLevel decoder preserves exact text bytes on decode.
            ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def configure_special_token_processing(tokenizer: Tokenizer) -> None:
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    if bos_id is None or eos_id is None:
        raise RuntimeError("[BOS]/[EOS] missing from trained vocabulary")

    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )


def run_sanity_check(tokenizer: Tokenizer) -> None:
    sample = "A caption with number 987\nand 42 apples.  Keep\tspacing."

    pretokenized_chunks = [piece for piece, _ in tokenizer.pre_tokenizer.pre_tokenize_str(sample)]
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)

    print("\nSanity check")
    print("sample:", sample)
    print("pretokenized chunks:", pretokenized_chunks)
    print("tokens:", encoded.tokens)
    print("decoded:", decoded)

    if "987" not in pretokenized_chunks or "42" not in pretokenized_chunks:
        raise RuntimeError("Numeric chunking check failed: expected pre-tokenizer chunks '987' and '42'")
    if not encoded.tokens or encoded.tokens[0] != "[BOS]" or encoded.tokens[-1] != "[EOS]":
        raise RuntimeError("Special token check failed: expected [BOS] ... [EOS]")
    if decoded != sample:
        raise RuntimeError("Round-trip check failed: decoded text does not match original exactly")


def main() -> None:
    args = parse_args()
    validate_args(args)

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise ValueError("HF_TOKEN is required in environment/.env")

    # Set thread count for Rust parallelism (after validation).
    os.environ["RAYON_NUM_THREADS"] = str(args.num_cpus)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset with streaming=True and select only the text column.
    # This avoids downloading heavy vision tokens, embeddings, or other metadata.
    print(f"Loading dataset {args.dataset_id} (streaming, text-only)...")
    source_dataset = load_dataset(
        args.dataset_id,
        split=args.source_split,
        streaming=True,
        token=hf_token,
    ).select_columns([args.text_column])

    # Apply deterministic shuffle with seed-based buffering.
    # No materialization; streaming uses an in-memory buffer_size for shuffling.
    shuffled_dataset = source_dataset.shuffle(seed=args.seed, buffer_size=100)

    # For streaming, we train on the entire shuffled stream.
    # In production, you could add .take(N) if you want to limit rows,
    # but we train on the full stream for maximum vocabulary coverage.
    train_stream = shuffled_dataset

    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # Create batch iterator for Rust backend saturation.
    iterator = train_caption_batch_iterator(
        stream_dataset=train_stream,
        text_column=args.text_column,
        batch_size=args.batch_size,
    )

    print(f"Training tokenizer from dataset={args.dataset_id} split={args.source_split}")
    print(f"Streaming with deterministic shuffle (seed={args.seed}, buffer_size=100)")
    print(f"Batch size: {args.batch_size}")
    print(f"Configured CPU threads: {args.num_cpus}")
    print(f"Trainer progress bar enabled: True")
    print("Starting training (streaming mode, no length pre-computation)...")
    # Note: length parameter omitted for IterableDataset; trainer uses internal progress.
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    configure_special_token_processing(tokenizer)

    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    tokenizer.model.save(str(output_dir))

    print("\nSaved tokenizer files:")
    print(f"- {tokenizer_json}")
    print(f"- {output_dir / 'vocab.json'}")
    print(f"- {output_dir / 'merges.txt'}")

    run_sanity_check(tokenizer)
    print("\nTokenizer training complete.")


if __name__ == "__main__":
    main()
