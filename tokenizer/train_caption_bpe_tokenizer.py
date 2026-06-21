#!/usr/bin/env python3
"""
Train a fast BPE tokenizer for image captioning from a streaming Hugging Face dataset.

Features:
  - Deterministic virtual split: train=90%, validation=5%, test=5%
  - Trains only on train partition
  - Numeric-aware pre-tokenization with regex `\\d+`
  - 16,000 vocabulary cap with required special tokens
  - Streaming iterator to stay memory-safe on constrained machines
  - Exports tokenizer.json, vocab.json, merges.txt into tokenizer/
  - Sanity check for round-trip decode and number token integrity
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator

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
        default=os.environ.get("TOKENIZER_TEXT_COLUMN", "txt").strip() or "txt",
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
    train_dataset,
    text_column: str,
    batch_size: int,
) -> Iterator[list[str]]:
    """
    Yield batches of text strings from the training dataset.
    This minimizes Python-Rust context switches and saturates the Rust backend.
    """
    batch = []
    for row in train_dataset:
        text_raw = row.get(text_column, "")
        if text_raw is not None:
            text = str(text_raw)
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
    # Set Rust parallelism globally before any tokenizers imports/usage.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = ""

    args = parse_args()
    validate_args(args)

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise ValueError("HF_TOKEN is required in environment/.env")

    # Set thread count after validation.
    os.environ["RAYON_NUM_THREADS"] = str(args.num_cpus)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset and perform upfront deterministic split.
    print(f"Loading dataset {args.dataset_id}...")
    source_dataset = load_dataset(
        args.dataset_id,
        split=args.source_split,
        token=hf_token,
    )

    # Upfront split: test=10%, then split remainder into train=90% val=10% of remainder.
    test_split = source_dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_val_split = test_split["train"].train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    test_dataset = test_split["test"]

    print(f"Dataset split: train={len(train_dataset)} val={len(val_dataset)} test={len(test_dataset)}")

    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    # Create batch iterator for Rust backend saturation.
    iterator = train_caption_batch_iterator(
        train_dataset=train_dataset,
        text_column=args.text_column,
        batch_size=args.batch_size,
    )

    print(f"Training tokenizer from dataset={args.dataset_id} split={args.source_split}")
    print(f"Upfront split ratios: train=81% validation=9% test=10% (deterministic seed={args.seed})")
    print(f"Batch size: {args.batch_size}")
    print(f"Configured CPU threads: {args.num_cpus}")
    print(f"Trainer progress bar enabled: True")
    tokenizer.train_from_iterator(iterator, trainer=trainer, length=len(train_dataset))

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
