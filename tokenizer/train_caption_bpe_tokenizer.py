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
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, Whitespace
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
        help="Source split to stream from dataset repo",
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
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.dataset_id:
        raise ValueError("Dataset id is required. Set TOKENIZER_DATASET_ID or pass --dataset-id.")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be > 0")
    if args.num_cpus <= 0:
        raise ValueError("--num-cpus must be > 0")


def split_bucket(example: Dict[str, object]) -> str:
    key = str(example.get("__key__") or example.get("id") or example.get("txt") or "")
    digest = hashlib.blake2b(key.encode("utf-8", errors="ignore"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def train_caption_iterator(
    dataset_id: str,
    source_split: str,
    text_column: str,
    hf_token: str,
    counters: Dict[str, int],
) -> Iterator[str]:
    stream: Iterable[Dict[str, object]] = load_dataset(
        dataset_id,
        split=source_split,
        streaming=True,
        token=hf_token,
    )

    for row in stream:
        partition = split_bucket(row)
        counters[partition] += 1

        if partition != "train":
            continue

        text = str(row.get(text_column, "")).strip()
        if text:
            yield text


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Sequence(
        [
            Split(pattern=Regex(r"\d+"), behavior="isolated"),
            Whitespace(),
        ]
    )
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
    sample = "A caption with number 987 and 42 apples."
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)

    print("\nSanity check")
    print("sample:", sample)
    print("tokens:", encoded.tokens)
    print("decoded:", decoded)

    if "987" not in encoded.tokens or "42" not in encoded.tokens:
        raise RuntimeError("Numeric chunking check failed: expected '987' and '42' as standalone tokens")
    if not encoded.tokens or encoded.tokens[0] != "[BOS]" or encoded.tokens[-1] != "[EOS]":
        raise RuntimeError("Special token check failed: expected [BOS] ... [EOS]")


def main() -> None:
    args = parse_args()
    validate_args(args)

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise ValueError("HF_TOKEN is required in environment/.env")

    # Let tokenizers/rayon use all available constrained vCPUs.
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(args.num_cpus)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer()
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    counters = {"train": 0, "validation": 0, "test": 0}
    iterator = train_caption_iterator(
        dataset_id=args.dataset_id,
        source_split=args.source_split,
        text_column=args.text_column,
        hf_token=hf_token,
        counters=counters,
    )

    print(f"Training tokenizer from dataset={args.dataset_id} split={args.source_split}")
    print("Virtual split ratios: train=90% validation=5% test=5% (deterministic hash partition)")
    print(f"Configured CPU threads: {args.num_cpus}")
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    configure_special_token_processing(tokenizer)

    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    tokenizer.model.save(str(output_dir))

    print("\nSaved tokenizer files:")
    print(f"- {tokenizer_json}")
    print(f"- {output_dir / 'vocab.json'}")
    print(f"- {output_dir / 'merges.txt'}")
    print("Observed partition counts while streaming source split:", counters)

    run_sanity_check(tokenizer)
    print("\nTokenizer training complete.")


if __name__ == "__main__":
    main()
