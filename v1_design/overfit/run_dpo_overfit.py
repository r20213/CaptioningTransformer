#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required. Install with: pip install pyyaml") from exc

# Resolve imports for v1 model code.
THIS_DIR = Path(__file__).resolve().parent
V1_ROOT = THIS_DIR.parent
SRC_DIR = V1_ROOT / "src"
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

from src import CaptioningTransformerV1


def _load_env_files() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_paths = [
        THIS_DIR / ".env",
        V1_ROOT / ".env",
        V1_ROOT.parent / "preprocessing" / ".env",
        V1_ROOT.parent / "tokenizer" / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)


def _hash_split(text: str) -> str:
    digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit DPO on 100 preference examples with inference report")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the checkpoint from the pretrain/SFT stage to initialize policy and reference models",
    )
    parser.add_argument(
        "--encoded-dataset-id",
        type=str,
        default=os.environ.get("HF_DATASET_REPO_ID", "").strip(),
        help="Dataset repo id that contains patch_tokens/chosen/rejected/sample_id",
    )
    parser.add_argument(
        "--original-dataset-id",
        type=str,
        default=os.environ.get("HF_SOURCE_DATASET_ID", "").strip(),
        help="Original dataset used for image lookup",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=os.environ.get("HF_DATASET_SPLIT", "train").strip() or "train",
    )
    parser.add_argument(
        "--tokenizer-repo-id",
        type=str,
        default=os.environ.get("TOKENIZER_REPO_ID", "").strip(),
    )
    parser.add_argument("--train-examples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6, help="DPO typically requires a smaller learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL penalty coefficient")
    parser.add_argument("--max-text-len", type=int, default=96)
    parser.add_argument("--max-gen-len", type=int, default=64)
    parser.add_argument("--num-infer", type=int, default=12)
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Slight repetition penalty for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(THIS_DIR / "artifacts_dpo"),
    )
    parser.add_argument(
        "--original-parquet-urls-json",
        type=str,
        default=os.environ.get("HF_SOURCE_PARQUET_URLS_JSON", "").strip(),
        help="Optional JSON string containing parquet URLs for the original dataset.",
    )
    return parser.parse_args()


@dataclass
class DPOTrainExample:
    sample_id: str
    chosen_caption: str
    rejected_caption: str
    image_tokens: torch.Tensor  # [1024, 768]
    chosen_input_ids: List[int]
    chosen_label_ids: List[int]
    rejected_input_ids: List[int]
    rejected_label_ids: List[int]


class CaptionTokenizer:
    def __init__(self, repo_id: str, hf_token: str):
        from transformers import AutoTokenizer

        self._tok = None
        try:
            self._tok = AutoTokenizer.from_pretrained(repo_id, token=hf_token, use_fast=True)
        except (ValueError, ImportError):
            pass
        
        if self._tok is None:
            try:
                self._tok = AutoTokenizer.from_pretrained(repo_id, token=hf_token, use_fast=False)
            except (ValueError, ImportError):
                pass
        
        if self._tok is None:
            try:
                import sentencepiece as spm
                from huggingface_hub import hf_hub_download
                
                model_file = hf_hub_download(
                    repo_id=repo_id,
                    filename="spm_unigram.model",
                    token=hf_token,
                )
                self._processor = spm.SentencePieceProcessor(model_file=model_file)
                self._use_direct_spm = True
            except ImportError as exc:
                raise ImportError(
                    "Could not load tokenizer. Requires: sentencepiece, transformers."
                ) from exc
        else:
            self._use_direct_spm = False
        
        if self._use_direct_spm:
            self.pad_id = 0
            self.bos_id = 2
            self.eos_id = 3
        else:
            self.pad_id = self._tok.pad_token_id if self._tok.pad_token_id is not None else 0
            self.bos_id = self._tok.bos_token_id if self._tok.bos_token_id is not None else 2
            self.eos_id = self._tok.eos_token_id if self._tok.eos_token_id is not None else 3

    def encode_caption(self, text: str, max_text_len: int) -> tuple[List[int], List[int]]:
        if self._use_direct_spm:
            core = self._processor.encode(text)
        else:
            core = self._tok.encode(text, add_special_tokens=False)
        core = core[: max(1, max_text_len - 2)]
        full = [self.bos_id] + core + [self.eos_id]
        return full[:-1], full[1:]

    def decode_ids(self, ids: List[int]) -> str:
        trimmed: List[int] = []
        for token_id in ids:
            if token_id == self.eos_id:
                break
            if token_id in (self.pad_id, self.bos_id):
                continue
            trimmed.append(token_id)
        
        if self._use_direct_spm:
            return self._processor.decode(trimmed).strip()
        else:
            return self._tok.decode(trimmed, skip_special_tokens=True).strip()


def load_dpo_examples(
    encoded_dataset_id: str,
    split: str,
    train_examples: int,
    max_text_len: int,
    hf_token: str,
    tokenizer: CaptionTokenizer,
) -> List[DPOTrainExample]:
    ds = load_dataset(encoded_dataset_id, split=split, streaming=True, token=hf_token)

    items: List[DPOTrainExample] = []
    for row in ds:
        chosen = str(row.get("chosen", row.get("chosen_caption", ""))).strip()
        rejected = str(row.get("rejected", row.get("rejected_caption", ""))).strip()
        sample_id = str(row.get("sample_id", "")).strip()
        patch_flat = row.get("patch_tokens")

        if not chosen or not rejected or not sample_id or patch_flat is None:
            continue
        if _hash_split(chosen) != "train":
            continue

        patch_tensor = torch.tensor(patch_flat, dtype=torch.float32)
        if patch_tensor.numel() != 1024 * 768:
            continue
        patch_tensor = patch_tensor.view(1024, 768)

        chosen_in, chosen_lbl = tokenizer.encode_caption(chosen, max_text_len=max_text_len)
        rejected_in, rejected_lbl = tokenizer.encode_caption(rejected, max_text_len=max_text_len)

        items.append(
            DPOTrainExample(
                sample_id=sample_id,
                chosen_caption=chosen,
                rejected_caption=rejected,
                image_tokens=patch_tensor,
                chosen_input_ids=chosen_in,
                chosen_label_ids=chosen_lbl,
                rejected_input_ids=rejected_in,
                rejected_label_ids=rejected_lbl,
            )
        )
        if len(items) >= train_examples:
            break

    if len(items) == 0:
        raise RuntimeError(
            f"Found No data."
        )
    return items


def build_dpo_sub_batch(
    examples: List[DPOTrainExample],
    is_chosen: bool,
    pad_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch_size = len(examples)
    image_tokens = torch.stack([ex.image_tokens for ex in examples], dim=0).to(device)
    
    if is_chosen:
        max_len = max(len(ex.chosen_input_ids) for ex in examples)
    else:
        max_len = max(len(ex.rejected_input_ids) for ex in examples)

    text_input = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    text_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    text_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, ex in enumerate(examples):
        input_ids = ex.chosen_input_ids if is_chosen else ex.rejected_input_ids
        label_ids = ex.chosen_label_ids if is_chosen else ex.rejected_label_ids
        seq_len = len(input_ids)
        
        text_input[i, :seq_len] = torch.tensor(input_ids, dtype=torch.long, device=device)
        text_labels[i, :seq_len] = torch.tensor(label_ids, dtype=torch.long, device=device)
        text_mask[i, :seq_len] = 1

    image_mask = torch.ones((batch_size, image_tokens.shape[1]), dtype=torch.long, device=device)
    attn_mask = torch.cat([image_mask, text_mask], dim=1)

    return {
        "image_tokens": image_tokens,
        "text_input": text_input,
        "text_labels": text_labels,
        "attention_mask": attn_mask,
    }


def compute_sequence_logps(
    model: CaptioningTransformerV1,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = model(
        image_tokens=batch["image_tokens"],
        text_tokens=batch["text_input"],
        attention_mask=batch["attention_mask"],
    )
    img_seq = batch["image_tokens"].shape[1]
    text_logits = outputs.logits[:, img_seq:, :]
    
    per_token_logps = F.log_softmax(text_logits, dim=-1)
    labels = batch["text_labels"]
    loss_mask = labels != -100
    
    clamped_labels = labels.clone()
    clamped_labels[~loss_mask] = 0
    
    gathered_logps = torch.gather(per_token_logps, dim=-1, index=clamped_labels.unsqueeze(-1)).squeeze(-1)
    return (gathered_logps * loss_mask).sum(dim=-1)


def assert_all_finite(named_tensors: Dict[str, torch.Tensor]) -> None:
    for name, tensor in named_tensors.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite values detected in {name}")


def train_dpo_overfit(
    model: CaptioningTransformerV1,
    ref_model: CaptioningTransformerV1,
    examples: List[DPOTrainExample],
    steps: int,
    batch_size: int,
    lr: float,
    beta: float,
    pad_id: int,
    device: torch.device,
    output_dir: Path,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    loss_csv = output_dir / "dpo_loss_curve.csv"
    with loss_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "chosen_rewards", "rejected_rewards", "margin", "grad_norm"])

        model.train()
        ref_model.eval()

        for step in range(1, steps + 1):
            raw_batch = random.sample(examples, k=batch_size)
            
            chosen_batch = build_dpo_sub_batch(raw_batch, is_chosen=True, pad_id=pad_id, device=device)
            rejected_batch = build_dpo_sub_batch(raw_batch, is_chosen=False, pad_id=pad_id, device=device)

            optimizer.zero_grad(set_to_none=True)

            policy_chosen_logps = compute_sequence_logps(model, chosen_batch)
            policy_rejected_logps = compute_sequence_logps(model, rejected_batch)

            with torch.no_grad():
                ref_chosen_logps = compute_sequence_logps(ref_model, chosen_batch)
                ref_rejected_logps = compute_sequence_logps(ref_model, rejected_batch)

            policy_log_ratios = policy_chosen_logps - policy_rejected_logps
            ref_log_ratios = ref_chosen_logps - ref_rejected_logps
            logits = policy_log_ratios - ref_log_ratios

            loss = -F.logsigmoid(beta * logits).mean()

            assert_all_finite({
                "loss": loss,
                "policy_chosen_logps": policy_chosen_logps,
                "policy_rejected_logps": policy_rejected_logps,
            })

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError("Non-finite gradient norm detected")

            optimizer.step()

            with torch.no_grad():
                chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).mean().item()
                rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).mean().item()
                margin = chosen_rewards - rejected_rewards

                for param_name, param in model.named_parameters():
                    if not torch.isfinite(param).all():
                        raise FloatingPointError(f"Non-finite parameter values in {param_name}")

            writer.writerow([step, float(loss.item()), chosen_rewards, rejected_rewards, margin, float(grad_norm)])
            
            if step == 1 or step % 25 == 0 or step == steps:
                print(
                    f"step={step:04d} loss={float(loss.item()):.4f} "
                    f"rewards_c={chosen_rewards:.4f} rewards_r={rejected_rewards:.4f} "
                    f"margin={margin:.4f} grad={float(grad_norm):.2f}"
                )


@dataclass
class InferenceRow:
    sample_id: str
    chosen_caption: str
    rejected_caption: str
    generated_caption: str


def generate_caption(
    model: CaptioningTransformerV1,
    tokenizer: CaptionTokenizer,
    image_tokens: torch.Tensor,
    max_gen_len: int,
    device: torch.device,
    repetition_penalty: float = 1.05,
) -> str:
    model.eval()
    generated = [tokenizer.bos_id]
    image_batch = image_tokens.unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_gen_len):
            text_input = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            image_mask = torch.ones((1, image_batch.shape[1]), dtype=torch.long, device=device)
            text_mask = torch.ones((1, text_input.shape[1]), dtype=torch.long, device=device)
            attn_mask = torch.cat([image_mask, text_mask], dim=1)

            outputs = model(image_tokens=image_batch, text_tokens=text_input, attention_mask=attn_mask)
            next_logits = outputs.logits[0, image_batch.shape[1] + text_input.shape[1] - 1].clone()

            if repetition_penalty != 1.0 and generated:
                seen = torch.tensor(list(set(generated)), dtype=torch.long, device=device)
                next_logits[seen] = torch.where(
                    next_logits[seen] > 0,
                    next_logits[seen] / repetition_penalty,
                    next_logits[seen] * repetition_penalty,
                )

            next_id = int(torch.argmax(next_logits).item())
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                break

    return tokenizer.decode_ids(generated)


def _extract_urls_from_payload(payload: Any, split: str) -> List[str]:
    urls: List[str] = []
    if isinstance(payload, dict) and "parquet_files" in payload:
        entries = payload.get("parquet_files", [])
        if isinstance(entries, list):
            for item in entries:
                if not isinstance(item, dict):
                    continue
                item_split = str(item.get("split", "")).strip()
                item_url = str(item.get("url", "")).strip()
                if item_split == split and item_url:
                    urls.append(item_url)
        return urls

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            item_split = str(item.get("split", "")).strip()
            item_url = str(item.get("url", "")).strip()
            if item_split == split and item_url:
                urls.append(item_url)
        return urls

    if isinstance(payload, dict):
        direct = payload.get(split)
        if isinstance(direct, list):
            for u in direct:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
        for _, value in payload.items():
            if not isinstance(value, dict):
                continue
            split_urls = value.get(split)
            if isinstance(split_urls, list):
                for u in split_urls:
                    if isinstance(u, str) and u.strip():
                        urls.append(u.strip())
        return urls
    return urls


def _load_sql_templates() -> Dict[str, str]:
    yaml_path = THIS_DIR / "queries" / "duckdb_queries.yaml"
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError("duckdb query yaml must map keys to SQL templates")
    return {str(k): str(v) for k, v in data.items()}


def fetch_images_with_duckdb(
    sample_ids: List[str],
    dataset_id: str,
    split: str,
    hf_token: str,
    output_dir: Path,
) -> Path:
    wanted_set = set(sample_ids)
    rows: List[Dict[str, Any]] = []

    print(f"  Streaming {dataset_id} (split={split}) to fetch images by __key__...")
    stream = load_dataset(dataset_id, split=split, streaming=True, token=hf_token)
    for row in stream:
        key = str(row.get("__key__", "")).strip()
        if key not in wanted_set:
            continue
        jpg_obj = row.get("jpg")
        image_bytes: Optional[bytes] = None
        if isinstance(jpg_obj, bytes):
            image_bytes = jpg_obj
        elif isinstance(jpg_obj, dict):
            raw = jpg_obj.get("bytes")
            image_bytes = bytes(raw) if raw is not None else None
        rows.append({"sample_id": key, "image_bytes": image_bytes})
        wanted_set.discard(key)
        if not wanted_set:
            print(f"  All {len(sample_ids)} images found.")
            break

    if wanted_set:
        print(f"  Warning: {len(wanted_set)} sample_id(s) not found in stream: {list(wanted_set)[:5]}")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    meta_rows: List[Dict[str, str]] = []
    by_id = {row["sample_id"]: row for row in rows}

    for sample_id in sample_ids:
        row = by_id.get(sample_id)
        local_path = ""
        if row is not None:
            image_bytes = row.get("image_bytes")
            if image_bytes:
                image_path = image_dir / f"{sample_id}.jpg"
                try:
                    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    img.save(image_path)
                    local_path = str(image_path)
                except Exception:
                    local_path = ""

        meta_rows.append({
            "sample_id": sample_id,
            "local_image_path": local_path,
            "source_image_path": "",
            "source_caption": "",
        })

    image_meta_parquet = output_dir / "image_metadata.parquet"
    pq.write_table(
        pa.Table.from_pydict({
            "sample_id": [r["sample_id"] for r in meta_rows],
            "local_image_path": [r["local_image_path"] for r in meta_rows],
            "source_image_path": [r["source_image_path"] for r in meta_rows],
            "source_caption": [r["source_caption"] for r in meta_rows],
        }),
        image_meta_parquet,
    )
    return image_meta_parquet


def write_reports(
    predictions_parquet: Path,
    image_meta_parquet: Path,
    output_dir: Path,
) -> None:
    sql_templates = _load_sql_templates()
    con = duckdb.connect()

    join_sql = sql_templates["join_predictions_with_images"].format(
        predictions_parquet=str(predictions_parquet).replace("'", "''"),
        image_meta_parquet=str(image_meta_parquet).replace("'", "''"),
    )
    joined = con.execute(join_sql).fetchall()

    md_path = output_dir / "dpo_inference_report.md"
    html_path = output_dir / "dpo_inference_report.html"

    with md_path.open("w", encoding="utf-8") as md:
        md.write("# DPO Overfit Inference Report\n\n")
        md.write("| sample_id | image | generated_caption | chosen_caption |\n")
        md.write("|---|---|---|---|\n")
        for sample_id, generated, chosen, image_path, source_path, _ in joined:
            image_text = image_path if image_path else (source_path if source_path else "<missing>")
            md.write(
                f"| {sample_id} | {image_text} | {generated.replace('|', ' ')} | {chosen.replace('|', ' ')} |\n"
            )

    with html_path.open("w", encoding="utf-8") as hf:
        hf.write("""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>V1 DPO Overfit Inference Report</title>
<style>
body { font-family: Segoe UI, sans-serif; margin: 20px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
th { background: #f4f4f4; }
img { max-width: 240px; height: auto; border-radius: 4px; }
.small { color: #666; font-size: 12px; }
</style>
</head>
<body>
<h1>V1 DPO Overfit Inference Report</h1>
<table>
<tr><th>Sample</th><th>Image</th><th>Generated Caption</th><th>Chosen (Target) Caption</th></tr>
""")

        for sample_id, generated, chosen, image_path, source_path, _ in joined:
            if image_path and Path(image_path).exists():
                try:
                    rel_image = os.path.relpath(image_path, start=output_dir)
                    rel_image = rel_image.replace("\\", "/")
                    image_html = f"<img src='{html.escape(rel_image)}' alt='image'/>"
                except ValueError:
                    image_html = f"<div class='small'>{html.escape(source_path or 'image missing')}</div>"
            else:
                image_html = f"<div class='small'>{html.escape(source_path or 'image missing')}</div>"

            hf.write("<tr>")
            hf.write(f"<td>{html.escape(sample_id)}</td>")
            hf.write(f"<td>{image_html}</td>")
            hf.write(f"<td>{html.escape(generated)}</td>")
            hf.write(f"<td>{html.escape(chosen)}</td>")
            hf.write("</tr>\n")

        hf.write("</table>\n</body>\n</html>\n")


def main() -> None:
    _load_env_files()
    args = parse_args()

    if not args.encoded_dataset_id:
        raise ValueError("--encoded-dataset-id is required")
    if not args.original_dataset_id:
        raise ValueError("--original-dataset-id is required")
    if not args.tokenizer_repo_id:
        raise ValueError("--tokenizer-repo-id is required")
    if args.train_examples != 100:
        raise ValueError("This harness is fixed to exactly 100 preference examples.")
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("--steps and --batch-size must be > 0")

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise EnvironmentError("HF_TOKEN is required")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = CaptionTokenizer(repo_id=args.tokenizer_repo_id, hf_token=hf_token)

    print("Collecting 100 preference examples from encoded dataset...")
    dpo_data = load_dpo_examples(
        encoded_dataset_id=args.encoded_dataset_id,
        split=args.split,
        train_examples=args.train_examples,
        max_text_len=args.max_text_len,
        hf_token=hf_token,
        tokenizer=tokenizer,
    )

    print(f"Building Policy and Reference models from checkpoint: {args.checkpoint_path}")
    
    def load_base_architecture():
        return CaptioningTransformerV1(
            vocab_size=16000,
            embed_dim=384,
            n_layers=30,
            n_heads=12,
            n_kv_heads=4,
            use_gradient_checkpointing=False,
            qk_norm=True,
        )

    model = load_base_architecture()
    ref_model = load_base_architecture()

    # Load weights into both policy and reference instances
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    ref_model.load_state_dict(state_dict)

    model.to(device)
    ref_model.to(device)
    
    # Freeze the reference model explicitly
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print("Training DPO overfit run with mathematical logging...")
    train_dpo_overfit(
        model=model,
        ref_model=ref_model,
        examples=dpo_data,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        pad_id=tokenizer.pad_id,
        device=device,
        output_dir=out_dir,
    )

    print("Running post-DPO inference validation...")
    infer_n = min(args.num_infer, len(dpo_data))
    infer_rows: List[InferenceRow] = []
    for ex in dpo_data[:infer_n]:
        pred = generate_caption(
            model=model,
            tokenizer=tokenizer,
            image_tokens=ex.image_tokens,
            max_gen_len=args.max_gen_len,
            device=device,
            repetition_penalty=args.repetition_penalty,
        )
        infer_rows.append(
            InferenceRow(
                sample_id=ex.sample_id,
                chosen_caption=ex.chosen_caption,
                rejected_caption=ex.rejected_caption,
                generated_caption=pred,
            )
        )

    predictions_parquet = out_dir / "dpo_inference_predictions.parquet"
    pq.write_table(
        pa.table({
            "sample_id": [r.sample_id for r in infer_rows],
            "actual_caption": [r.chosen_caption for r in infer_rows],  # Map chosen as target column for standard query script
            "generated_caption": [r.generated_caption for r in infer_rows],
        }),
        predictions_parquet,
    )

    print("Fetching original images via DuckDB SQL query...")
    image_meta_parquet = fetch_images_with_duckdb(
        sample_ids=[r.sample_id for r in infer_rows],
        dataset_id=args.original_dataset_id,
        split=args.split,
        hf_token=hf_token,
        output_dir=out_dir,
    )

    print("Creating final joined report...")
    write_reports(
        predictions_parquet=predictions_parquet,
        image_meta_parquet=image_meta_parquet,
        output_dir=out_dir,
    )

    print("Done.")
    print(f"Open DPO report: {out_dir / 'dpo_inference_report.html'}")


if __name__ == "__main__":
    main()