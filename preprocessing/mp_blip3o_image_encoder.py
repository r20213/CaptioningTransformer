#!/usr/bin/env python3
"""
Multiprocessing image encoding pipeline for BLIP3o long-caption pretraining data.

Launch example (2 GPUs):
    python mp_blip3o_image_encoder.py \
            --num-procs 2 \
            --model-id google/tipsv2-b14 \
            --dataset-id BLIP3o/BLIP3o-Pretrain-Long-Caption \
            --batch-size 64

Requirements:
  - torch, torchvision, transformers, datasets, pyarrow, pillow, huggingface_hub
    - 2x CUDA GPUs available (for --num-procs 2)
  - HF_TOKEN set in environment

Environment variables:
    - HF_TOKEN: required for dataset/model/hub operations
    - MODEL_ID: optional default for --model-id
    - MODEL_REVISION: optional pinned revision/commit for --model-revision
    - HF_DATASET_ID: optional default for --dataset-id
    - HF_DATASET_SPLIT: optional default for --split
    - BATCH_SIZE: optional default for --batch-size
    - NUM_WORKERS: optional default for --num-workers
    - NUM_PROCS: optional default for --num-procs
    - OUTPUT_DIR: optional default for --output-dir
    - ENCODE_MAX_ROWS: optional exact global number of rows to process across all ranks
    - HF_DATASET_REPO_ID: optional dataset repo id for periodic uploads (e.g. user/repo)
    - HF_UPLOAD_EVERY_ROWS: number of rows per rank between Hub uploads (0 = disable)
    - HF_UPLOAD_PREFIX: optional folder prefix in dataset repo (default: parquet)
    - HF_CREATE_REPO_IF_MISSING: optional bool, auto-create dataset repo when uploads are enabled (default: true)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Load .env from the same directory as this script before anything else reads os.environ.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell-exported vars

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoModel

try:
    from huggingface_hub import HfApi as HuggingFaceHubAPI
except Exception:
    HuggingFaceHubAPI = None

try:
    from huggingface_hub.errors import RepositoryNotFoundError
except Exception:
    RepositoryNotFoundError = Exception

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiprocessing image encoder for BLIP3o streaming dataset")
    default_model_id = os.environ.get("MODEL_ID", "google/tipsv2-b14").strip() or "google/tipsv2-b14"
    default_model_revision = os.environ.get("MODEL_REVISION", "").strip()
    default_dataset_id = (
        os.environ.get("HF_DATASET_ID", "BLIP3o/BLIP3o-Pretrain-Long-Caption").strip()
        or "BLIP3o/BLIP3o-Pretrain-Long-Caption"
    )
    default_split = os.environ.get("HF_DATASET_SPLIT", "train").strip() or "train"
    default_batch_size = int(os.environ.get("BATCH_SIZE", "64").strip() or "64")
    default_num_workers = int(os.environ.get("NUM_WORKERS", "0").strip() or "0")
    default_num_procs = int(os.environ.get("NUM_PROCS", "2").strip() or "2")
    default_output_dir = os.environ.get("OUTPUT_DIR", "").strip()

    parser.add_argument("--model-id", type=str, default=default_model_id)
    parser.add_argument(
        "--model-revision",
        type=str,
        default=default_model_revision,
        help="Optional pinned HF model revision/commit hash",
    )
    parser.add_argument("--dataset-id", type=str, default=default_dataset_id)
    parser.add_argument("--split", type=str, default=default_split)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=default_batch_size,
        help="64 is typically safe on T4 for this setup",
    )
    parser.add_argument("--num-procs", type=int, default=default_num_procs, help="Number of GPU worker processes")
    parser.add_argument("--num-workers", type=int, default=default_num_workers, help="Use 0 for streaming stability")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Legacy fallback row limit. Prefer ENCODE_MAX_ROWS for exact global row control.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="Optional output directory; default uses temp dir",
    )
    return parser.parse_args()


def decode_image(raw_image: Any) -> Image.Image:
    if isinstance(raw_image, Image.Image):
        return raw_image.convert("RGB")
    if isinstance(raw_image, dict):
        if raw_image.get("bytes") is not None:
            return Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
        if raw_image.get("path"):
            return Image.open(raw_image["path"]).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(raw_image)}")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((448, 448), antialias=True),
            transforms.ToTensor(),
        ]
    )


def collate_fn(batch: List[Dict[str, Any]], image_tf: transforms.Compose) -> Dict[str, Any]:
    images: List[torch.Tensor] = []
    captions: List[str] = []
    sample_ids: List[str] = []

    for ex in batch:
        pil = decode_image(ex["jpg"])
        img = image_tf(pil)
        images.append(img)
        captions.append(str(ex.get("txt", "")))
        sample_ids.append(str(ex.get("__key__", "")))

    pixel_values = torch.stack(images, dim=0)
    return {
        "pixel_values": pixel_values,
        "caption": captions,
        "sample_id": sample_ids,
    }


def infer_patch_tokens(model: torch.nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    # Keep model inference strictly in fp16 on T4.
    pixel_values = pixel_values.to(dtype=torch.float16)

    if hasattr(model, "encode_image") and callable(getattr(model, "encode_image")):
        out = model.encode_image(pixel_values)
    elif hasattr(model, "get_image_features") and callable(getattr(model, "get_image_features")):
        out = model.get_image_features(pixel_values=pixel_values)
    else:
        out = model(pixel_values=pixel_values)

    if isinstance(out, torch.Tensor):
        patch_tokens = out
    elif isinstance(out, dict):
        patch_tokens = out.get("patch_tokens") or out.get("last_hidden_state")
        if patch_tokens is None:
            raise RuntimeError("Model output dict did not contain patch_tokens or last_hidden_state")
    else:
        patch_tokens = getattr(out, "patch_tokens", None)
        if patch_tokens is None:
            patch_tokens = getattr(out, "last_hidden_state", None)
        if patch_tokens is None and hasattr(out, "__iter__"):
            out_list = list(out)
            if not out_list:
                raise RuntimeError("Model output iterable was empty")
            patch_tokens = out_list[0]
        if patch_tokens is None:
            raise RuntimeError("Unable to extract patch tokens from model output")

    patch_tokens = patch_tokens.to(dtype=torch.float16)

    if patch_tokens.ndim != 3:
        raise RuntimeError(f"Expected 3D patch tokens [B, 1024, 768], got shape={tuple(patch_tokens.shape)}")

    if patch_tokens.shape[1:] != (1024, 768):
        raise RuntimeError(
            f"Expected patch token shape [B, 1024, 768], got shape={tuple(patch_tokens.shape)}"
        )

    return patch_tokens


def write_parquet_chunk(
    output_dir: str,
    local_rank: int,
    chunk_idx: int,
    patch_tokens: torch.Tensor,
    captions: List[str],
    sample_ids: List[str],
) -> str:
    patch_tokens_cpu = patch_tokens.detach().cpu().contiguous()

    rows_patch: List[List[float]] = []
    for i in range(patch_tokens_cpu.shape[0]):
        # Flatten [1024, 768] -> [786432] serialized float list per row.
        rows_patch.append(patch_tokens_cpu[i].view(-1).tolist())

    table = pa.Table.from_pydict(
        {
            "patch_tokens": rows_patch,
            "caption": captions,
            "sample_id": sample_ids,
        }
    )

    out_path = os.path.join(output_dir, f"rank{local_rank:02d}_chunk{chunk_idx:07d}.parquet")
    pq.write_table(table, out_path, compression="zstd")
    return out_path


def build_output_dir(output_dir_arg: str) -> str:
    if output_dir_arg:
        os.makedirs(output_dir_arg, exist_ok=True)
        return output_dir_arg
    else:
        return tempfile.mkdtemp(prefix="blip3o_patch_tokens_")


def resolve_global_row_limit(args: argparse.Namespace) -> int:
    env_limit = os.environ.get("ENCODE_MAX_ROWS", "").strip()
    if env_limit:
        try:
            parsed = int(env_limit)
        except ValueError as exc:
            raise ValueError("ENCODE_MAX_ROWS must be an integer") from exc
        if parsed < 0:
            raise ValueError("ENCODE_MAX_ROWS must be >= 0")
        return parsed

    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    return args.max_samples


def compute_rank_target(global_limit: int, world_size: int, local_rank: int) -> int:
    if global_limit <= 0:
        return 0
    base = global_limit // world_size
    remainder = global_limit % world_size
    return base + (1 if local_rank < remainder else 0)


def resolve_hub_upload_config() -> Tuple[str, int, str]:
    repo_id = os.environ.get("HF_DATASET_REPO_ID", "").strip()
    prefix = os.environ.get("HF_UPLOAD_PREFIX", "parquet").strip() or "parquet"
    every_raw = os.environ.get("HF_UPLOAD_EVERY_ROWS", "0").strip() or "0"
    try:
        every_rows = int(every_raw)
    except ValueError as exc:
        raise ValueError("HF_UPLOAD_EVERY_ROWS must be an integer") from exc
    if every_rows < 0:
        raise ValueError("HF_UPLOAD_EVERY_ROWS must be >= 0")
    return repo_id, every_rows, prefix


def should_create_repo_if_missing() -> bool:
    return os.environ.get("HF_CREATE_REPO_IF_MISSING", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def ensure_dataset_repo_exists(hf_token: str, repo_id: str) -> None:
    if not repo_id:
        return
    if HuggingFaceHubAPI is None:
        raise ImportError("huggingface_hub is required for periodic uploads. Install it or disable uploads.")

    api = HuggingFaceHubAPI(token=hf_token)
    create_if_missing = should_create_repo_if_missing()
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        return
    except RepositoryNotFoundError:
        if not create_if_missing:
            raise RuntimeError(
                f"Dataset repo '{repo_id}' does not exist and HF_CREATE_REPO_IF_MISSING is disabled"
            )
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to validate dataset repo '{repo_id}': {exc}") from exc


def should_delete_local_after_upload() -> bool:
    return os.environ.get("HF_DELETE_LOCAL_AFTER_UPLOAD", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def flush_periodic_uploads(
    api: Any,
    repo_id: str,
    prefix: str,
    local_paths: List[str],
) -> None:
    delete_local = should_delete_local_after_upload()
    for local_file in local_paths:
        path_in_repo = f"{prefix}/{os.path.basename(local_file)}"
        try:
            api.upload_file(
                path_or_fileobj=local_file,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Dataset repo '{repo_id}' not found during upload. "
                "Create it first or enable HF_CREATE_REPO_IF_MISSING=1"
            ) from exc
        if delete_local:
            try:
                os.remove(local_file)
            except OSError:
                pass


def compile_model_or_raise(model: torch.nn.Module, rank: int) -> torch.nn.Module:
    mode = "max-autotune"

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build")

    try:
        compiled = torch.compile(model, mode=mode)
        if rank == 0:
            print(f"torch.compile enabled with mode='{mode}'")
        return compiled
    except Exception as exc:
        raise RuntimeError(f"torch.compile failed: {exc}") from exc


def run_worker(
    local_rank: int,
    args: argparse.Namespace,
    output_dir: str,
    global_row_limit: int,
    upload_repo_id: str,
    upload_every_chunks: int,
    upload_prefix: str,
    result_queue: Any,
) -> None:
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable is required")

    world_size = args.num_procs
    rank_target = compute_rank_target(global_row_limit, world_size, local_rank)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    from_pretrained_kwargs = {
        "trust_remote_code": True,
        "token": hf_token,
    }
    if args.model_revision:
        from_pretrained_kwargs["revision"] = args.model_revision

    # Prefer `dtype` to avoid deprecation warnings; fall back for older versions.
    try:
        model = AutoModel.from_pretrained(
            args.model_id,
            dtype=torch.float16,
            **from_pretrained_kwargs,
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            **from_pretrained_kwargs,
        )

    model.to(device=device, dtype=torch.float16)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model = compile_model_or_raise(model=model, rank=local_rank)

    # Warm-up pays compilation cost up front.
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16):
        warmup_pixels = torch.randn(1, 3, 448, 448, device=device, dtype=torch.float16)
        _ = infer_patch_tokens(model, warmup_pixels)

    image_tf = build_transform()

    ds = load_dataset(
        args.dataset_id,
        split=args.split,
        streaming=True,
        token=hf_token,
    )

    shard_count = world_size if world_size > 0 else 1
    ds = ds.shard(num_shards=shard_count, index=local_rank)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, image_tf),
        drop_last=False,
    )

    hub_api = None
    pending_uploads: List[str] = []

    if upload_every_chunks > 0:
        if not upload_repo_id:
            raise ValueError(
                "HF_UPLOAD_EVERY_CHUNKS is set, but HF_DATASET_REPO_ID is missing. "
                "Set HF_DATASET_REPO_ID like 'username/dataset-repo'."
            )
        if HuggingFaceHubAPI is None:
            raise ImportError(
                "huggingface_hub is required for periodic uploads. Install it or disable uploads."
            )
        hub_api = HuggingFaceHubAPI(token=hf_token)

    processed = 0
    chunk_idx = 0
    progress_bar = None

    if tqdm is not None:
        progress_bar = tqdm(
            total=(rank_target if rank_target > 0 else None),
            desc=f"rank{local_rank} rows",
            unit="rows",
            dynamic_ncols=True,
            position=local_rank,
            leave=True,
        )

    with torch.inference_mode():
        for batch in loader:
            if rank_target > 0 and processed >= rank_target:
                break

            captions = batch["caption"]
            sample_ids = batch["sample_id"]
            pixel_values = batch["pixel_values"]

            if rank_target > 0:
                remaining = rank_target - processed
                if remaining <= 0:
                    break
                if len(captions) > remaining:
                    captions = captions[:remaining]
                    sample_ids = sample_ids[:remaining]
                    pixel_values = pixel_values[:remaining]

            pixel_values = pixel_values.to(device=device, non_blocking=True)

            autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16)
            if device.type != "cuda":
                autocast_ctx = contextlib.nullcontext()

            with autocast_ctx:
                patch_tokens = infer_patch_tokens(model, pixel_values)

            out_path = write_parquet_chunk(
                output_dir=output_dir,
                local_rank=local_rank,
                chunk_idx=chunk_idx,
                patch_tokens=patch_tokens,
                captions=captions,
                sample_ids=sample_ids,
            )

            processed += len(captions)
            chunk_idx += 1

            if progress_bar is not None:
                progress_bar.update(len(captions))

            if hub_api is not None and upload_every_chunks > 0:
                pending_uploads.append(out_path)
                if len(pending_uploads) >= upload_every_chunks:
                    flush_periodic_uploads(
                        api=hub_api,
                        repo_id=upload_repo_id,
                        prefix=upload_prefix,
                        local_paths=pending_uploads,
                    )
                    pending_uploads = []

            if local_rank == 0 and chunk_idx % 10 == 0:
                if progress_bar is not None:
                    progress_bar.write(
                        f"rank0 progress: chunks={chunk_idx}, samples={processed}, last_file={out_path}"
                    )
                else:
                    print(f"rank0 progress: chunks={chunk_idx}, samples={processed}, last_file={out_path}")

    if progress_bar is not None:
        progress_bar.close()

    if hub_api is not None and pending_uploads:
        flush_periodic_uploads(
            api=hub_api,
            repo_id=upload_repo_id,
            prefix=upload_prefix,
            local_paths=pending_uploads,
        )

    result_queue.put(processed)


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN environment variable is required")

    # This script already manages multiprocessing. Running it under torchrun will oversubscribe processes.
    torchrun_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torchrun_world_size > 1:
        raise RuntimeError(
            "This script must not be launched with torchrun. "
            "Run it directly: python preprocessing/mp_blip3o_image_encoder.py"
        )

    if args.num_procs <= 0:
        raise ValueError("--num-procs must be >= 1")

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for this script")

    available_gpus = torch.cuda.device_count()
    if args.num_procs > available_gpus:
        raise ValueError(
            f"Requested num_procs={args.num_procs}, but only {available_gpus} CUDA devices are available"
        )

    global_row_limit = resolve_global_row_limit(args)
    upload_repo_id, upload_every_rows, upload_prefix = resolve_hub_upload_config()
    # Convert user-supplied row count to chunk count using batch_size.
    upload_every_chunks = max(1, upload_every_rows // args.batch_size) if upload_every_rows > 0 else 0
    output_dir = build_output_dir(args.output_dir)

    print(f"Starting multiprocessing inference with num_procs={args.num_procs}")
    print(f"Writing parquet shards to: {output_dir}")

    if global_row_limit > 0:
        print(f"Exact global row limit enabled via ENCODE_MAX_ROWS/--max-samples: {global_row_limit}")
    if upload_every_chunks > 0:
        ensure_dataset_repo_exists(hf_token=hf_token, repo_id=upload_repo_id)
        print(
            "Periodic Hub uploads enabled: "
            f"every ~{upload_every_rows} rows (~{upload_every_chunks} chunks) per rank to dataset {upload_repo_id}"
        )

    mp.set_start_method("spawn", force=True)
    result_queue: Any = mp.SimpleQueue()

    mp.spawn(
        run_worker,
        args=(
            args,
            output_dir,
            global_row_limit,
            upload_repo_id,
            upload_every_chunks,
            upload_prefix,
            result_queue,
        ),
        nprocs=args.num_procs,
        join=True,
    )

    global_processed = 0
    for _ in range(args.num_procs):
        global_processed += int(result_queue.get())

    print("Encoding complete across all ranks.")
    print(f"Parquet output directory: {output_dir}")
    print(f"Global rows processed: {global_processed}")
    if global_row_limit > 0 and global_processed != global_row_limit:
        print(
            f"Warning: requested {global_row_limit} rows, but processed {global_processed}. "
            "The stream may have ended before reaching the target."
        )

    # -----------------------------------------------------------------
    # Optional rank-0 aggregation + push to Hugging Face Hub.
    # Uses HF_TOKEN from environment for authentication.
    #
    # from datasets import Dataset, concatenate_datasets
    # from huggingface_hub import HfApi as HuggingFaceHubAPI
    #
    # api = HuggingFaceHubAPI(token=os.environ["HF_TOKEN"])
    # repo_id = "your-username/blip3o-tipsv2-patchtokens"
    #
    # parquet_files = sorted(
    #     os.path.join(output_dir, f)
    #     for f in os.listdir(output_dir)
    #     if f.endswith(".parquet")
    # )
    #
    # shards = [Dataset.from_parquet(p) for p in parquet_files]
    # merged = concatenate_datasets(shards)
    # merged.push_to_hub(repo_id=repo_id, token=os.environ["HF_TOKEN"])
    # -----------------------------------------------------------------


if __name__ == "__main__":
    main()
