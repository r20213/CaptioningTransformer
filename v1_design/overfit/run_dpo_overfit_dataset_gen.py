#!/usr/bin/env python3
import argparse
import json
import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src import CaptioningTransformerV1
from run_dpo_overfit import CaptionTokenizer, generate_caption
from run_overfit_100 import load_training_examples 
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def generate_dpo_pairs(checkpoint_path: str, args: argparse.Namespace):
    # 1. HF Authentication before everything else
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not found in environment. Proceeding without explicit login.")

    # 2. Setup Multi-GPU Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}. Number of available GPUs: {num_gpus}")
    
    # 3. Initialize Tokenizer
    tokenizer = CaptionTokenizer(repo_id=args.tokenizer_repo_id, hf_token=hf_token)
    
    # 4. Initialize and Load Model
    model = CaptioningTransformerV1(
        vocab_size=16000, embed_dim=384, n_layers=30, n_heads=12,
        n_kv_heads=4, use_gradient_checkpointing=False, qk_norm=True
    )
    
    # Load state dict on CPU first to prevent GPU 0 OOM, then transfer
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Wrap model with DataParallel if multiple GPUs exist
    if num_gpus > 1:
        print(f"Wrapping model in DataParallel across {num_gpus} GPUs.")
        model = nn.DataParallel(model)

    # 6. Load Data
    data = load_training_examples(
        encoded_dataset_id=args.encoded_dataset_id,
        split=args.split,
        train_examples=args.train_examples,
        max_text_len=args.max_text_len,
        hf_token=hf_token,
        tokenizer=tokenizer,
    )
    
    # Limit dataset to the targeted slice
    dataset_slice = data[:100]
    
    # 7. Generate Pairs (Batched to make use of both GPUs)
    # Set batch size to match or be a multiple of your GPU count
    batch_size = max(2, num_gpus) 
    
    print(f"Generating captions with a batch size of {batch_size}...")
    with open("dpo_dataset.jsonl", "w") as f:
        for i in tqdm(range(0, len(dataset_slice), batch_size)):
            batch = dataset_slice[i:i + batch_size]
            
            # Stack tokens along a new batch dimension
            # Assumes ex.image_tokens is a 1D tensor of patches
            batch_image_tokens = torch.stack([ex.image_tokens for ex in batch]).to(device)
            
            # generate_caption handles the forward pass under the hood.
            # nn.DataParallel splits this batch across your GPUs automatically.
            chosen_batch = generate_caption(model, tokenizer, batch_image_tokens, 64, device, repetition_penalty=1.05)
            rejected_batch = generate_caption(model, tokenizer, batch_image_tokens, 64, device, repetition_penalty=0.5)
            
            # If your generate_caption function returns a single string instead of a list when batched, 
            # wrap them back into a list to keep the iteration consistent:
            if isinstance(chosen_batch, str):
                chosen_batch = [chosen_batch]
            if isinstance(rejected_batch, str):
                rejected_batch = [rejected_batch]

            # Write batch results to jsonl
            for idx, ex in enumerate(batch):
                chosen = chosen_batch[idx] if idx < len(chosen_batch) else ""
                rejected = rejected_batch[idx] if idx < len(rejected_batch) else ""
                
                if chosen == rejected:
                    rejected = "This image is unidentifiable." 
                    
                json.dump({
                    "sample_id": ex.sample_id,
                    "patch_tokens": ex.image_tokens.tolist(),
                    "chosen": chosen,
                    "rejected": rejected
                }, f)
                f.write("\n")

    print("Dataset saved to dpo_dataset.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DPO preference pairs")
    
    # Required arguments
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to .pt file")
    
    # Data loading arguments
    parser.add_argument("--encoded-dataset-id", type=str, default=os.environ.get("HF_DATASET_REPO_ID", ""))
    parser.add_argument("--train-examples", type=int, default=100)
    parser.add_argument("--max-text-len", type=int, default=96)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--tokenizer-repo-id", type=str, default=os.environ.get("TOKENIZER_REPO_ID", ""))
    
    args = parser.parse_args()
    
    # Validate missing inputs
    if not args.encoded_dataset_id or not args.tokenizer_repo_id:
        print("Error: --encoded-dataset-id and --tokenizer-repo-id must be provided or set via ENV vars.")
        exit(1)
        
    generate_dpo_pairs(args.checkpoint_path, args)