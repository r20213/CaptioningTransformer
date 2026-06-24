#!/usr/bin/env python3
import argparse
import json
import os
import sys
import torch
import copy
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
    # 1. Force HF Authentication before everything else downloads
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN not found in environment. Proceeding without explicit login.")

    # 2. Setup Available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs detected: {num_gpus}")
    
    # 3. Initialize Tokenizer
    tokenizer = CaptionTokenizer(repo_id=args.tokenizer_repo_id, hf_token=hf_token)
    
    # 4. Initialize Base Model
    base_model = CaptioningTransformerV1(
        vocab_size=16000, embed_dim=384, n_layers=30, n_heads=12,
        n_kv_heads=4, use_gradient_checkpointing=False, qk_norm=True
    )
    
    # Load state dict on CPU first to avoid GPU 0 memory spikes
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    base_model.load_state_dict(state_dict)
    base_model.eval()

    # 5. Distribute clones of the model to available devices
    models_pool = {}
    if num_gpus >= 2:
        print("Distributing model replicas across 2 GPUs to balance the workload...")
        models_pool[0] = copy.deepcopy(base_model).to("cuda:0")
        models_pool[1] = copy.deepcopy(base_model).to("cuda:1")
        del base_model # Free up the base CPU architecture memory
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Fallback to single device execution on: {device}")
        models_pool[0] = base_model.to(device)

    # 6. Load Data
    data = load_training_examples(
        encoded_dataset_id=args.encoded_dataset_id,
        split=args.split,
        train_examples=args.train_examples,
        max_text_len=args.max_text_len,
        hf_token=hf_token,
        tokenizer=tokenizer,
    )
    
    # 7. Generate Pairs (Alternating between GPUs to leverage both)
    print("Generating DPO preference pairs...")
    with open("dpo_dataset.jsonl", "w") as f:
        for idx, ex in enumerate(tqdm(data[:100])):
            
            # Alternate target execution device based on item index loop
            if num_gpus >= 2:
                gpu_id = idx % 2
                current_device = torch.device(f"cuda:{gpu_id}")
                current_model = models_pool[gpu_id]
            else:
                current_device = next(models_pool[0].parameters()).device
                current_model = models_pool[0]

            # Put target patch tokens on the active execution GPU
            image_tokens = ex.image_tokens.to(current_device)
            
            # Generate options utilizing the specific GPU model instance
            chosen = ex.caption  # Use the original caption as the chosen one
            rejected = generate_caption(current_model, tokenizer, image_tokens, 64, current_device, repetition_penalty=0.5)
            
            if chosen == rejected:
                rejected = "This image is unidentifiable." 
                
            json.dump({
                "sample_id": ex.sample_id,
                "patch_tokens": ex.image_tokens.tolist(), # Keeps original format
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