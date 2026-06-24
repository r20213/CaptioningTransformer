#!/usr/bin/env python3
import argparse
import json
import os
import sys
import torch
from tqdm import tqdm
# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src import CaptioningTransformerV1
from run_dpo_overfit import CaptionTokenizer, generate_caption
from run_overfit_100 import load_training_examples 
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def generate_dpo_pairs(checkpoint_path: str, args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    
    # Initialize Tokenizer
    tokenizer = CaptionTokenizer(repo_id=args.tokenizer_repo_id, hf_token=hf_token)
    
    # Initialize and Load Model
    model = CaptioningTransformerV1(
        vocab_size=16000, embed_dim=384, n_layers=30, n_heads=12,
        n_kv_heads=4, use_gradient_checkpointing=False, qk_norm=True
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    # Load Data
    data = load_training_examples(
        encoded_dataset_id=args.encoded_dataset_id,
        split=args.split,
        train_examples=args.train_examples,
        max_text_len=args.max_text_len,
        hf_token=hf_token,
        tokenizer=tokenizer,
    )
    
    # Generate Pairs
    with open("dpo_dataset.jsonl", "w") as f:
        for ex in tqdm(data[:100]):
            chosen = generate_caption(model, tokenizer, ex.image_tokens, 64, device, repetition_penalty=1.05)
            rejected = generate_caption(model, tokenizer, ex.image_tokens, 64, device, repetition_penalty=0.5)
            
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
    
    # Data loading arguments (matching your previous usage)
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