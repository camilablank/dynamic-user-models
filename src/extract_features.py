#!/usr/bin/env python3
"""
extract_features.py — Optimized, Resumable GPU Feature Extractor

For each attribute (emotion|knowledge|confidence|trust) and for each user turn:
  - Builds transcript up to that turn (User:/Assistant:)
  - Appends: "Assistant: I think the {attribute} of this user is"
  - Runs model forward pass and extracts the last token hidden state.
Saves per-attribute .npy files to --out_dir for later probe training.
"""

import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

ATTRS = ["emotion", "knowledge", "confidence", "trust"]

def forward_last_token_hidden(model, tokenizer, prompts, device, dtype, max_seq_len, batch_size):
    """Tokenize prompts, forward pass, get last token hidden states."""
    all_vecs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches", leave=False):
        batch_prompts = prompts[i:i+batch_size]
        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            # Last token from each sequence
            last_vecs = last_hidden[torch.arange(last_hidden.size(0)), enc["attention_mask"].sum(dim=1) - 1]
            all_vecs.append(last_vecs.cpu().numpy())

    return np.vstack(all_vecs)

def load_jsonl(path):
    """Load JSONL file into a list of dicts."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def save_numpy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)

def already_done(out_dir, attr):
    """Check if features for attr already exist."""
    out_path = os.path.join(out_dir, f"{attr}.npy")
    return os.path.exists(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Phase 1 dataset (.jsonl)")
    ap.add_argument("--model", required=True, help="HF model name or path")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", required=True, help="Directory to save feature arrays")
    ap.add_argument("--attr", choices=ATTRS, help="Only run for one attribute")
    ap.add_argument("--layer_index", type=int, default=-1, help="Which hidden layer to extract from")
    ap.add_argument("--resume", action="store_true", help="Skip attributes already extracted")
    args = ap.parse_args()

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }[args.dtype]

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    data = load_jsonl(args.data)

    attrs_to_run = [args.attr] if args.attr else ATTRS
    for attr in attrs_to_run:
        if args.resume and already_done(args.out_dir, attr):
            print(f"[SKIP] {attr} already extracted.")
            continue

        print(f"[START] Extracting features for attribute: {attr}")
        prompts = []
        for conv in data:
            for turn in conv["dialog"]:
                if turn["role"] == "user":
                    prefix = []
                    for t in conv["dialog"]:
                        prefix.append(f"{t['role'].capitalize()}: {t['content']}")
                        if t is turn:
                            break
                    # Append special probe prompt
                    prefix.append(f"Assistant: I think the {attr} of this user is")
                    prompts.append("\n".join(prefix))

        X = forward_last_token_hidden(model, tokenizer, prompts, args.device, torch_dtype, args.max_seq_len, args.batch_size)

        out_path = os.path.join(args.out_dir, f"{attr}.npy")
        save_numpy(out_path, X)
        print(f"[DONE] Saved {attr} features to {out_path}")

if __name__ == "__main__":
    main()
