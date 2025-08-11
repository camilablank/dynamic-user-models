#!/usr/bin/env python3
"""
extract_features.py (GPU, optimized)

Phase 3 feature extraction:
  For each attribute (emotion|knowledge|confidence|trust), and for each user turn:
    - Build transcript up to that turn (User:/Assistant:)
    - Append: "Assistant: I think the {attribute} of this user is"
    - Run model with output_hidden_states=True
    - Take final-layer last-token hidden state as the feature vector

Outputs (one per attribute):
  <out_dir>/<attr>_feats.npz with fields:
    X: float32 [N, hidden_size]
    y: object [N]               (string labels)
    metas: object [N,2]         (conv_id, turn_idx)
    conv_gt_json: JSON string {conv_id: {"switch": int|None, "labels": [str,...]}}
    model: str
    hidden_size: int
    layer_index: int            (usually -1, final layer)
    special_prompt: str

Usage:
  python src/extract_features.py \
    --data data/dialogs_mixed.jsonl \
    --model meta-llama/Llama-2-13b-hf \
    --device cuda --dtype bfloat16 \
    --batch_size 8 --max_seq_len 2048 \
    --out_dir features/llama2_13b
"""

import argparse, json, os, math
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ATTRS = ["emotion", "knowledge", "confidence", "trust"]
SPECIAL_PROBE_FMT = "Assistant: I think the {attribute} of this user is"
USER_PREFIX, ASSIST_PREFIX = "User: ", "Assistant: "

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f: rows.append(json.loads(line))
    return rows

def group_by_attr(dialogs):
    g = defaultdict(list)
    for d in dialogs:
        at = d.get("attribute_type")
        if at in ATTRS:
            g[at].append(d)
    return g

def format_plain_chat_upto_turn(d: Dict[str, Any], upto_user_turn: int) -> str:
    """
    Build a simple 'User:/Assistant:' transcript including the given user turn
    and any assistant replies up to that point.
    """
    msgs = d["messages"]
    max_idx = upto_user_turn * 2 + 1
    parts = []
    for i, m in enumerate(msgs):
        if i > max_idx: break
        if m["role"] == "user":
            parts.append(f"{USER_PREFIX}{m['text']}")
        else:
            parts.append(f"{ASSIST_PREFIX}{m['text']}")
    return "\n".join(parts)

def append_probe(prompt_so_far: str, attribute: str) -> str:
    sep = "\n" if prompt_so_far and not prompt_so_far.endswith("\n") else ""
    return f"{prompt_so_far}{sep}{SPECIAL_PROBE_FMT.format(attribute=attribute)}"

def build_turn_prompts_for_attr(d, attribute: str):
    """
    Returns:
      texts:  list[str]   probe prompts per user turn
      labels: list[str]   gold label for that user turn
    """
    labels_seq = d["user_state_per_turn"]
    texts, labels = [], []
    for ut in range(len(labels_seq)):
        base = format_plain_chat_upto_turn(d, ut)
        texts.append(append_probe(base, attribute))
        labels.append(labels_seq[ut])
    return texts, labels

@torch.no_grad()
def forward_last_token_hidden(
    model, tokenizer, texts: List[str],
    device: str, torch_dtype, max_seq_len: int, batch_size: int
) -> np.ndarray:
    """
    Batched forward pass; returns [N, hidden_size] float32 array with last real-token hidden state.
    """
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="extract", leave=False):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # [B, T, H]

        # last non-pad token index per sample: sum(attn_mask)-1
        last_idx = attention_mask.sum(dim=1) - 1  # [B]
        # gather last hidden per sample
        B, T, H = hidden.shape
        idx = last_idx.view(B, 1, 1).expand(B, 1, H)  # [B,1,H]
        last_h = torch.gather(hidden, dim=1, index=idx).squeeze(1)  # [B,H]

        feats.append(last_h.to(torch.float32).cpu().numpy())
        # free memory
        del out, hidden, input_ids, attention_mask, last_h; torch.cuda.empty_cache()
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, model.config.hidden_size), dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to Phase-1 JSONL")
    ap.add_argument("--model", required=True, help="HF model id, e.g., meta-llama/Llama-2-13b-hf")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--attr", choices=ATTRS, help="Only extract this attribute")
    ap.add_argument("--layer_index", type=int, default=-1, help="Hidden layer index to use (-1 = final)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Fast math on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[load] model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device == "cuda" else None
    )
    model.eval()

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        print("[warn] model.config.hidden_size missing; continuing.")

    dialogs = load_jsonl(args.data)
    by_attr = group_by_attr(dialogs)
    if args.attr:
        by_attr = {args.attr: by_attr.get(args.attr, [])}

    for attribute, dlist in by_attr.items():
        if not dlist:
            print(f"[skip] no dialogs for {attribute}")
            continue
        print(f"[attr] {attribute}  dialogs={len(dlist)}")

        # Build prompts/labels/metas & conv_gt
        prompts, labels, metas = [], [], []
        conv_gt = {}
        for d in dlist:
            cid = d["id"]
            conv_gt[cid] = {"switch": d.get("switch_user_turn"), "labels": d["user_state_per_turn"]}
            p_texts, p_labels = build_turn_prompts_for_attr(d, attribute)
            for i, (t, y) in enumerate(zip(p_texts, p_labels)):
                prompts.append(t); labels.append(y); metas.append((cid, i))

        print(f"[feats] extracting N={len(prompts)}  batch={args.batch_size}  max_len={args.max_seq_len}  dtype={args.dtype}")
        X = forward_last_token_hidden(model, tokenizer, prompts, args.device, torch_dtype, args.max_seq_len, args.batch_size)

        out_path = os.path.join(args.out_dir, f"{attribute}_feats.npz")
        np.savez_compressed(
            out_path,
            X=X.astype("float32"),
            y=np.array(labels, dtype=object),
            metas=np.array(metas, dtype=object),
            conv_gt_json=json.dumps(conv_gt),
            model=args.model,
            hidden_size=(hidden_size if hidden_size is not None else -1),
            layer_index=args.layer_index,
            special_prompt=SPECIAL_PROBE_FMT
        )
        print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
