#!/usr/bin/env python3
"""
extract_features.py — Optimized, Resumable GPU Feature Extractor

Phase 3: Extracts final-layer hidden states for probe training.

For each attribute (emotion|knowledge|confidence|trust) and for each user turn:
  1. Build transcript up to that turn (User:/Assistant:)
  2. Append probe: "Assistant: I think the {attribute} of this user is"
  3. Run model with output_hidden_states=True
  4. Save last token hidden state as a feature vector

Outputs: <out_dir>/<attr>_feats.npz with:
    X: np.float32 [N, hidden_size]
    y: list[str] — gold labels
    metas: list[(conv_id, turn_idx)]
    conv_gt_json: ground truth per conversation
    model: str
    hidden_size: int
    layer_index: int
    special_prompt: str
"""

import argparse
import json
import os
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ATTRS = ["emotion", "knowledge", "confidence", "trust"]
SPECIAL_PROBE_FMT = "Assistant: I think the {attribute} of this user is"
USER_PREFIX, ASSIST_PREFIX = "User: ", "Assistant: "


# ------------------ Data Helpers ------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def group_by_attr(dialogs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g = defaultdict(list)
    for d in dialogs:
        at = d.get("attribute_type")
        if at in ATTRS:
            g[at].append(d)
    return g

def format_plain_chat_upto_turn(d: Dict[str, Any], upto_user_turn: int) -> str:
    msgs = d["messages"]
    max_idx = upto_user_turn * 2 + 1
    parts = []
    for i, m in enumerate(msgs):
        if i > max_idx:
            break
        if m["role"] == "user":
            parts.append(f"{USER_PREFIX}{m['text']}")
        else:
            parts.append(f"{ASSIST_PREFIX}{m['text']}")
    return "\n".join(parts)

def append_probe(prompt_so_far: str, attribute: str) -> str:
    sep = "\n" if prompt_so_far and not prompt_so_far.endswith("\n") else ""
    return f"{prompt_so_far}{sep}{SPECIAL_PROBE_FMT.format(attribute=attribute)}"

def build_turn_prompts_for_attr(d: Dict[str, Any], attribute: str) -> Tuple[List[str], List[str]]:
    labels_seq = d["user_state_per_turn"]
    texts, labels = [], []
    for ut in range(len(labels_seq)):
        base = format_plain_chat_upto_turn(d, ut)
        texts.append(append_probe(base, attribute))
        labels.append(labels_seq[ut])
    return texts, labels


# ------------------ Save / Resume ------------------

def _safe_save_npz(path: str, **arrays):
    d = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(dir=d, delete=False, suffix=".tmp") as tmp:
        np.savez_compressed(tmp.name, **arrays)
        tmp_path = tmp.name
    os.replace(tmp_path, path)

def _load_existing_npz(path: str):
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "X": data["X"],
        "y": data["y"].astype(object).tolist(),
        "metas": data["metas"].tolist(),
        "conv_gt_json": str(data.get("conv_gt_json", "{}")),
        "model": str(data.get("model", "")),
        "hidden_size": int(data.get("hidden_size", -1)),
        "layer_index": int(data.get("layer_index", -1)),
        "special_prompt": str(data.get("special_prompt", "")),
    }


# ------------------ Feature Extraction ------------------

@torch.no_grad()
def forward_last_token_hidden(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    torch_dtype,
    max_seq_len: int,
    batch_size: int,
) -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="extract", leave=False):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]  # [B, T, H]

        last_idx = attention_mask.sum(dim=1) - 1
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        last_h = torch.gather(hidden, 1, idx).squeeze(1)

        feats.append(last_h.to(torch.float32).cpu().numpy())

        del out, hidden, input_ids, attention_mask, last_h
        if device == "cuda":
            torch.cuda.empty_cache()

    if feats:
        return np.concatenate(feats, axis=0)
    return np.zeros((0, getattr(model.config, "hidden_size", 0)), dtype=np.float32)


# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--attr", choices=ATTRS)
    ap.add_argument("--layer_index", type=int, default=-1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[load] model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Padding fix
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device == "cuda" else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    hidden_size = getattr(model.config, "hidden_size", None)

    dialogs = load_jsonl(args.data)
    by_attr = group_by_attr(dialogs)
    if args.attr:
        by_attr = {args.attr: by_attr.get(args.attr, [])}

    for attribute, dlist in by_attr.items():
        if not dlist:
            print(f"[skip] no dialogs for {attribute}")
            continue
        print(f"[attr] {attribute}  dialogs={len(dlist)}")

        conv_gt = {
            d["id"]: {"switch": d.get("switch_user_turn"), "labels": d["user_state_per_turn"]}
            for d in dlist
        }

        all_prompts, all_labels, all_metas = [], [], []
        for d in dlist:
            cid = d["id"]
            p_texts, p_labels = build_turn_prompts_for_attr(d, attribute)
            for i, (t, y) in enumerate(zip(p_texts, p_labels)):
                all_prompts.append(t)
                all_labels.append(y)
                all_metas.append((cid, i))

        out_path = os.path.join(args.out_dir, f"{attribute}_feats.npz")
        existing = _load_existing_npz(out_path) if args.resume else None

        already_done = set(existing["metas"]) if existing else set()
        to_process_idx = [i for i, m in enumerate(all_metas) if m not in already_done]

        if not to_process_idx:
            print(f"[resume] nothing left for {attribute}")
            continue

        print(f"[todo] extracting {len(to_process_idx)} turns for {attribute}")
        subset_prompts = [all_prompts[i] for i in to_process_idx]
        subset_labels = [all_labels[i] for i in to_process_idx]
        subset_metas = [all_metas[i] for i in to_process_idx]

        X_new = forward_last_token_hidden(
            model, tokenizer, subset_prompts, args.device, torch_dtype, args.max_seq_len, args.batch_size
        )

        if existing:
            X_all = np.concatenate([existing["X"], X_new], axis=0)
            y_all = existing["y"] + subset_labels
            metas_all = existing["metas"] + subset_metas
        else:
            X_all, y_all, metas_all = X_new, subset_labels, subset_metas

        _safe_save_npz(
            out_path,
            X=X_all,
            y=np.array(y_all, dtype=object),
            metas=np.array(metas_all, dtype=object),
            conv_gt_json=json.dumps(conv_gt),
            model=args.model,
            hidden_size=hidden_size,
            layer_index=args.layer_index,
            special_prompt=SPECIAL_PROBE_FMT.format(attribute=attribute),
        )
        print(f"[done] saved {out_path}  (total={len(y_all)})")


if __name__ == "__main__":
    main()
