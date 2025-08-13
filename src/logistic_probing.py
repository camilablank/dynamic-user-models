#!/usr/bin/env python3
"""
phase_3_probes.py

Phase 3: Train linear logistic probes on residual stream representations
captured at the last token of a special probe message:
  "Assistant: I think the {state} of this user is"

Per state family (emotion, knowledge, confidence, trust), we:
  - Extract features (final-layer last-token hidden state) per user turn
  - Train multinomial logistic regression (linear probe)
  - Evaluate per-turn accuracy, switch latency error, consistency score

Usage (Llama‑2‑13B, 5120-dim):
  python src/phase_3_probes.py \
    --data data/dialogs_mixed.jsonl \
    --model meta-llama/Llama-2-13b-hf \
    --batch_size 2 --max_seq_len 2048 \
    --device cuda --dtype bfloat16 \
    --out_csv results/probes_mixed.csv

Notes:
- If your model isn't chat-tuned, we use a simple "User:/Assistant:" format.
- If you have a chat template, pass --use_chat_template to let HF apply it.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTRS = ["emotion", "knowledge", "confidence", "trust"]

SPECIAL_PROBE_FMT = "Assistant: I think the {state} of this user is"
USER_PREFIX = "User: "
ASSIST_PREFIX = "Assistant: "

# ---------- metrics helpers ----------

def first_switch_turn(seq: List[str]) -> int | None:
    if len(seq) < 2:
        return None
    for i in range(1, len(seq)):
        if seq[i] != seq[i-1]:
            return i
    return None

def switch_latency_error(prov_switch: int | None, inf_switch: int | None, max_len: int) -> float:
    if prov_switch is None and inf_switch is None:
        return 0.0
    if prov_switch is None or inf_switch is None:
        return 1.0
    denom = max(1, max_len - 1)
    return min(1.0, abs(prov_switch - inf_switch) / denom)

# ---------- data loading ----------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def group_by_attr(dialogs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g = defaultdict(list)
    for d in dialogs:
        at = d.get("state_type")
        if at in ATTRS:
            g[at].append(d)
    return g

# ---------- prompt building ----------

def format_plain_chat_upto_turn(d: Dict[str, Any], upto_user_turn: int) -> str:
    """
    Build a simple 'User:/Assistant:' transcript up through the given user turn index (inclusive).
    upto_user_turn is 0-based in user turns.
    """
    msgs = d["messages"]
    # user turns are at even indices 0,2,4..., assistant at odd 1,3,5...
    # we want to include user turn upto_user_turn and all assistant replies before it.
    MAX_IDX = upto_user_turn * 2 + 1  # include assistant reply after that user if it exists
    parts = []
    for i, m in enumerate(msgs):
        if i > MAX_IDX:
            break
        role = m["role"]
        txt = m["text"]
        if role == "user":
            parts.append(f"{USER_PREFIX}{txt}")
        else:
            parts.append(f"{ASSIST_PREFIX}{txt}")
    return "\n".join(parts)

def append_probe(prompt_so_far: str, state: str) -> str:
    return (prompt_so_far + ("\n" if prompt_so_far and not prompt_so_far.endswith("\n") else "")
            + SPECIAL_PROBE_FMT.format(state=state))

# ---------- feature extraction ----------

@torch.no_grad()
def extract_last_token_repr(
    model, tokenizer, texts: List[str], device: str, dtype: torch.dtype,
    max_seq_len: int, use_chat_template: bool
) -> np.ndarray:
    """
    For each text in 'texts', tokenize+forward with output_hidden_states=True,
    grab final layer last-token hidden state. Returns [N, hidden_size] numpy array.
    """
    feats = []
    for t in texts:
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            # Try to wrap as a single assistant message in chat template
            # (Many templates expect a list of messages with roles.)
            chat_like = [{"role": "user", "content": t}]
            enc = tokenizer.apply_chat_template(chat_like, add_generation_prompt=False, tokenize=True, return_tensors="pt")
            input_ids = enc.to(device)
        else:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = {k: v.to(device) for k, v in enc.items()}

        out = model(**input_ids, output_hidden_states=True)
        last_h = out.hidden_states[-1]  # [1, seq_len, hidden]
        vec = last_h[:, -1, :].squeeze(0).to(torch.float32).cpu().numpy()  # [hidden]
        feats.append(vec)
    return np.stack(feats, axis=0)

def build_turn_prompts_for_attr(d: Dict[str, Any], state: str, use_chat_template: bool) -> Tuple[List[str], List[str]]:
    """
    For a single dialog d and state family, return:
      texts: list of prompts (one per user turn)
      labels: list of labels (gold per-turn labels)
    """
    labels_seq = d["user_state_per_turn"]
    # number of user turns present in messages
    n_user_turns = len(labels_seq)
    texts = []
    labels = []
    for ut in range(n_user_turns):
        base = format_plain_chat_upto_turn(d, ut)
        probe = append_probe(base, state)
        texts.append(probe)
        labels.append(labels_seq[ut])
    return texts, labels

def collect_dataset_for_attr(dialogs_attr: List[Dict[str, Any]], state: str, use_chat_template: bool) -> Tuple[List[str], List[str], List[Tuple[str,int]], Dict[str, Tuple[int, List[str]]]]:
    """
    Builds:
      prompts: List[str]  (per user turn)
      labels:  List[str]
      metas:   List[(conv_id, turn_idx)]
      conv_gt: Dict[conv_id] -> (switch_idx, labels_seq)
    """
    prompts, labels, metas = [], [], []
    conv_gt = {}
    for d in dialogs_attr:
        conv_id = d["id"]
        switch_idx = d.get("switch_user_turn")
        labels_seq = d["user_state_per_turn"]
        conv_gt[conv_id] = (switch_idx, labels_seq)

        p_texts, p_labels = build_turn_prompts_for_attr(d, state, use_chat_template)
        for i, (t, y) in enumerate(zip(p_texts, p_labels)):
            prompts.append(t)
            labels.append(y)
            metas.append((conv_id, i))
    return prompts, labels, metas, conv_gt

# ---------- training/eval ----------

def eval_probe(X_tr: np.ndarray, y_tr: List[str], X_te: np.ndarray, y_te: List[str],
               meta_te: List[Tuple[str,int]], conv_gt: Dict[str, Tuple[int, List[str]]]) -> Dict[str, float]:
    # multinomial logistic regression (linear, softmax)
    clf = LogisticRegression(
        max_iter=2000, n_jobs=1, solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    # per-turn accuracy
    acc = float((y_pred == np.array(y_te)).mean()) if len(y_te) else 0.0

    # switch latency across test conversations
    preds_by_conv = defaultdict(list)
    for (cid, ti), pred in sorted(zip(meta_te, y_pred), key=lambda x: (x[0][0], x[0][1])):
        preds_by_conv[cid].append(pred)

    total_sw_err, count = 0.0, 0
    for cid, seq in preds_by_conv.items():
        if cid not in conv_gt:
            continue
        gt_switch, gt_labels = conv_gt[cid]
        m = min(len(seq), len(gt_labels))
        if m == 0:
            continue
        seq = seq[:m]; gt_labels = gt_labels[:m]
        inf_sw = first_switch_turn(seq)
        err = switch_latency_error(gt_switch, inf_sw, len(gt_labels))
        total_sw_err += err
        count += 1
    mean_sw_err = (total_sw_err / count) if count else 1.0

    score = 0.7 * acc + 0.3 * (1.0 - mean_sw_err)
    return {
        "per_turn_acc": round(acc, 4),
        "mean_switch_error": round(mean_sw_err, 4),
        "consistency_score": round(score, 4),
    }

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Phase‑1 JSONL")
    ap.add_argument("--model", required=True, help="HF model id, e.g. meta-llama/Llama-2-13b-hf")
    ap.add_argument("--device", default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16","float32"])
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1, help="Feature extraction is sequential per prompt; batching helps with KV cache but we keep 1 for safety.")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attr", choices=ATTRS, help="Only run this state")
    ap.add_argument("--use_chat_template", action="store_true", help="Apply tokenizer chat template if available")
    ap.add_argument("--save_feats_dir", type=str, default=None, help="Optional: save features/labels per state to npz")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional: save summary CSV")
    args = ap.parse_args()

    # dtype
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[load] model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device=="cuda" else None
    )
    model.eval()

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        print("[warn] model.config.hidden_size not found.")
    else:
        print(f"[info] model hidden_size={hidden_size}")
        if hidden_size != 5120:
            print(f"[warn] hidden_size != 5120 (yours={hidden_size}). That's fine—the probe adapts, but your spec referenced 5120.")

    # data
    dialogs = load_jsonl(args.data)
    by_attr = group_by_attr(dialogs)
    if args.attr:
        by_attr = {args.attr: by_attr.get(args.attr, [])}

    rows = []
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True) if args.out_csv else None
    if args.save_feats_dir:
        os.makedirs(args.save_feats_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    for state, dlist in by_attr.items():
        if not dlist:
            print(f"[skip] no dialogs for {state}")
            continue
        print(f"[run] state={state} dialogs={len(dlist)}")

        # Build prompts/labels
        prompts, labels, metas, conv_gt = collect_dataset_for_attr(dlist, state, args.use_chat_template)

        # Train/test split by conversation id
        conv_ids = sorted({cid for cid, _ in metas})
        tr_ids, te_ids = train_test_split(conv_ids, test_size=args.test_size, random_state=args.seed)

        # Partition prompts/labels/metas
        prompts_tr, labels_tr, metas_tr = [], [], []
        prompts_te, labels_te, metas_te = [], [], []
        for p, y, (cid, ti) in zip(prompts, labels, metas):
            if cid in tr_ids:
                prompts_tr.append(p); labels_tr.append(y); metas_tr.append((cid, ti))
            else:
                prompts_te.append(p); labels_te.append(y); metas_te.append((cid, ti))

        # Feature extraction (could be batched; we do 1-by-1 for simplicity/robustness)
        print(f"[feats] extracting train ({len(prompts_tr)})")
        X_tr = extract_last_token_repr(model, tokenizer, prompts_tr, args.device, torch_dtype, args.max_seq_len, args.use_chat_template)
        print(f"[feats] extracting test  ({len(prompts_te)})")
        X_te = extract_last_token_repr(model, tokenizer, prompts_te, args.device, torch_dtype, args.max_seq_len, args.use_chat_template)

        if args.save_feats_dir:
            np.savez_compressed(
                os.path.join(args.save_feats_dir, f"{state}_feats.npz"),
                X_tr=X_tr, y_tr=np.array(labels_tr, dtype=object),
                X_te=X_te, y_te=np.array(labels_te, dtype=object),
                metas_tr=np.array(metas_tr, dtype=object),
                metas_te=np.array(metas_te, dtype=object)
            )

        # Train/eval probe
        metrics = eval_probe(X_tr, labels_tr, X_te, labels_te, metas_te, conv_gt)
        row = {
            "state": state,
            "per_turn_acc": metrics["per_turn_acc"],
            "mean_switch_error": metrics["mean_switch_error"],
            "consistency_score": metrics["consistency_score"],
            "n_train_turns": len(labels_tr),
            "n_test_turns": len(labels_te),
            "n_train_convs": len(tr_ids),
            "n_test_convs": len(te_ids),
            "hidden_size": int(hidden_size) if hidden_size else None,
            "model": args.model
        }
        rows.append(row)

        print("state,per_turn_acc,mean_switch_error,consistency_score,n_train_turns,n_test_turns,n_train_convs,n_test_convs,hidden_size,model")
        print(",".join(str(row[k]) for k in [
            "state","per_turn_acc","mean_switch_error","consistency_score",
            "n_train_turns","n_test_turns","n_train_convs","n_test_convs","hidden_size","model"
        ]))

    # Save CSV summary
    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8") as f:
            f.write("state,per_turn_acc,mean_switch_error,consistency_score,n_train_turns,n_test_turns,n_train_convs,n_test_convs,hidden_size,model\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in [
                    "state","per_turn_acc","mean_switch_error","consistency_score",
                    "n_train_turns","n_test_turns","n_train_convs","n_test_convs","hidden_size","model"
                ]) + "\n")
        print(f"[saved] {args.out_csv}")

if __name__ == "__main__":
    main()