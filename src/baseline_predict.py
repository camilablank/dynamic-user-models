#!/usr/bin/env python3
"""
baseline_predict.py

Phase 2 baseline:
- Uses ONLY the user's turn text (no assistant text, no history, no LLM internals)
- TF-IDF (1-2 grams) + Logistic Regression
- Evaluates per-turn accuracy, switch latency error, and a composite consistency score
- Works per attribute family (emotion, knowledge, confidence, trust) or all present in the file

Usage:
  python src/baseline_predict.py --data data/dialogs_knowledge.jsonl
  python src/baseline_predict.py --data data/dialogs_mixed.jsonl --out_csv results/baseline_mixed.csv
  python src/baseline_predict.py --data data/dialogs_emotion.jsonl --attr emotion --test_size 0.25 --seed 123
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ALL_ATTRS = ["emotion", "knowledge", "confidence", "trust"]

# -------------------- Helpers --------------------

def first_switch_turn(seq):
    """Return the first index where class changes; None if never changes or length < 2."""
    if not seq or len(seq) < 2:
        return None
    for i in range(1, len(seq)):
        if seq[i] != seq[i - 1]:
            return i
    return None

def switch_latency_error(provided_switch, inferred_switch, max_len):
    """
    Normalized switch error in [0,1].
    0 = perfect; 1 = worst (missing or far off).
    """
    if provided_switch is None and inferred_switch is None:
        return 0.0
    if provided_switch is None or inferred_switch is None:
        return 1.0
    denom = max(1, max_len - 1)
    return min(1.0, abs(provided_switch - inferred_switch) / denom)

def load_jsonl(path):
    dialogs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dialogs.append(json.loads(line))
    return dialogs

def collect_samples_for_attr(dialogs, attr_type):
    """
    Build turn-level samples for a given attribute.
    Returns:
      texts: list[str]            -- user turn texts
      labels: list[str]           -- gold label per user turn
      metas: list[(conv_id,int)]  -- (conversation id, turn index)
      conv_gt: dict[conv_id] -> (switch_idx, labels_seq)
    """
    texts, labels, metas = [], [], []
    conv_gt = {}
    for d in dialogs:
        if d.get("attribute_type") != attr_type:
            continue
        conv_id = d.get("id")
        prov_labels = d.get("user_state_per_turn", [])
        prov_switch = d.get("switch_user_turn", None)
        msgs = d.get("messages", [])
        user_turns = [msgs[i]["text"] for i in range(0, len(msgs), 2)]

        # align lengths just in case
        n = min(len(user_turns), len(prov_labels))
        user_turns = user_turns[:n]
        prov_labels = prov_labels[:n]

        for ti, (txt, lab) in enumerate(zip(user_turns, prov_labels)):
            texts.append(txt)
            labels.append(lab)
            metas.append((conv_id, ti))

        conv_gt[conv_id] = (prov_switch, prov_labels)
    return texts, labels, metas, conv_gt

def evaluate_split(X_train, y_train, meta_train, X_test, y_test, meta_test, conv_gt):
    """
    Train TF-IDF + LR on train split, evaluate on test split.
    Returns metrics dict.
    """
    # Vectorize
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X_trv = vec.fit_transform(X_train)
    X_tev = vec.transform(X_test)

    # Classifier
    clf = LogisticRegression(max_iter=2000, n_jobs=1)
    clf.fit(X_trv, y_train)
    y_pred = clf.predict(X_tev)

    # Per-turn accuracy
    y_test_np = np.array(y_test)
    y_pred_np = np.array(y_pred)
    per_turn_acc = float((y_pred_np == y_test_np).mean()) if len(y_test_np) else 0.0

    # Regroup test predictions by conversation and turn index (sorted)
    preds_by_conv = defaultdict(list)
    for (cid, ti), pred in sorted(zip(meta_test, y_pred_np), key=lambda x: (x[0][0], x[0][1])):
        preds_by_conv[cid].append(pred)

    # Switch latency error (per test conversation)
    total_sw_err, count = 0.0, 0
    for cid, preds_seq in preds_by_conv.items():
        if cid not in conv_gt:
            continue
        prov_switch, prov_labels = conv_gt[cid]
        # Ensure same length guard (rare misalignment)
        m = min(len(preds_seq), len(prov_labels))
        preds_seq = preds_seq[:m]
        prov_labels = prov_labels[:m]
        inf_sw = first_switch_turn(preds_seq)
        sw_err = switch_latency_error(prov_switch, inf_sw, len(prov_labels))
        total_sw_err += sw_err
        count += 1
    mean_switch_err = (total_sw_err / count) if count else 1.0

    # Composite consistency score
    consistency_score = 0.7 * per_turn_acc + 0.3 * (1.0 - mean_switch_err)

    return {
        "per_turn_acc": round(per_turn_acc, 4),
        "mean_switch_error": round(mean_switch_err, 4),
        "consistency_score": round(consistency_score, 4),
    }

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to generated dataset (.jsonl)")
    ap.add_argument("--attr", choices=ALL_ATTRS, help="If set, evaluate only this attribute type")
    ap.add_argument("--test_size", type=float, default=0.3, help="Test fraction by conversation id")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default=None, help="Optional path to save CSV of results")
    args = ap.parse_args()

    dialogs = load_jsonl(args.data)

    # Attribute types present in file (filtered if --attr provided)
    present_attrs = sorted(set(d.get("attribute_type") for d in dialogs if d.get("attribute_type") in ALL_ATTRS))
    if args.attr:
        if args.attr not in present_attrs:
            print(f"[warn] Requested attr '{args.attr}' not found in file; available: {present_attrs}")
            return
        attr_list = [args.attr]
    else:
        attr_list = present_attrs

    rows = []
    for attr_type in attr_list:
        texts, labels, metas, conv_gt = collect_samples_for_attr(dialogs, attr_type)
        if not texts:
            print(f"[skip] No samples for attr {attr_type}")
            continue

        # Train/test split by conversation ID
        conv_ids = sorted({cid for cid, _ in metas})
        train_ids, test_ids = train_test_split(conv_ids, test_size=args.test_size, random_state=args.seed)

        # Build splits using metas to keep aligned indices
        X_train, y_train, meta_train = [], [], []
        X_test, y_test, meta_test = [], [], []
        for txt, lab, (cid, ti) in zip(texts, labels, metas):
            if cid in train_ids:
                X_train.append(txt); y_train.append(lab); meta_train.append((cid, ti))
            else:
                X_test.append(txt); y_test.append(lab); meta_test.append((cid, ti))

        if not X_test or not X_train:
            print(f"[warn] Not enough data for attr {attr_type} after split (train={len(X_train)}, test={len(X_test)}). Skipping.")
            continue

        metrics = evaluate_split(X_train, y_train, meta_train, X_test, y_test, meta_test, conv_gt)
        rows.append({
            "attribute": attr_type,
            **metrics,
            "n_train_turns": len(X_train),
            "n_test_turns": len(X_test),
            "n_train_convs": len(train_ids),
            "n_test_convs": len(test_ids),
        })

    # Print table
    print("attribute,per_turn_acc,mean_switch_error,consistency_score,n_train_turns,n_test_turns,n_train_convs,n_test_convs")
    for r in rows:
        print(",".join(str(r[k]) for k in [
            "attribute","per_turn_acc","mean_switch_error","consistency_score",
            "n_train_turns","n_test_turns","n_train_convs","n_test_convs"
        ]))

    # Optional CSV
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8") as f:
            f.write("attribute,per_turn_acc,mean_switch_error,consistency_score,n_train_turns,n_test_turns,n_train_convs,n_test_convs\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in [
                    "attribute","per_turn_acc","mean_switch_error","consistency_score",
                    "n_train_turns","n_test_turns","n_train_convs","n_test_convs"
                ]) + "\n")
        print(f"[saved] {args.out_csv}")

if __name__ == "__main__":
    main()
