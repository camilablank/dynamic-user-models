#!/usr/bin/env python3
"""
train_probes.py  (CPU)
Phase 3 — probe training & evaluation on pre-extracted features.

Inputs:
  <features_dir>/<attr>_feats.npz  (from extract_features.py)

Outputs:
  --out_csv results/probes_summary.csv  (one row per attribute)

Usage:
  python src/train_probes.py --features_dir features/llama2_13b --out_csv results/probes_llama2_13b.csv --test_size 0.3 --seed 42
"""
import argparse, json, os, glob
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ATTRS = ["emotion","knowledge","confidence","trust"]

def first_switch_turn(seq: List[str]):
    if len(seq)<2: return None
    for i in range(1,len(seq)):
        if seq[i]!=seq[i-1]: return i
    return None

def switch_latency_error(prov_switch, inf_switch, max_len:int)->float:
    if prov_switch is None and inf_switch is None: return 0.0
    if prov_switch is None or inf_switch is None: return 1.0
    denom = max(1, max_len-1)
    return min(1.0, abs(prov_switch - inf_switch) / denom)

def load_npz(path:str):
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(object).tolist()
    metas = data["metas"].tolist()           # [(conv_id, turn_idx), ...]
    conv_gt = json.loads(str(data["conv_gt_json"]))
    model = str(data["model"])
    hidden_size = int(data["hidden_size"])
    return X, y, metas, conv_gt, model, hidden_size

def train_eval_one(X, y, metas, conv_gt, test_size:float, seed:int):
    # split by conversation id
    conv_ids = sorted({cid for cid,_ in metas})
    tr_ids, te_ids = train_test_split(conv_ids, test_size=test_size, random_state=seed)

    X_tr, y_tr, meta_tr = [], [], []
    X_te, y_te, meta_te = [], [], []
    for xi, yi, (cid,ti) in zip(X, y, metas):
        if cid in tr_ids:
            X_tr.append(xi); y_tr.append(yi); meta_tr.append((cid,ti))
        else:
            X_te.append(xi); y_te.append(yi); meta_te.append((cid,ti))
    X_tr = np.vstack(X_tr) if X_tr else np.zeros((0, X.shape[1]), dtype=X.dtype)
    X_te = np.vstack(X_te) if X_te else np.zeros((0, X.shape[1]), dtype=X.dtype)

    if len(X_tr)==0 or len(X_te)==0:
        return None

    clf = LogisticRegression(max_iter=2000, n_jobs=1, solver="lbfgs", multi_class="multinomial")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc = float((y_pred == np.array(y_te)).mean()) if len(y_te) else 0.0

    # regroup preds by conv
    preds_by_conv = defaultdict(list)
    for (cid, ti), pred in sorted(zip(meta_te, y_pred), key=lambda x:(x[0][0], x[0][1])):
        preds_by_conv[cid].append(pred)

    tot_err, cnt = 0.0, 0
    for cid, seq in preds_by_conv.items():
        if cid not in conv_gt: continue
        gt_switch = conv_gt[cid]["switch"]
        gt_labels = conv_gt[cid]["labels"]
        m = min(len(seq), len(gt_labels))
        if m==0: continue
        seq = seq[:m]; gt_labels = gt_labels[:m]
        inf_sw = first_switch_turn(seq)
        err = switch_latency_error(gt_switch, inf_sw, len(gt_labels))
        tot_err += err; cnt += 1
    mean_sw_err = (tot_err/cnt) if cnt else 1.0
    score = 0.7*acc + 0.3*(1.0-mean_sw_err)

    return {
        "per_turn_acc": round(acc,4),
        "mean_switch_error": round(mean_sw_err,4),
        "consistency_score": round(score,4),
        "n_train_turns": len(X_tr),
        "n_test_turns": len(X_te),
        "n_train_convs": len(tr_ids),
        "n_test_convs": len(te_ids),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows=[]
    model_name=None; hidden_size=None

    for attr in ATTRS:
        path = os.path.join(args.features_dir, f"{attr}_feats.npz")
        if not os.path.exists(path):
            # ignore missing attribute files
            continue
        X,y,metas,conv_gt,model,hs = load_npz(path)
        model_name = model_name or model
        hidden_size = hidden_size or hs
        metrics = train_eval_one(X,y,metas,conv_gt,args.test_size,args.seed)
        if metrics is None:
            print(f"[skip] insufficient data for {attr}")
            continue
        row = {
            "attribute": attr,
            **metrics,
            "hidden_size": hidden_size,
            "model": model_name
        }
        rows.append(row)

    # print & save
    print("attribute,per_turn_acc,mean_switch_error,consistency_score,n_train_turns,n_test_turns,n_train_convs,n_test_convs,hidden_size,model")
    with open(args.out_csv,"w",encoding="utf-8") as f:
        f.write("attribute,per_turn_acc,mean_switch_error,consistency_score,n_train_turns,n_test_turns,n_train_convs,n_test_convs,hidden_size,model\n")
        for r in rows:
            line = ",".join(str(r[k]) for k in [
                "attribute","per_turn_acc","mean_switch_error","consistency_score",
                "n_train_turns","n_test_turns","n_train_convs","n_test_convs",
                "hidden_size","model"
            ])
            print(line)
            f.write(line + "\n")
    print(f"[saved] {args.out_csv}")

if __name__=="__main__":
    main()
