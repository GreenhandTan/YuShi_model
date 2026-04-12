import argparse
import csv
import json
from pathlib import Path

import torch

from model import ContentAuditExpert
from dataset import SimpleTokenizer


def load_jsonl(path):
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                continue
            t = str(d.get("text", "")).strip()
            if not t:
                continue
            y = int(d.get("is_violation", 0))
            texts.append(t)
            labels.append(y)
    return texts, labels


def calc_metrics(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / total if total else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
    }


def run_raw_inference(model, tokenizer, texts, max_length, batch_size):
    raw_preds = []
    confs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results = model.audit_batch(batch, tokenizer=tokenizer, max_length=max_length, batch_size=batch_size)
        for r in results:
            raw_preds.append(1 if bool(r.get("is_violation", False)) else 0)
            confs.append(float(r.get("confidence", 0.0)))
    return raw_preds, confs


def evaluate_thresholds(labels, raw_preds, confs, thr_start, thr_end, thr_step, alpha, beta):
    rows = []
    thr = thr_start
    while thr <= thr_end + 1e-9:
        tp = fp = tn = fn = 0
        for y, raw, c in zip(labels, raw_preds, confs):
            pred = 1 if (raw == 1 and c >= thr) else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1

        m = calc_metrics(tp, fp, tn, fn)
        balance_gap = abs(m["fpr"] - m["fnr"])
        cost = alpha * m["fpr"] + beta * m["fnr"]
        rows.append({
            "threshold": round(thr, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": round(m["accuracy"], 6),
            "precision": round(m["precision"], 6),
            "recall": round(m["recall"], 6),
            "f1": round(m["f1"], 6),
            "fpr": round(m["fpr"], 6),
            "fnr": round(m["fnr"], 6),
            "balance_gap": round(balance_gap, 6),
            "cost": round(cost, 6),
        })
        thr += thr_step
    return rows


def pick_best(rows, objective):
    if objective == "cost":
        return min(rows, key=lambda r: (r["cost"], r["balance_gap"], -r["f1"]))
    return min(rows, key=lambda r: (r["balance_gap"], -r["f1"], r["cost"]))


def parse_args():
    p = argparse.ArgumentParser(description="Violation threshold grid search")
    p.add_argument("--ckpt", required=True, help="Checkpoint path")
    p.add_argument("--vocab", required=True, help="Vocab path")
    p.add_argument("--val_jsonl", required=True, help="Validation JSONL with is_violation labels")
    p.add_argument("--device", default="auto", help="auto/cpu/cuda")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--thr_start", type=float, default=0.30)
    p.add_argument("--thr_end", type=float, default=0.80)
    p.add_argument("--thr_step", type=float, default=0.01)
    p.add_argument("--objective", choices=["balanced", "cost"], default="balanced")
    p.add_argument("--alpha", type=float, default=1.0, help="FPR weight for cost objective")
    p.add_argument("--beta", type=float, default=1.0, help="FNR weight for cost objective")
    p.add_argument("--out_csv", default="./checkpoints_final_9to1/threshold_search_results.csv")
    p.add_argument("--out_best", default="./checkpoints_final_9to1/best_threshold.json")
    return p.parse_args()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()

    device = resolve_device(args.device)
    tokenizer = SimpleTokenizer.load(args.vocab)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    margs = ckpt.get("args", {})

    model = ContentAuditExpert(
        vocab_size=tokenizer.vocab_size,
        dim=margs.get("dim", 256),
        n_layers=margs.get("n_layers", 6),
        n_heads=margs.get("n_heads", 4),
        ffn_multiplier=margs.get("ffn_multiplier", 4),
        max_seq_len=margs.get("max_seq_len", 256),
        pad_token_id=tokenizer.pad_token_id,
    )
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()})
    model.to(device)
    model.eval()

    texts, labels = load_jsonl(args.val_jsonl)
    if not texts:
        raise RuntimeError("Validation set is empty.")

    raw_preds, confs = run_raw_inference(model, tokenizer, texts, args.max_length, args.batch_size)
    rows = evaluate_thresholds(
        labels=labels,
        raw_preds=raw_preds,
        confs=confs,
        thr_start=args.thr_start,
        thr_end=args.thr_end,
        thr_step=args.thr_step,
        alpha=args.alpha,
        beta=args.beta,
    )
    best = pick_best(rows, args.objective)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    best_payload = {
        "objective": args.objective,
        "device": str(device),
        "samples": len(texts),
        "best": best,
        "ckpt": args.ckpt,
        "val_jsonl": args.val_jsonl,
        "search_range": {
            "start": args.thr_start,
            "end": args.thr_end,
            "step": args.thr_step,
        },
    }
    out_best = Path(args.out_best)
    out_best.parent.mkdir(parents=True, exist_ok=True)
    with open(out_best, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(best_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
