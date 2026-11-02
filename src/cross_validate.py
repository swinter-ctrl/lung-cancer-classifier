"""5-fold block cross-validation: a more reliable estimate than one train/test split.

Each class is cut into K contiguous blocks of file numbers (cuts snapped to
likely patient boundaries, see prepare_data.py); each block is held out once.
Every fold trains for a fixed number of epochs, so no validation split is
needed for checkpoint selection.

Usage:  python src/cross_validate.py --folds 5 --epochs 10
Output: models/cv_metrics.json (per-fold metrics, mean/std, pooled ROC)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from config import CLASSES, MODELS_DIR, SEED
from model import build_model, eval_transform, train_transform
from prepare_data import block_cuts, collect
from train import CTDataset, run_epoch


def assign_folds(rows, k):
    out = []
    for label in CLASSES:
        items = [r for r in rows if r["label"] == label]
        cuts = block_cuts(items, [i / k for i in range(1, k)])
        for i, r in enumerate(items):
            r["fold"] = sum(i >= c for c in cuts)
            out.append(r)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = assign_folds(collect(), args.folds)
    print(pd.crosstab(df.label, df.fold, margins=True))

    folds, oof_prob, oof_y = [], [], []
    t0 = time.time()
    for k in range(args.folds):
        torch.manual_seed(SEED)
        train_ds = CTDataset(df[df.fold != k], train_transform())
        test_ds = CTDataset(df[df.fold == k], eval_transform())
        counts = train_ds.df.label.value_counts().reindex(CLASSES).values
        weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

        model = build_model(len(CLASSES)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
        for _ in range(args.epochs):
            run_epoch(model, train_dl, device, loss_fn, opt)
            sched.step()

        test_dl = DataLoader(test_ds, batch_size=args.batch, num_workers=2)
        _, _, prob, y = run_epoch(model, test_dl, device, loss_fn)
        pred = prob.argmax(1)
        folds.append({
            "fold": k,
            "n_test": len(y),
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "auc": float(roc_auc_score(y, prob[:, 1])),
            "confusion_matrix": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist(),
        })
        oof_prob.append(prob)
        oof_y.append(y)
        print(f"fold {k}: acc {folds[-1]['accuracy']:.3f}  f1 {folds[-1]['macro_f1']:.3f}  "
              f"auc {folds[-1]['auc']:.3f}  cm {folds[-1]['confusion_matrix']}")

    prob, y = np.concatenate(oof_prob), np.concatenate(oof_y)
    fpr, tpr, _ = roc_curve(y, prob[:, 1])
    summary = {m: {"mean": float(np.mean([f[m] for f in folds])), "std": float(np.std([f[m] for f in folds]))}
               for m in ("accuracy", "macro_f1", "auc")}
    result = {
        "classes": CLASSES,
        "params": vars(args) | {"device": device, "model": "efficientnet_b0", "split": "block"},
        "folds": folds,
        "summary": summary,
        "pooled": {
            "confusion_matrix": confusion_matrix(y, prob.argmax(1)).tolist(),
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))},
        },
        "seconds": time.time() - t0,
    }
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "cv_metrics.json").write_text(json.dumps(result, indent=2))
    for m, s in summary.items():
        print(f"{m}: {s['mean']:.3f} ± {s['std']:.3f}")


if __name__ == "__main__":
    main()
