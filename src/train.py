"""Fine-tune EfficientNet-B0 and save weights + evaluation metrics.

Usage:  python src/train.py --epochs 15
Outputs: models/model.pt, models/metrics.json (read by the Streamlit dashboard)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, roc_curve)
from torch.utils.data import DataLoader, Dataset

from config import CLASSES, MODELS_DIR, RAW_DIR, SEED, SPLITS_DIR
from model import build_model, eval_transform, train_transform


class CTDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(RAW_DIR / r.path).convert("RGB")
        return self.tf(img), CLASSES.index(r.label)


def run_epoch(model, loader, device, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    total, correct, loss_sum, probs, ys = 0, 0, 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
            probs.append(out.softmax(1).detach().cpu().numpy())
            ys.append(y.cpu().numpy())
    return loss_sum / total, correct / total, np.concatenate(probs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = CTDataset(pd.read_csv(SPLITS_DIR / "train.csv"), train_transform())
    val_ds = CTDataset(pd.read_csv(SPLITS_DIR / "val.csv"), eval_transform())
    test_ds = CTDataset(pd.read_csv(SPLITS_DIR / "test.csv"), eval_transform())
    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch, shuffle=sh, num_workers=2)

    # Class weights = inverse frequency, to offset the small benign class.
    counts = train_ds.df.label.value_counts().reindex(CLASSES).values
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    model = build_model(len(CLASSES), pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history, best_f1 = [], -1.0
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tl, ta, _, _ = run_epoch(model, dl(train_ds, True), device, loss_fn, opt)
        vl, va, vp, vy = run_epoch(model, dl(val_ds, False), device, loss_fn)
        sched.step()
        vf1 = f1_score(vy, vp.argmax(1), average="macro")
        history.append({"epoch": ep, "train_loss": tl, "train_acc": ta,
                        "val_loss": vl, "val_acc": va, "val_f1": vf1})
        print(f"ep {ep:02d} train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f} f1 {vf1:.3f}")
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), MODELS_DIR / "model.pt")
    train_time = time.time() - t0

    # Evaluate the best checkpoint on the held-out test set.
    model.load_state_dict(torch.load(MODELS_DIR / "model.pt", map_location=device))
    t1 = time.time()
    _, test_acc, tp, ty = run_epoch(model, dl(test_ds, False), device, loss_fn)
    infer_ms = (time.time() - t1) / len(test_ds) * 1000
    pred = tp.argmax(1)

    roc = {}
    for i, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(ty == i, tp[:, i])
        roc[c] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}

    metrics = {
        "classes": CLASSES,
        "params": vars(args) | {"device": device, "model": "efficientnet_b0"},
        "split_sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "history": history,
        "test": {
            "accuracy": float(accuracy_score(ty, pred)),
            "macro_f1": float(f1_score(ty, pred, average="macro")),
            "confusion_matrix": confusion_matrix(ty, pred, labels=range(len(CLASSES))).tolist(),
            "report": classification_report(ty, pred, labels=range(len(CLASSES)),
                                            target_names=CLASSES, output_dict=True, zero_division=0),
            "roc": roc,
        },
        "timing": {"train_seconds": train_time, "inference_ms_per_image": infer_ms},
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test accuracy {metrics['test']['accuracy']:.3f}, macro F1 {metrics['test']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
