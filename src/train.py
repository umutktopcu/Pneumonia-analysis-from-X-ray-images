"""Train a binary classifier on the preprocessed cache and evaluate it (Stage 4b; reused in Stage 5).

Protocol (D10): fit on train; early-stop on val ROC-AUC; choose the decision threshold on val
(Youden J); report val and test ONCE with the best checkpoint. Test never influences a choice.

Usage (from project root, venv active):
    python src/train.py --run-name smoke --epochs 1 --limit 128               # quick smoke test
    python src/train.py --run-name smallcnn_noweight                          # option (a)
    python src/train.py --run-name smallcnn_posweight --pos-weight auto       # option (b)
Outputs (history.csv and learning_curves.png are refreshed after every epoch):
         outputs/runs/<run-name>/{config.json, history.csv, learning_curves.png,
         predictions.csv, metrics.csv} and checkpoints/<run-name>.pt (git-ignored)
"""
import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from metrics import report, roc_auc, youden_threshold
from models import build_model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # Kaggle
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds MPS/CUDA; GPU kernels may still be slightly non-deterministic


def run_epoch(model, loader, device, criterion, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, otherwise only evaluates.
    Returns (mean loss, PNEUMONIA probabilities, labels)."""
    training = optimizer is not None
    model.train(training)  # BatchNorm uses batch statistics and Dropout is active only in training
    total_loss, scores, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device).float()
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            scores.append(torch.sigmoid(logits).detach().cpu())
            labels.append(y.cpu())
    return total_loss / len(loader.dataset), torch.cat(scores).numpy(), torch.cat(labels).numpy()


def plot_curves(history: pd.DataFrame, out: Path) -> None:
    """Learning curves, redrawn after every epoch so training can be watched live.
    Overfitting shows up as train loss falling while val loss rises, or a widening train-val AUC gap."""
    best = history.loc[history["val_auc"].idxmax(), "epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, style in (("train", "o-"), ("val", "s-")):
        axes[0].plot(history["epoch"], history[f"{split}_loss"], style, label=split)
        axes[1].plot(history["epoch"], history[f"{split}_auc"], style, label=split)
    axes[0].set(xlabel="epoch", ylabel="BCE loss", title="Loss")
    axes[1].set(xlabel="epoch", ylabel="ROC-AUC", title="ROC-AUC (train is measured on augmented batches)")
    for ax in axes:
        ax.axvline(best, color="gray", linestyle="--", linewidth=1, label=f"best val AUC (epoch {best})")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--model", default="small_cnn")
    p.add_argument("--variant", default="sq224", help="cache folder under data/processed (sq224 / sq224_clahe)")
    p.add_argument("--pos-weight", choices=["none", "auto"], default="none",
                   help="auto = n_normal / n_pneumonia in train (class-weighted BCE, option b)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=5, help="early stopping on val AUC")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="debug: use only this many images per split")
    p.add_argument("--splits", type=Path, default=Path("outputs/02_splits/splits.csv"))
    p.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device()
    splits = pd.read_csv(args.splits)
    if args.limit:
        splits = splits.groupby(["split", "label"], group_keys=False).head(args.limit // 2)

    root = args.processed / args.variant
    train_ds = ChestXrayDataset(splits, "train", root, train=True)
    eval_ds = {s: ChestXrayDataset(splits, s, root) for s in ("val", "test")}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed))
    eval_loaders = {s: DataLoader(d, batch_size=args.batch_size) for s, d in eval_ds.items()}

    n_pos = int(train_ds.labels.sum())
    n_neg = len(train_ds) - n_pos
    # pos_weight multiplies the loss of positive (PNEUMONIA) examples: < 1 down-weights the majority class
    pos_weight = torch.tensor(n_neg / n_pos, device=device) if args.pos_weight == "auto" else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path("outputs/runs") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path("checkpoints") / f"{args.run_name}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    config = {**vars(args), "device": str(device), "n_train": len(train_ds), "n_pos": n_pos, "n_neg": n_neg,
              "pos_weight_value": None if pos_weight is None else float(pos_weight),
              "n_params": sum(t.numel() for t in model.parameters()), "torch": torch.__version__}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"device={device}  train={len(train_ds)} (P={n_pos}, N={n_neg})  params={config['n_params']:,}")

    history, best_auc, epochs_without_gain = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_scores, train_y = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_scores, val_y = run_epoch(model, eval_loaders["val"], device, criterion)
        train_auc, val_auc = roc_auc(train_y, train_scores), roc_auc(val_y, val_scores)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_auc": train_auc, "val_auc": val_auc})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)  # saved every epoch,
        plot_curves(pd.DataFrame(history), out_dir / "learning_curves.png")  # so it can be watched live
        print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
              f"train_auc {train_auc:.4f}  val_auc {val_auc:.4f}  ({time.time() - start:.0f}s)")
        if val_auc > best_auc:
            best_auc, epochs_without_gain = val_auc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= args.patience:
                print(f"early stopping: no val AUC gain for {args.patience} epochs")
                break

    history = pd.DataFrame(history)

    # Final evaluation with the best checkpoint; threshold from val only.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    results, preds = {}, []
    for split, loader in eval_loaders.items():
        _, scores, y = run_epoch(model, loader, device, criterion)
        results[split] = (y, scores)
        preds.append(pd.DataFrame({"split": split, "path": [str(q) for q in eval_ds[split].paths],
                                   "label": y.astype(int), "score": scores}))
    threshold = youden_threshold(*results["val"])
    metrics = pd.DataFrame([{"run": args.run_name, "split": s, **report(y, sc, threshold)}
                            for s, (y, sc) in results.items()])
    pd.concat(preds).to_csv(out_dir / "predictions.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    cols = ["split", "threshold", "accuracy", "sensitivity", "specificity", "f1", "roc_auc"]
    print(f"\nbest val AUC {best_auc:.4f} (epoch {int(history.loc[history.val_auc.idxmax(), 'epoch'])})")
    print(metrics[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
