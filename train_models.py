"""Train and evaluate the FinBERT earnings-call return predictor."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import EarningsDataset, load_records
from models import EarningsModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def signal(value: float, threshold: float = 0.5) -> str:
    """Convert a return percentage into a Buy/Sell/Hold signal."""
    if value > threshold:
        return "BUY"
    if value < -threshold:
        return "SELL"
    return "HOLD"


def regression_metrics(preds: Iterable[float], labels: Iterable[float]) -> Dict[str, float]:
    """Compute regression and direction metrics for predicted returns."""
    pred_arr = np.asarray(list(preds), dtype=float)
    label_arr = np.asarray(list(labels), dtype=float)
    errors = pred_arr - label_arr
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(np.mean(errors)),
        "directional_accuracy": float(np.mean((pred_arr >= 0) == (label_arr >= 0))),
    }


def signal_accuracy(
    preds: Iterable[float],
    labels: Iterable[float],
    thresholds: Iterable[float] = (0.5, 1.0, 2.0, 3.0),
) -> List[Dict[str, float]]:
    """Compute Buy/Sell/Hold agreement for several thresholds."""
    pred_arr = np.asarray(list(preds), dtype=float)
    label_arr = np.asarray(list(labels), dtype=float)
    rows = []
    for threshold in thresholds:
        correct = sum(
            signal(pred, threshold) == signal(label, threshold)
            for pred, label in zip(pred_arr, label_arr)
        )
        rows.append(
            {
                "threshold": float(threshold),
                "correct": int(correct),
                "total": int(len(pred_arr)),
                "accuracy": float(correct / len(pred_arr)),
            }
        )
    return rows


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        preds = model(input_ids, attention_mask)
        loss = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, device, criterion, n_test):
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            preds = model(input_ids, attention_mask)
            val_loss += criterion(preds, labels).item()
            correct += ((preds >= 0) == (labels >= 0)).sum().item()
    rmse = (val_loss / len(loader)) ** 0.5
    dir_acc = correct / n_test
    return rmse, dir_acc


def run_phase(model, loader, eval_loader, optimizer, device, criterion, epochs, n_test, label):
    history = []
    print(f"\n{label}")
    for epoch in range(epochs):
        loss = train_epoch(model, loader, optimizer, device, criterion)
        rmse, dir_acc = eval_epoch(model, eval_loader, device, criterion, n_test)
        history.append({"epoch": epoch + 1, "loss": loss, "rmse": rmse, "dir_acc": dir_acc})
        print(
            f"  Epoch {epoch+1}/{epochs} | "
            f"Loss: {loss:.4f} | RMSE: {rmse:.4f} | Dir Acc: {dir_acc:.2f}"
        )
    return history


def build_data_loaders(args):
    records = load_records(args.data_dir)
    dataset = EarningsDataset(records, max_length=args.max_length)
    n_test = max(1, int(len(dataset) * args.test_size))
    n_train = len(dataset) - n_test
    split_generator = torch.Generator().manual_seed(args.seed)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=split_generator)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    return train_loader, test_loader, n_train, n_test


def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    train_loader, test_loader, n_train, n_test = build_data_loaders(args)
    print(f"Train: {n_train} | Test: {n_test}")

    model = EarningsModel(dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    os.makedirs(args.output_dir, exist_ok=True)
    history = []

    if args.gradual_unfreeze:
        model.freeze_bert()
        optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=args.head_lr)
        history.extend(
            {"phase": "head_only", **row}
            for row in run_phase(
                model,
                train_loader,
                test_loader,
                optimizer,
                device,
                criterion,
                args.head_epochs,
                n_test,
                "Phase 1: Training regression head only (FinBERT frozen)...",
            )
        )
        model.unfreeze_bert()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("\nPhase 2: Fine-tuning full model...")
    best_rmse = float("inf")
    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device, criterion)
        rmse, dir_acc = eval_epoch(model, test_loader, device, criterion, n_test)
        history.append(
            {
                "phase": "full_finetune",
                "epoch": epoch + 1,
                "loss": loss,
                "rmse": rmse,
                "dir_acc": dir_acc,
            }
        )
        print(
            f"  Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {loss:.4f} | RMSE: {rmse:.4f} | Dir Acc: {dir_acc:.2f}"
        )
        if rmse < best_rmse:
            best_rmse = rmse
            model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model (RMSE={best_rmse:.4f})")

    metrics_path = os.path.join(args.output_dir, "training_history.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "n_train": n_train,
                "n_test": n_test,
                "gradual_unfreeze": args.gradual_unfreeze,
                "history": history,
                "best_rmse": best_rmse,
            },
            f,
            indent=2,
        )

    print(f"\nDone. Best RMSE: {best_rmse:.4f}")
    print(f"Model saved to {args.output_dir}/best_model.pt")
    print(f"History saved to {metrics_path}")


def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, n_train, n_test = build_data_loaders(args)

    model = EarningsModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch_preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            preds.extend(batch_preds.cpu().numpy().tolist())
            labels.extend(batch["label"].numpy().tolist())

    metrics = regression_metrics(preds, labels)
    signals = signal_accuracy(preds, labels, args.thresholds)
    results = {
        "model_path": args.model_path,
        "seed": args.seed,
        "n_train": n_train,
        "n_test": n_test,
        "metrics": metrics,
        "signal_accuracy": signals,
        "predictions": [
            {"predicted": float(pred), "actual": float(label)}
            for pred, label in zip(preds, labels)
        ],
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation to {output_path}")

    print(json.dumps({k: results[k] for k in ["seed", "n_train", "n_test", "metrics"]}, indent=2))
    print("Signal accuracy:")
    for row in signals:
        print(
            f"  +/-{row['threshold']:.1f}%: "
            f"{row['correct']}/{row['total']} = {row['accuracy']:.2f}"
        )

# make able to be run in terminal
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--data_dir", type=str, default="notebooks/data/transcripts")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output_path", type=str, default="checkpoints/evaluation.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--head_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    parser.add_argument(
        "--no_gradual_unfreeze",
        dest="gradual_unfreeze",
        action="store_false",
        help="Skip the frozen-head phase and fine-tune all parameters immediately.",
    )
    parser.set_defaults(gradual_unfreeze=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_model(args)
    else:
        evaluate_model(args)

