"""Train FinBERT on earnings call transcripts with reproducible splits."""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from earnings_transcript_predictor.models import EarningsModel
from earnings_transcript_predictor.dataset import load_records, EarningsDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    records = load_records(args.data_dir)
    dataset = EarningsDataset(records, max_length=512)

    n_test = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_test
    split_generator = torch.Generator().manual_seed(args.seed)
    train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=split_generator)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    print(f"Train: {n_train} | Test: {n_test}")

    model = EarningsModel(dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    os.makedirs(args.output_dir, exist_ok=True)
    history = []

    if args.gradual_unfreeze:
        model.freeze_bert()
        optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=args.head_lr)
        history.extend(
            {
                "phase": "head_only",
                **row,
            }
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="notebooks/data/transcripts")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--head_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--no_gradual_unfreeze",
        dest="gradual_unfreeze",
        action="store_false",
        help="Skip the frozen-head phase and fine-tune all parameters immediately.",
    )
    parser.set_defaults(gradual_unfreeze=True)
    args = parser.parse_args()
    main(args)
