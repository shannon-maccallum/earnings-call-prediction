"""
train_model.py

Trains EarningsModel on earnings call transcripts using gradual unfreezing.

Gradual unfreezing strategy (Howard & Ruder, 2018):
    Phase 1: Freeze FinBERT, train only the regression head (lr=1e-3, 2 epochs)
    Phase 2: Unfreeze all parameters, fine-tune full model (lr=2e-5, 3 epochs)

This overcomes catastrophic forgetting when fine-tuning a 110M parameter model
on a small dataset (~148 transcripts).

Usage:
    python scripts/train_model.py --data_dir notebooks/data/transcripts --output_dir checkpoints/

On Talapas (SLURM):
    sbatch train.sh
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from earnings_transcript_predictor.models import EarningsModel
from earnings_transcript_predictor.dataset import load_records, EarningsDataset


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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    records = load_records(args.data_dir)
    dataset = EarningsDataset(records, max_length=512)

    # 80/20 train/test split
    n_test = max(1, int(len(dataset) * 0.2))
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8)
    print(f"Train: {n_train} | Test: {n_test}")

    model = EarningsModel().to(device)
    criterion = nn.MSELoss()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Phase 1: freeze FinBERT, train head only ──────────────────────────
    model.freeze_bert()
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=1e-3)
    print("\nPhase 1: Training head only (FinBERT frozen)...")
    for epoch in range(2):
        loss = train_epoch(model, train_loader, optimizer, device, criterion)
        rmse, dir_acc = eval_epoch(model, test_loader, device, criterion, n_test)
        print(f"  Epoch {epoch+1}/2 | Loss: {loss:.4f} | RMSE: {rmse:.4f} | Dir Acc: {dir_acc:.2f}")

    # ── Phase 2: unfreeze all, fine-tune full model ───────────────────────
    model.unfreeze_bert()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    print("\nPhase 2: Fine-tuning full model...")
    best_rmse = float("inf")
    for epoch in range(3):
        loss = train_epoch(model, train_loader, optimizer, device, criterion)
        rmse, dir_acc = eval_epoch(model, test_loader, device, criterion, n_test)
        print(f"  Epoch {epoch+1}/3 | Loss: {loss:.4f} | RMSE: {rmse:.4f} | Dir Acc: {dir_acc:.2f}")
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")
            print(f"  Saved best model (RMSE={best_rmse:.4f})")

    print(f"\nDone. Best RMSE: {best_rmse:.4f}")
    print(f"Model saved to {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="notebooks/data/transcripts")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    main(args)
