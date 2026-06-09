"""Evaluate a saved FinBERT return predictor checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from earnings_transcript_predictor.dataset import EarningsDataset, load_records
from earnings_transcript_predictor.evaluation import regression_metrics, signal_accuracy
from earnings_transcript_predictor.models import EarningsModel


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(args.data_dir)
    dataset = EarningsDataset(records, max_length=args.max_length)

    n_test = max(1, int(len(dataset) * args.test_size))
    n_train = len(dataset) - n_test
    generator = torch.Generator().manual_seed(args.seed)
    _, test_ds = random_split(dataset, [n_train, n_test], generator=generator)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="notebooks/data/transcripts")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output_path", type=str, default="checkpoints/evaluation.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 1.0, 2.0, 3.0])
    main(parser.parse_args())
