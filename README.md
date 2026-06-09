# Earnings Call Transcript Predictor

**Shannon Maccallum | Intro to Deep Learning | University of Oregon**

This project tests whether language from earnings call transcripts can predict a stock's short-term post-earnings return. The model reads transcript text and predicts the stock's 3-trading-day return percentage. The prediction can also be converted into a simple Buy, Sell, or Hold signal.

The main conclusion is cautious: FinBERT can be fine-tuned for this task, but the dataset is small, so results vary a lot depending on the train/test split. The best observed split looked strong, but the result is not stable enough to claim reliable trading performance.

## Repository Structure

```text
README.md
.gitignore
dataset.py
models.py
train_models.py
notebooks/
  data collection.ipynb
  data_demo.ipynb
  evaluation.ipynb
  model.ipynb
  data/
    examples/example_transcripts.json
    transcripts/*.json
```

## What Each Main File Does

`dataset.py`

Contains the dataset code and non-standard data loaders:

- `EarningsDataset`: PyTorch dataset that tokenizes transcripts for FinBERT.
- `load_records`: loads transcript JSON files from `notebooks/data/transcripts/`.
- `TranscriptLoader`: fetches transcripts from Alpha Vantage and caches them locally.
- `PriceLoader`: fetches post-earnings stock returns from Yahoo Finance.

`models.py`

Contains the model:

- `EarningsModel`: FinBERT encoder plus a linear regression head that predicts one number, `return_pct`.

`train_models.py`

Contains training and evaluation helpers:

- reproducible train/test splitting
- gradual unfreezing training loop
- checkpoint saving
- MAE, RMSE, bias, directional accuracy, and Buy/Sell/Hold signal accuracy

`notebooks/evaluation.ipynb`

Loads a trained checkpoint, evaluates it on the held-out test split, and creates the final result charts.

`notebooks/`

Contains the original project notebooks and data:

- `data collection.ipynb`: transcript collection and cleanup work
- `data_demo.ipynb`: data pipeline demonstration
- `model.ipynb`: notebook version of model training
- `data/transcripts/`: 173 earnings call transcript JSON files
- `data/examples/`: small example data for demos

## Dataset

The dataset contains 173 earnings call transcripts from large public companies across technology, finance, healthcare, consumer, and industrial/energy sectors. The files cover 2022 Q3 through 2024 Q3.

Each JSON file contains:

- `symbol`
- `quarter`
- `earnings_date`
- `transcript`
- `return_pct`

The target variable is:

```text
return_pct = (close_price_day3 - close_price_day0) / close_price_day0 * 100
```

## Training

Install the needed packages in your Python environment, then run:

```bash
python train_models.py \
  --mode train \
  --data_dir notebooks/data/transcripts \
  --output_dir checkpoints \
  --seed 42
```

The best model is saved to:

```text
checkpoints/best_model.pt
```

Training history is saved to:

```text
checkpoints/training_history.json
```

These output files are intentionally ignored by Git so GitHub stays clean.

## Evaluation

You can evaluate with the notebook:

```text
notebooks/evaluation.ipynb
```

Or from the command line:

```bash
python train_models.py \
  --mode evaluate \
  --data_dir notebooks/data/transcripts \
  --model_path checkpoints/best_model.pt \
  --seed 42
```

## Results

Because the dataset is small, results are unstable across runs. The best observed run looked strong, but repeated runs showed that performance depends heavily on the random split.

| Metric | Best Observed Run | Typical Observed Range |
|---|---:|---:|
| Signal Accuracy (+/-0.5%) | 83% | 45-83% |
| Directional Accuracy | 90% | 59-90% |
| MAE | 1.57% | 1.57-4.13% |
| RMSE | 2.04% | 2.04-6.24% |

## Limitations

The dataset is too small for stable generalization. FinBERT has many parameters, and 173 labeled examples is not enough to support a strong real-world trading claim.

The model also only sees the first 512 tokens of each transcript, so it may miss important information from later prepared remarks and analyst Q&A.

Overall, this project shows a working deep learning pipeline for earnings-call-based return prediction, but the current experiment is underpowered and would need more data and better financial controls before being used beyond a class project.
