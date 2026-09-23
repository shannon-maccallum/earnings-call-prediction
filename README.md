# Earnings Call Transcript Predictor

**Shannon Maccallum | Intro to Deep Learning | University of Oregon**

This project tests whether language from earnings call transcripts can predict a stock's short-term post-earnings return. The model reads transcript text on the earnings-call date and predicts the stock's 3-trading-day return percentage. The prediction is converted into a simple Buy, Sell, or Hold signal and evaluated against what the stock actually did over the next 3 trading days.

The main conclusion is cautious: FinBERT can be fine-tuned for this task, but the dataset is small, so results vary a lot depending on the train/test split. The best observed split looked strong, but the result is not stable enough to claim reliable trading performance.

## Repository Structure

```text
README.md
.gitignore
dataset.py
models.py
train_models.py
scripts/
  build_pdf_dataset.py
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
- MAE, RMSE, bias, directional accuracy, Buy/Sell/Hold signal accuracy, and a 3-day trading-strategy backtest

`scripts/build_pdf_dataset.py`

Converts the cleaned quarterly earnings-call PDF archive into the same JSON format used by the model. It extracts transcript text from each PDF, reads the fiscal quarter and call date from the normalized filename/header, and uses Yahoo Finance to compute the 3-trading-day `return_pct` label.

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

The original model dataset contains 173 earnings call transcripts from large public companies across technology, finance, healthcare, consumer, and industrial/energy sectors. The files cover 2022 Q3 through 2024 Q3.

The expanded source archive contains cleaned quarterly earnings-call PDFs organized by ticker. To use the PDFs instead of the original Alpha Vantage JSON files, first convert them into model-ready JSON:

```bash
python scripts/build_pdf_dataset.py \
  --pdf_root ~/Library/CloudStorage/OneDrive-UniversityOfOregon/transcripts \
  --output_dir notebooks/data/pdf_transcripts \
  --return_window 3
```

On Talapas, replace `--pdf_root` with the location where you copied the cleaned PDF archive.

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

The trading signal is created from the model's predicted return:

```text
predicted_return > +0.5%  -> BUY
predicted_return < -0.5%  -> SELL
otherwise                 -> HOLD
```

The 3-day strategy return is then:

```text
BUY  -> actual 3-day stock return
SELL -> negative actual 3-day stock return, equivalent to a short position
HOLD -> 0% return
```

## Training

Install the needed packages in your Python environment, then run:

```bash
python train_models.py \
  --mode train \
  --data_dir notebooks/data/pdf_transcripts \
  --output_dir checkpoints \
  --chunks_per_transcript 6 \
  --seed 42
```

To train on the smaller original dataset instead, use `--data_dir notebooks/data/transcripts`.

The `--chunks_per_transcript 6` option is important for the PDF dataset. FinBERT can only read 512 tokens at a time, so this creates several transcript windows per earnings call and averages chunk predictions back to the call level during evaluation. Without chunking, the model mostly sees the beginning of the PDF text.

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
  --data_dir notebooks/data/pdf_transcripts \
  --model_path checkpoints/best_model.pt \
  --chunks_per_transcript 6 \
  --seed 42
```

Evaluation reports both signal accuracy and the event-driven trading backtest:

- `Signal Accuracy`: whether the predicted Buy/Sell/Hold signal matches the actual 3-day outcome label.
- `Trade Win Rate`: among non-Hold predictions, whether the long/short trade made money after 3 trading days.
- `Mean Strategy Return`: average return across all test calls, with Hold counted as 0%.
- `Compounded Strategy Return`: compounded return from following every test signal in sequence.

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
