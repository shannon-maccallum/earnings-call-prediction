# Earnings Call Transcript Predictor

Predicting post-earnings stock returns from earnings call transcripts using FinBERT, a transformer model pre-trained on financial text.

**Shannon Maccallum | Intro to Deep Learning | University of Oregon**

---

## Project Purpose

Stock prices react immediately to earnings calls — but can we predict that reaction before the market fully prices it in? Executives choose their words carefully, and their tone, emphasis, and language around guidance, growth, and risk all carry signals. This project investigates whether the specific wording used by executives during earnings calls can predict short-term stock returns using deep learning.

The approach fine-tunes FinBERT — a version of BERT pre-trained on 4.9 billion tokens of financial text — to predict the 3-day post-earnings return percentage directly from transcript text. The model generates Buy/Sell/Hold trading signals, achieving **83% signal accuracy at a ±0.5% threshold** and **90% directional accuracy**.

---

## Dataset

### Sources
- **Transcripts:** Alpha Vantage API (`EARNINGS_CALL_TRANSCRIPT` endpoint), cached locally as JSON after first fetch to avoid re-hitting the 25/day free tier limit.
- **Returns:** Yahoo Finance via `yfinance`, computing the 3-day post-earnings return: `(price_day3 - price_day0) / price_day0 × 100`

### Statistics
- **148 transcripts** across 25 S&P 500 companies
- **Date range:** 2022 Q3 through 2024 Q3
- **Sectors:** Tech (AAPL, MSFT, GOOGL, NVDA, META), Finance (JPM, BAC, GS, V, MA), Healthcare (JNJ, PFE, UNH, ABBV, MRK), Consumer (AMZN, TSLA, MCD, NKE, SBUX), Energy/Industrial (XOM, CVX, HD, HON, UPS)

### Target Variable
`return_pct` — the percentage change in closing price from earnings day to 3 trading days later. This is a continuous float (e.g. `2.45` or `-3.71`).

### Data Location
Transcripts are stored in `notebooks/data/transcripts/` as individual JSON files named `{TICKER}_{QUARTER}.json`. Each file contains the transcript text, return_pct label, and metadata.

### Data Collection
```bash
conda activate earnings-call-prediction
cd notebooks
jupyter notebook  # open data collection.ipynb
```

---

## How to Train

### Setup
```bash
git clone https://github.com/shannon-maccallum/earnings-call-prediction.git
cd earnings-call-prediction
pip install -e .
```

### Add your API key
```bash
echo "ALPHA_VANTAGE_API_KEY=your_key_here" > .env
```

### Train locally
```bash
python scripts/train_model.py \
    --data_dir notebooks/data/transcripts \
    --output_dir checkpoints/
```

### Train on Talapas (recommended — requires A100 GPU)
```bash
sbatch train.sh
```

Monitor job:
```bash
squeue -u smaccall
tail -f logs/train_<jobid>.out
```

### Trained model weights
The trained model weights are stored on Talapas at:
```
/gpfs/home/smaccall/earnings-call-prediction/checkpoints/best_model.pt
```

---

## Results

### Key Metrics (test set, n=29)

| Metric | Value |
|--------|-------|
| Signal Accuracy (±0.5%) | **83%** |
| Directional Accuracy | **90%** |
| Mean Absolute Error | **1.57%** |
| RMSE | **2.04%** |
| Prediction Bias | **0.35%** |

### Signal Accuracy by Threshold

| Threshold | Accuracy |
|-----------|----------|
| ±0.5% | **83%** |
| ±1.0% | 76% |
| ±1.5% | 55% |
| ±2.0% | 62% |
| ±2.5% | 59% |
| ±3.0% | 69% |

Random baseline = 50%. The model beats random across all thresholds.

### Visualizations
See `notebooks/evaluation.ipynb` for all plots including predicted vs actual returns, per-sample comparison, and error distribution.

---

## Model Architecture

FinBERT's original 3-class sentiment output is replaced with a single `Linear(768→1)` regression layer. The 768-dimensional `[CLS]` token embedding produced by FinBERT for each transcript is mapped to a single predicted return percentage.

### Gradual Unfreezing (Howard & Ruder, 2018)
Standard fine-tuning failed on this small dataset — the model either overfitted badly at higher epochs or predicted near-zero for everything. Gradual unfreezing solved this:

**Phase 1 (2 epochs, lr=1e-3):** Freeze all FinBERT parameters. Train only the new regression head. Forces the head to learn to read FinBERT's existing financial language representations without disturbing them.

**Phase 2 (3 epochs, lr=2e-5):** Unfreeze all parameters. Fine-tune the full model at a very low learning rate using AdamW (weight_decay=0.01) with gradient clipping. FinBERT makes small targeted adjustments rather than overwriting its pretrained knowledge.

This approach is critical when fine-tuning a 110M parameter model on only 148 samples.

---

## Limitations

- **Small dataset:** 148 transcripts is limited for a model of this size. Results on the test set (n=29) may not generalize reliably. Academic papers in this space typically use 1,000–10,000+ samples.
- **Return window:** The 3-day return window captures short-term drift but may include noise from unrelated market events. If the call happens after market close, same-day reactions are partially captured in day 0.
- **No consensus estimates:** The model predicts raw return, not surprise relative to analyst expectations. Incorporating EPS surprise as a second input would likely improve signal quality significantly.
- **Data collection errors:** 74 transcripts were accidentally saved as empty API error responses early in collection. These were detected and removed, but reduced the effective dataset size.

## Future Work

- **Scale data via SEC EDGAR scraping** — 8-K filings contain earnings transcripts, are completely free, and would enable thousands of training samples.
- **Add analyst consensus EPS** as a second model input to predict surprise, not just raw return.
- **Build a live interface** that ingests transcripts in real time and generates buy/sell/hold signals as earnings calls happen.

---

## Repository Structure

```
earnings-call-prediction/
├── earnings_transcript_predictor/
│   ├── models.py          # EarningsModel (FinBERT + regression head)
│   ├── dataset.py         # EarningsDataset + load_records()
│   └── __init__.py
├── scripts/
│   └── train_model.py     # Training script with gradual unfreezing
├── notebooks/
│   ├── data_demo.ipynb    # Dataset exploration and loading demo
│   ├── evaluation.ipynb   # Model evaluation and all result plots
│   ├── model.ipynb        # Full training + results notebook
│   └── data/transcripts/  # 148 JSON transcript files
├── train.sh               # SLURM job script for Talapas
├── setup.py
└── README.md
```
