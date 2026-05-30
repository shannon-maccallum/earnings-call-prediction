# Earnings Call Prediction

Predicting post-earnings stock returns from earnings call transcript language using deep learning (FinBERT / transformer-based models).

## Project Overview

This project investigates whether the specific wording and tone used by executives during earnings calls can predict short-term stock returns. We combine Alpha Vantage earnings call transcripts with Yahoo Finance price data to create a labeled dataset, then train a transformer-based model to predict post-call returns.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/shannon-maccallum/earnings-call-prediction.git
cd earnings-call-prediction
pip install -e .
```

### 2. Configure API key

Create a `.env` file in the project root:

```
ALPHA_VANTAGE_API_KEY=your_key_here
```

Get a free key at https://www.alphavantage.co/support/#api-key

### 3. Run the data demo notebook

```bash
jupyter notebook notebooks/data_demo.ipynb
```

## Repository Structure

```
earnings-call-prediction/
├── earnings_transcript_predictor/   # Main Python package
│   ├── data/
│   │   ├── transcript_loader.py     # Alpha Vantage transcript fetcher
│   │   ├── price_loader.py          # Yahoo Finance return calculator
│   │   └── dataset.py               # PyTorch Dataset class
│   ├── models/
│   │   └── transcript_classifier.py # FinBERT-based model
│   └── utils/
│       └── helpers.py               # Shared utilities
├── notebooks/
│   └── data_demo.ipynb              # Demo: load transcripts + returns
├── data/
│   └── examples/                    # Example data points (no API key needed)
│       └── example_transcripts.json
├── tests/
├── setup.py
└── README.md
```

## Model Inputs and Targets

- **Input:** Earnings call transcript text (tokenized, fed into FinBERT)
- **Target:** Post-earnings stock return (float), defined as the % price change from market close on earnings day to close N days later (configurable: 1, 3, or 5 days)

## Evaluation

Model performance is measured using:
- **MSE / RMSE** on held-out transcripts
- **Directional accuracy** (did the model correctly predict up vs. down?)
- Baseline comparison against a "predict zero return" dummy model
