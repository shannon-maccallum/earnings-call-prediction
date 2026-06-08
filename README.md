# Earnings Call Transcript Predictor

Predicting post-earnings stock returns from earnings call transcripts using FinBERT.

**Shannon Maccallum | Intro to Deep Learning | University of Oregon**

---

## Project Purpose

Stock prices react immediately to earnings calls — but can we predict that reaction before the market fully prices it in? Executives choose their words carefully. Their tone, emphasis, and language around guidance, growth, and risk all carry signals — but those signals are buried in unstructured text.

This project investigates whether the specific wording used by executives during earnings calls can predict short-term stock returns using deep learning. The approach fine-tunes FinBERT — a version of BERT pre-trained on 4.9 billion tokens of financial text — to predict the 3-day post-earnings return percentage directly from transcript text. The model then converts predictions into Buy/Sell/Hold trading signals.

This interests me because finance is deeply human. Markets move on expectations, confidence, and fear — not just numbers. I want to quantify that human layer. The longer-term vision is a live interface that ingests transcripts in real time and generates trading signals as earnings calls happen.

---

## Dataset

### Sources
- **Transcripts:** Alpha Vantage API (`EARNINGS_CALL_TRANSCRIPT` endpoint), fetched and cached locally as JSON after the first call to avoid re-hitting the 25/day free tier limit.
- **Returns:** Yahoo Finance via `yfinance`, computing the 3-day post-earnings return: `(price_day3 - price_day0) / price_day0 × 100`

### Statistics
- **148 transcripts** across 25 S&P 500 companies
- **Date range:** 2022 Q3 through 2024 Q3
- **Sectors:** Tech (AAPL, MSFT, GOOGL, NVDA, META), Finance (JPM, BAC, GS, V, MA), Healthcare (JNJ, PFE, UNH, ABBV, MRK), Consumer (AMZN, TSLA, MCD, NKE, SBUX), Energy/Industrial (XOM, CVX, HD, HON, UPS)

### Target Variable
`return_pct` — the percentage change in closing price from earnings day to 3 trading days later. This is a continuous float (e.g. `2.45` or `-3.71`). The 3-day window was chosen to capture short-term drift after the call while avoiding too much noise from unrelated market events.

### Data Location
Transcripts are stored in `notebooks/data/transcripts/` as individual JSON files named `{TICKER}_{QUARTER}.json`.

### How to Collect Data
```bash
conda activate earnings-call-prediction
jupyter notebook notebooks/data_demo.ipynb
```

---

## Model Architecture

### FinBERT Backbone
FinBERT (ProsusAI) is BERT pre-trained on 4.9 billion tokens of financial text — news articles, earnings filings, and analyst reports. It understands earnings call language out of the box: terms like "margin compression", "sequential growth", and "guidance revision" are already encoded in its weights.

### From Sentiment Classifier to Return Predictor
FinBERT was originally designed to classify text as positive, negative, or neutral sentiment. To repurpose it for return prediction, we intercept the rich internal representation it builds before making that classification decision.

Specifically, FinBERT's encoder produces a **768-dimensional vector** for the special `[CLS]` token — a numerical fingerprint of the entire transcript, where each of the 768 dimensions captures some learned aspect of the financial language. We then apply a single **Linear(768→1)** layer on top of this vector.

This linear layer performs a matrix multiplication: it has 768 learned weights (one per dimension of the transcript vector) plus a bias term. The prediction is computed as:

```
predicted_return = w[0]*v[0] + w[1]*v[1] + ... + w[767]*v[767] + bias
```

Where `v` is the 768-dim transcript vector and `w` are the learned weights. During training the model learns which combinations of those 768 dimensions are predictive of post-earnings returns — for example, high values in dimensions that encode confident forward guidance language might push the prediction positive, while dimensions encoding cautious risk language push it negative.

### Training: Gradual Unfreezing
Standard fine-tuning failed on this small dataset — the model either overfitted badly at higher epochs or predicted near-zero for everything. The solution was gradual unfreezing (Howard & Ruder, 2018):

**Phase 1 (2 epochs, lr=1e-3):** Freeze all of FinBERT's 110 million parameters. Train only the new Linear(768→1) layer. This forces the head to learn how to read FinBERT's existing representations without overwriting the pretrained financial language knowledge.

**Phase 2 (3 epochs, lr=2e-5):** Unfreeze all parameters. Fine-tune the full model at a very low learning rate. FinBERT makes small targeted adjustments to its representations for our specific task rather than relearning from scratch.

### Optimizer: AdamW
AdamW is an optimizer that adjusts how much each of the model's 110 million parameters updates after every batch, maintaining a separate adaptive learning rate per parameter and applying weight decay (0.01) to prevent overfitting on our small dataset. Gradient clipping (max norm 1.0) is applied before each weight update to prevent unstable training during FinBERT fine-tuning.

### Loss Function
Mean Squared Error (MSE) — measures how far predicted return percentages are from actual returns. MSE penalizes large errors more than small ones, which is appropriate here since large prediction errors are especially costly for trading decisions.

---

## How to Train

### Setup
```bash
git clone https://github.com/shannon-maccallum/earnings-call-prediction.git
cd earnings-call-prediction
pip install -e .
```

### Train locally
```bash
python scripts/train_model.py \
    --data_dir notebooks/data/transcripts \
    --output_dir checkpoints/
```

### Train on Talapas (A100 GPU)
```bash
sbatch train.sh
squeue -u smaccall       # monitor job
tail -f logs/train_*.out  # view output
```

### Trained model weights
```
/gpfs/home/smaccall/earnings-call-prediction/checkpoints/best_model.pt
```

---

## Results

### Key Metrics (test set, n=29, 80/20 split)

| Metric | Best Run | Typical Range |
|--------|----------|---------------|
| Signal Accuracy (±0.5%) | **83%** | 45–83% |
| Directional Accuracy | **90%** | 59–90% |
| Mean Absolute Error | **1.57%** | 1.57–4.13% |
| RMSE | **2.04%** | 2.04–6.24% |
| Prediction Bias | **0.35%** | varies |

Random baseline = 50%. The best run significantly beats random; typical runs beat random at wider thresholds.

### Signal Accuracy by Threshold (best run)

| Threshold | Accuracy |
|-----------|----------|
| ±0.5% | **83%** |
| ±1.0% | 76% |
| ±2.0% | 62% |
| ±3.0% | 69% |

### Visualization
See `notebooks/evaluation.ipynb` for all plots including:
- Predicted vs actual return scatter plot
- Per-sample actual vs predicted bar chart
- Distribution of prediction errors
- Signal accuracy by threshold bar chart

---

## Limitations and Discussion

**Small dataset:** 148 transcripts with a 29-sample test set is the most significant limitation. Results vary substantially across runs (45–83% signal accuracy) because with only 29 test samples, a few different samples in the split can swing accuracy by 15–20 percentage points. This makes it difficult to draw firm conclusions about generalization. A minimum of 1,000+ samples would be needed for stable evaluation.

**Data collection issues:** 74 transcripts were accidentally saved as empty API error responses early in collection before a validity check was added. These were detected and removed, but reduced the effective dataset size significantly. More careful pipeline design from the start would have avoided this.

**Return window:** The 3-day return window captures short-term drift but may include noise from market-wide events unrelated to the earnings call. If the call happens after market close, same-day price reactions are partially captured in the day 0 close already.

**No consensus estimates:** The model predicts raw return, not surprise relative to analyst expectations. A stock can drop even after a strong earnings beat if expectations were too high. Incorporating EPS surprise (actual minus estimated EPS) as a second input would likely improve signal quality significantly — this is the most important next step.

**Truncation:** FinBERT accepts a maximum of 512 tokens, but full earnings call transcripts are typically 5,000–15,000 words. Only the first ~400 words are used, which captures the operator introduction and early CEO remarks but misses the CFO financial details and analyst Q&A — sections that may contain important signals.

**Future work:** Scale data collection via SEC EDGAR scraping (thousands of transcripts, free), add analyst consensus EPS as a second input feature, extend to full transcript length using a sliding window or hierarchical model, and build a live interface that ingests transcripts in real time.

