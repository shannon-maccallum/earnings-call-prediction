# Earnings Call Transcript Predictor

**Shannon Maccallum | Intro to Deep Learning | University of Oregon**

This project investigates whether the language in earnings call transcripts can predict short-term stock returns. The model takes an earnings call transcript as text and predicts the stock's 3-trading-day post-earnings return percentage. The final prediction is also converted into a simple trading signal: Buy, Sell, or Hold.

The main result is honest rather than flashy: FinBERT can be fine-tuned for this task, and some train/test splits produce strong-looking results, but the dataset is too small for stable generalization. The first run I observed had very high signal and directional accuracy, but later runs were much weaker. That variance is an important conclusion of the project, not just a failure. With only 173 examples, a 34-sample test set can change dramatically depending on which examples land in it.

## Project Purpose

Earnings calls are one of the places where financial information and human language meet. Executives describe growth, risk, guidance, margins, demand, and uncertainty in ways that may contain signal beyond the raw numbers. This project asks whether a deep learning model can learn that signal from transcript text and predict the immediate stock reaction after earnings.

I use FinBERT because it is a BERT model pretrained on financial text. Instead of using FinBERT for sentiment classification, I replace the sentiment classifier with a regression head that predicts a continuous return percentage. This makes the project a supervised regression problem:

```text
earnings call transcript -> FinBERT encoder -> Linear regression head -> return_pct
```

## Dataset

The dataset contains **173 earnings call transcripts** from large public companies across technology, finance, healthcare, consumer, and industrial/energy sectors. The files cover **2022 Q3 through 2024 Q3**. Each example is stored as a JSON file in:

```text
notebooks/data/transcripts/
```

Each JSON file contains:

- `symbol`: ticker symbol
- `quarter`: earnings quarter
- `earnings_date`: date used for the return calculation
- `transcript`: Alpha Vantage transcript speaker turns
- `return_pct`: target variable

Transcripts were collected from the Alpha Vantage `EARNINGS_CALL_TRANSCRIPT` endpoint and cached locally to avoid repeated API calls. Returns were collected with `yfinance`. The target is:

```text
return_pct = (close_price_day3 - close_price_day0) / close_price_day0 * 100
```

The median transcript length is about 8,100 words, but FinBERT can only accept 512 tokens. This means the current model uses the beginning of each transcript, which usually includes the operator introduction and early prepared remarks.

## Repository Structure

```text
earnings_transcript_predictor/
  dataset.py        # Dataset and transcript loading utilities
  models.py         # FinBERT regression model
  data_loaders.py   # Alpha Vantage and yfinance helpers
  evaluation.py     # Shared metrics

dataset.py          # Compatibility wrapper for assignment requirements
models.py           # Compatibility wrapper for assignment requirements
train_model.py      # Training script
evaluate_model.py   # Evaluation script
notebooks/data_demo.ipynb
notebooks/model.ipynb
evaluation.ipynb
```

## Setup

```bash
git clone https://github.com/shannon-maccallum/earnings-call-prediction.git
cd earnings-call-prediction
python -m venv venv
source venv/bin/activate
pip install -e .
```

On Talapas, the project path is:

```text
/gpfs/home/smaccall/earnings-call-prediction/
```

## Training

The training script uses an 80/20 split with a fixed random seed. By default, it uses gradual unfreezing:

1. Freeze FinBERT and train only the new regression head for 2 epochs.
2. Unfreeze FinBERT and fine-tune the full model for 3 epochs with a small learning rate.

This is more appropriate for a small dataset than immediately updating all 110M FinBERT parameters.

Run locally or on an interactive GPU node:

```bash
python train_model.py \
  --data_dir notebooks/data/transcripts \
  --output_dir checkpoints/ \
  --seed 42
```

Run on Talapas:

```bash
sbatch train.sh
```

The trained model weights are saved to:

```text
checkpoints/best_model.pt
/gpfs/home/smaccall/earnings-call-prediction/checkpoints/best_model.pt
```

The training history is saved to:

```text
checkpoints/training_history.json
```

## Evaluation

Evaluate a trained checkpoint with:

```bash
python evaluate_model.py \
  --data_dir notebooks/data/transcripts \
  --model_path checkpoints/best_model.pt \
  --seed 42
```

The evaluation notebook, `evaluation.ipynb`, loads the trained model, runs predictions on the held-out test set, computes metrics, and creates the plots used in this README.

The proposed metrics are:

- **MAE**: mean absolute error in return percentage points
- **RMSE**: root mean squared error
- **Prediction bias**: average prediction error
- **Directional accuracy**: whether predicted and actual returns have the same sign
- **Signal accuracy**: whether predicted and actual Buy/Sell/Hold labels match at thresholds such as +/-0.5%, +/-1%, and +/-2%

## Results

Because the dataset is small, results are unstable across runs. The best observed run looked strong, but repeated runs showed that the result depends heavily on the random split.

| Metric | Best Observed Run | Typical Observed Range |
|---|---:|---:|
| Signal Accuracy (+/-0.5%) | 83% | 45-83% |
| Directional Accuracy | 90% | 59-90% |
| MAE | 1.57% | 1.57-4.13% |
| RMSE | 2.04% | 2.04-6.24% |

The best run should be interpreted as a best-case scenario, not as the model's expected real-world accuracy. A 34-sample test set is small enough that a few lucky or unlucky examples can swing the accuracy by a large amount.

## Visualization

The evaluation notebook produces:

- predicted vs. actual return scatter plot
- actual vs. predicted bar chart for each test example
- prediction error histogram
- signal accuracy by threshold chart

After running `evaluation.ipynb`, the signal accuracy chart is saved as:

```text
signal_accuracy.png
```

## Limitations

The biggest limitation is dataset size. FinBERT is a large pretrained model, but 173 labeled examples is still very small for supervised fine-tuning. Gradual unfreezing helps reduce overfitting, but it does not create new information. Adding 25 more transcripts helped slightly, but a truly stable evaluation would likely need hundreds or thousands more examples.

The second limitation is truncation. Full earnings calls are thousands of words, while BERT can only process 512 tokens. Important information in CFO remarks and analyst Q&A is probably excluded. A better version would use a sliding-window or hierarchical model that reads the full transcript.

The third limitation is that the target is raw stock return, not earnings surprise. A stock can fall after good earnings if investors expected even better results. Adding analyst consensus estimates, EPS surprise, revenue surprise, and market return controls would make the target more financially meaningful.

Overall, this project shows a working deep learning pipeline for earnings-call-based return prediction, but it also shows that the current dataset is not large enough to support a strong claim of predictive trading performance. The most defensible conclusion is that transcript language may contain useful information, but the current experiment is underpowered and needs more data and better financial controls before it could be used outside a class project.
