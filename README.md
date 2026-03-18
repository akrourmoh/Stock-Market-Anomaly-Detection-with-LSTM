# S&P 500 Anomaly Detection with LSTM Autoencoder

A deep learning project that uses an LSTM Autoencoder to detect anomalies in the S&P 500 index closing prices. The model learns to reconstruct normal market behaviour and flags time windows with unusually high reconstruction error as anomalies.

---

## Project Structure

```
LSTM/
├── app.ipynb                # Main notebook (full pipeline)
├── S&P_500_Index_Data.csv   # Historical S&P 500 daily closing prices
└── README.md
```

---

## How It Works

The core idea behind LSTM Autoencoder anomaly detection:

1. Train the autoencoder **only on normal data** (pre-2008 market)
2. The model learns to compress and reconstruct normal price patterns
3. At inference time, **anomalous windows** produce high reconstruction error (MAE) because the model has never seen that kind of behaviour
4. A threshold (mean + 3σ of training errors) separates normal from anomalous

---

## Notebook Walkthrough

| Section | Description |
|---|---|
| 1. Import Libraries | numpy, pandas, matplotlib, scikit-learn, TensorFlow/Keras |
| 2. Load & Inspect Data | Load CSV, parse dates, plot full price history |
| 3. Data Preprocessing | MinMax scaling, sliding windows of length 30 |
| 4. Train/Test Split | Pre-2008 = training (normal), 2008+ = test (includes crises) |
| 5. Build LSTM Autoencoder | Encoder → LSTM(64) → RepeatVector → Decoder → LSTM(64) → Dense |
| 6. Train | 50 epochs max, early stopping on val_loss (patience=5) |
| 7. Evaluate | Loss curves, reconstruction error distribution, threshold |
| 8. Detect Anomalies | Flag windows where MAE > threshold, plot on price chart |

---

## Model Architecture

```
Input (30, 1)
    └── LSTM(64)              ← Encoder
    └── RepeatVector(30)      ← Bottleneck
    └── LSTM(64)              ← Decoder
    └── TimeDistributed(Dense(1))
```

- **Loss:** Mean Absolute Error (MAE)
- **Optimizer:** Adam
- **Anomaly threshold:** mean + 3 × std of training reconstruction errors

---

## Results

| Metric | Value |
|---|---|
| Best Train MAE | 0.004608 |
| Best Val MAE | 0.005937 |
| Epochs trained | 25 |

The small gap between train and validation MAE confirms the model generalizes well without overfitting. The anomaly detector successfully identifies major market disruptions such as the **2008 financial crisis** and the **2020 COVID-19 crash**.

---

## Requirements

```
tensorflow
scikit-learn
pandas
numpy
matplotlib
```

Install with:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

---

## Usage

1. Clone the repository
2. Install dependencies
3. Open `app.ipynb` in Jupyter or VS Code
4. Run all cells top to bottom

---

## Dataset

The dataset contains daily closing prices of the S&P 500 index from **January 1986** to present (~8,000 trading days), with two columns:

| Column | Description |
|---|---|
| `date` | Trading date |
| `close` | Closing price |
