#!/usr/bin/env python3
"""
lstm_crypto_predict.py

Train and evaluate an LSTM that detects moments when the next 6 candles are expected
all to be green, based on the preceding 18 minutes of OHLC data.

Usage (example):
    python lstm_crypto_predict.py \
        --train_csv path/to/train_history.csv \
        --test_csv  path/to/test_24h.csv \
        --epochs 20 \
        --threshold 0.5

The script will save the trained model (h5) next to the training CSV and produce
``predictions.png`` which visualises the 24-hour test period with markers at
predicted growth points.

CSV expectations:
    The CSV must contain at least the following columns (names are case-insensitive):
        open_time, open, high, low, close, close_time
    Additional columns will be ignored. ``open_time`` and ``close_time`` can be
    either POSIX timestamps in milliseconds/seconds or ISO-8601 strings; they
    will be converted to pandas ``datetime``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


SEQ_LEN = 18         # minutes of history provided to the network
LOOKAHEAD = 6        # minutes of future to check for all-green candles
FEATURE_COLS = ["open", "high", "low", "close"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate LSTM on crypto OHLC data")
    p.add_argument("--train_csv", required=True, help="CSV file with historical training data (minute timeframe)")
    p.add_argument("--test_csv", required=True, help="24-hour CSV file for back-testing (minute timeframe)")
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs (default: 20)")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for classification (default: 0.5)")
    p.add_argument("--model_out", type=str, default="trained_lstm.h5", help="Path where the trained model will be saved")
    p.add_argument("--figure_out", type=str, default="predictions.png", help="Path of the generated plot PNG")
    return p.parse_args()


def _normalise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure time columns are pandas datetime and set index to open_time."""
    # Handle milliseconds timestamps: if year is >= 3000 assume milliseconds then convert
    for col in ("open_time", "close_time"):
        if col not in df.columns:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            # Heuristic: treat numbers > 1e12 as ms, >1e9 as s
            factor = 1e12 if df[col].max() > 1e12 else 1e9
            unit = "ms" if factor == 1e12 else "s"
            df[col] = pd.to_datetime(df[col], unit=unit)
        else:
            df[col] = pd.to_datetime(df[col])
    if "open_time" in df.columns:
        df = df.set_index("open_time")
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read CSV and return cleaned DataFrame with datetime index."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]  # make case-insensitive
    required = set([*FEATURE_COLS, "open_time", "close"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} is missing required columns: {missing}")
    df = _normalise_timestamps(df)
    df = df.sort_index()
    # Keep only needed columns for model. (Store original close for plotting later.)
    df[FEATURE_COLS] = df[FEATURE_COLS].astype(float)
    return df


def build_sequences(df: pd.DataFrame, seq_len: int, lookahead: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create supervised learning samples.

    Returns:
        X: (samples, seq_len, features)
        y: (samples,) binary labels
    """
    ohlc = df[FEATURE_COLS].values
    opens = df["open"].values
    closes = df["close"].values

    sequences = []
    labels = []
    total = len(df)
    for i in range(seq_len, total - lookahead):
        seq = ohlc[i - seq_len : i]
        future_open = opens[i : i + lookahead]
        future_close = closes[i : i + lookahead]
        all_green = np.all(future_close > future_open)
        sequences.append(seq)
        labels.append(1 if all_green else 0)
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


def scale_data(train_X: np.ndarray, test_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features using StandardScaler fitted on training data."""
    # Reshape for scaler: treat each feature column separately across all timesteps.
    nsamples, seq_len, nfeat = train_X.shape
    scaler = StandardScaler()
    train_X_2d = train_X.reshape(-1, nfeat)
    test_X_2d = test_X.reshape(-1, nfeat)
    scaler.fit(train_X_2d)
    train_scaled = scaler.transform(train_X_2d).reshape(nsamples, seq_len, nfeat)
    test_scaled = scaler.transform(test_X_2d).reshape(test_X.shape[0], seq_len, nfeat)
    return train_scaled, test_scaled, scaler


def build_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(train_csv: str | Path, epochs: int, batch_size: int, model_out: str) -> Tuple[Sequential, StandardScaler, pd.DataFrame]:
    df_train = load_csv(train_csv)
    X, y = build_sequences(df_train, SEQ_LEN, LOOKAHEAD)
    # Split further for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale
    X_train_scaled, X_val_scaled, scaler = scale_data(X_train, X_val)

    model = build_model((SEQ_LEN, len(FEATURE_COLS)))
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=2)
    model.save(model_out)
    return model, scaler, df_train


def predict_on_test(model: Sequential, scaler: StandardScaler, test_csv: str | Path, threshold: float, figure_out: str):
    df_test = load_csv(test_csv)
    X_test, y_test = build_sequences(df_test, SEQ_LEN, LOOKAHEAD)
    # Scale
    X_test_scaled = scaler.transform(X_test.reshape(-1, len(FEATURE_COLS))).reshape(X_test.shape)

    probs = model.predict(X_test_scaled, verbose=0).flatten()
    preds = (probs >= threshold).astype(int)

    # Evaluation metrics (optional print)
    if y_test.size > 0:
        accuracy = (preds == y_test).mean()
        print(f"Test accuracy: {accuracy:.4f} (threshold={threshold})")

    # Build list of timestamps corresponding to predictions (offset by SEQ_LEN)
    pred_times = df_test.index[SEQ_LEN : SEQ_LEN + len(preds)]
    df_plot = df_test.iloc[: len(df_test)]  # copy for plotting

    # Prepare additional plot of prediction markers
    addplots = []
    if preds.sum() > 0:
        marker_times = pred_times[preds == 1]
        marker_prices = df_plot.loc[marker_times, "close"]
        ap = mpf.make_addplot(marker_prices, type="scatter", markersize=70, marker="^", color="g")
        addplots.append(ap)

    mpf.plot(
        df_plot,
        type="candle",
        style="charles",
        title="Test period with predicted 6-minute growth points",
        addplot=addplots,
        figsize=(16, 8),
        savefig=dict(fname=figure_out, dpi=150)
    )
    print(f"Saved figure to {figure_out}")


def main():
    args = parse_args()
    print("Loading and training on", args.train_csv)
    model, scaler, _ = train_model(args.train_csv, args.epochs, args.batch_size, args.model_out)
    print("Training finished. Saved model to", args.model_out)

    print("Predicting on", args.test_csv)
    predict_on_test(model, scaler, args.test_csv, args.threshold, args.figure_out)


if __name__ == "__main__":
    main()