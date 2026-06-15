---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: ForesightX-Pattern (data science service)
---

# ForesightX-Pattern — Data Science & Model Service

This service is the dedicated data science microservice that provides offline training and online inference.

## Business problem it solves

Generates predictive signals and confidence intervals from market time-series so downstream services can make explainable decisions.

## Responsibilities

- Offline ML pipelines: ingestion, feature engineering, training, evaluation
- Online inference: cached model loading and real-time predictions
- Artifact tracking and reproducibility via DVC + MLflow

## Tech stack

- Python, PyTorch, scikit-learn, FastAPI
- DVC + MLflow for artifact/version tracking
- Docker for service containerization

## Model architecture (current)

- Sequence encoder (LSTM/GRU) + stock embedding
- Fusion head predicts next 3 hourly closes
- MC Dropout for confidence + interval estimates

## Feature engineering

- OHLCV, returns
- RSI, EMA fast/slow, MACD (signal + histogram)
- Rolling mean/std, time features (hour/day-of-week encodings)

## Training workflow (offline)

1. Fetch + clean market data
2. Build features and sequences
3. Train foundation model across multiple tickers
4. Persist artifacts: `model.pt`, `scaler.pkl`, `metadata.json`
5. Log runs and metrics to MLflow/DagsHub

## Inference API

- `GET /health`
- `POST /predict`

Request:

```json
{
	"ticker": "RELIANCE.NS",
	"timestamp": "2026-04-19T10:00:00+05:30"
}
```

Response:

```json
{
	"ticker": "RELIANCE.NS",
	"predictions": [624.1, 626.4, 628.0],
	"confidence": 0.91,
	"intervals": [[620.0, 628.0], [622.2, 630.6], [623.5, 632.5]],
	"model_version": "7"
}
```

## Evaluation metrics

- RMSE, MAE, MAPE
- Direction accuracy and error distributions

## Limitations & assumptions

- Assumes stable market regimes; performance can degrade during structural shifts
- Requires consistent data feeds and feature integrity

## Integration points

- Called by orchestration for pattern predictions
- Uses data service for latest bars when building inference sequences

## Why a separate service

- ML training/inference is compute-heavy and benefits from independent scaling
- Keeps experimental pipelines isolated from latency-sensitive APIs
