---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: ForesightX-data
---

# ForesightX-data

## Purpose

Market data service responsible for price/history APIs, technical indicators, news lookups, and real-time streaming.

## Responsibilities

- Fetch current prices and historical bars
- Compute RSI + MACD and persist indicator snapshots
- Normalize and store news data
- Cache responses in Redis for low-latency reads
- Provide websocket stream for tick updates

## Tech stack

- FastAPI, async SQLAlchemy
- Redis caching
- External sources via yfinance

## API endpoints

- `GET /price/{ticker}`
- `GET /history/{ticker}?limit=30`
- `GET /bars/{ticker}?interval=1h&limit=240`
- `GET /indicators/{ticker}`
- `GET /news/{ticker}`
- `GET /instruments/search?q=...&limit=15`
- `POST /stream/ingest`
- `WS /stream/{ticker}`
- `GET /health`

## I/O contract

- Input: ticker symbols and query params
- Output: typed JSON responses for price, history, indicators, news, and bars

## Data ownership

- `instruments`
- `daily_price_snapshots`
- `technical_indicator_snapshots`
- `news_articles` and `instrument_news`

## Real-time stream flow

1. Upstream adapter pushes ticks to `POST /stream/ingest`
2. Service persists tick and refreshes caches
3. Websocket clients subscribed to `WS /stream/{ticker}` receive updates

## Deployment notes

- Configure via `.env` in the service root
- Key env vars: `DATABASE_URL`, `REDIS_URL`, `CACHE_TTL_SECONDS`, `STREAM_HEARTBEAT_SECONDS`
- Default port: `8001`
