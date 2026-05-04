---
title: Schema
---

# Database Schema Overview

This section explains the database choices and schema design rationale. The repository uses Postgres for relational storage; consider time-series optimizations where appropriate.

## Service-owned tables

Auth service:

- `users` (identity, auth provider linkage, verification flags)

Profile service:

- `users` (profile + cash balance)
- `portfolio_positions`
- `portfolio_transactions`

Data service:

- `instruments`
- `daily_price_snapshots`
- `technical_indicator_snapshots`
- `news_articles`
- `instrument_news`

Orchestration service:

- `analysis_jobs`
- `analysis_job_events`

Pattern service:

- Model metadata and inference records (plus artifacts in DVC/MLflow)

## Indexing & query patterns

- Market data: index on (instrument_ticker, observed_at) for fast range queries
- Portfolio positions: unique (user_id, ticker) for upserts
- Job events: index by (analysis_job_id, sequence_number) for trace reconstruction

Partitioning is recommended for large time-series tables as data grows.
