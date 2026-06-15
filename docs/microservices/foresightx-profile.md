---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: ForesightX-profile
---

# ForesightX-profile

## Purpose

Owns user financial context: profiles, risk level, cash balance, and portfolio holdings.

## Responsibilities

- Persist profiles and portfolio positions
- Compute total portfolio value and P&L
- Provide risk-level data for orchestration
- Accept portfolio updates for BUY/SELL flows

## API endpoints

- `GET /portfolio/{user_id}`
- `POST /portfolio/update`
- `GET /portfolio/{user_id}/history`
- `GET /profiles/{user_id}`
- `PATCH /profiles/{user_id}`
- `POST /profiles/{user_id}/photo`
- `POST /profiles`
- `GET /risk/{user_id}`
- `GET /health`

## Dependencies

- Postgres (service-owned)
- Data service for price lookups
- Optional S3 for avatar storage

## Deployment notes

- Configure via `.env` in the service root
- Key env vars: `DATABASE_URL`, `DATA_SERVICE_URL`, `SEED_DEMO_DATA`
- Default port: `8002`
