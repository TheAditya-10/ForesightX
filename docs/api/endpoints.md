---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: API Endpoints
---

# API Endpoints

This page summarizes the public API endpoints across services. Use the OpenAPI UI (`/docs`) for full request/response schemas.

## Auth service

- `POST /auth/sign-up`
- `POST /auth/sign-in`
- `POST /auth/token/refresh`
- `POST /auth/sign-out`
- `GET /auth/me`
- `GET /oauth/google/authorize`
- `GET /oauth/google/callback`
- `GET /health`

## Data service

- `GET /price/{ticker}`
- `GET /history/{ticker}?limit=30`
- `GET /bars/{ticker}?interval=1h&limit=240`
- `GET /indicators/{ticker}`
- `GET /news/{ticker}`
- `GET /instruments/search?q=...&limit=15`
- `POST /stream/ingest`
- `WS /stream/{ticker}`
- `GET /health`

## Profile service

- `GET /portfolio/{user_id}`
- `POST /portfolio/update`
- `GET /portfolio/{user_id}/history`
- `GET /profiles/{user_id}`
- `PATCH /profiles/{user_id}`
- `POST /profiles/{user_id}/photo`
- `POST /profiles`
- `GET /risk/{user_id}`
- `GET /health`

## Orchestration service

- `POST /analyze`
- `GET /analysis/jobs`
- `GET /analysis/jobs/{job_id}`
- `GET /instruments/search?q=...&limit=15`
- `GET /health`

## Pattern service

- `POST /predict`
- `GET /health`

Each FastAPI service publishes interactive OpenAPI documentation at its service-level `/docs` route in development. In production, expose those routes only when the access policy allows it.
