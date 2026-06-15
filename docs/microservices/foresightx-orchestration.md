---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: ForesightX-orchestration
---

# ForesightX-orchestration

## Purpose

Decision engine for the platform. It coordinates multi-service analysis, applies risk checks, and returns explainable recommendations.

## Responsibilities

- Accept analysis requests and run a controlled workflow
- Call data, pattern, and profile services in parallel
- Use Gemini for structured decisions with deterministic fallback
- Persist analysis jobs and ordered trace events

## Tech stack

- FastAPI + LangGraph workflow
- Postgres for orchestration job state
- Gemini API for structured decisioning

## API endpoints

- `POST /analyze`
- `GET /instruments/search?q=...&limit=15`
- `GET /analysis/jobs`
- `GET /analysis/jobs/{job_id}`
- `GET /health`

## Workflow nodes

- `event_node`
- `data_fetch_node`
- `analysis_node`
- `decision_node`
- `risk_check_node`
- `response_node`

## Dependencies

- Data service for price, indicators, sentiment, and pattern predictions
- Profile service for portfolio context
- Pattern service for model predictions

## Deployment notes

- Configure via `.env` in the service root
- Key env vars: `DATABASE_URL`, `DATA_SERVICE_URL`, `PROFILE_SERVICE_URL`, `PATTERN_SERVICE_URL`, `GEMINI_API_KEY`
- Default port: `8000`
