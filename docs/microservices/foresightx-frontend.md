---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: ForesightX-frontend
---

# ForesightX-frontend

## Purpose

User-facing UI for the platform, providing dashboards for analysis, explainability, portfolio, pattern lab, and alerts.

## Tech stack

- React + Vite + Tailwind
- Proxy-based API routing via `/api/*`

## Service routing (proxy)

- `/api/auth/*` → auth service
- `/api/orchestration/*` → orchestration service
- `/api/data/*` → data service
- `/api/profile/*` → profile service
- `/api/pattern/*` → pattern service

## Build & run

- `npm run dev` (Vite dev)
- `npm run build` (production)

## Deployment notes

- Docker build uses `VITE_*_URL` env vars for service endpoints
- Production is served via nginx container
