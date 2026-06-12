---
id: architecture
title: System Architecture
description: Runtime topology, service ownership, and request flow.
---

# System Architecture

ForesightX uses focused microservices so that identity, market data, prediction, user context, and recommendation logic can evolve independently. The frontend and NGINX provide a unified product surface while the backend retains explicit ownership boundaries.

![ForesightX system architecture](/img/diagrams/system-architecture.jpg)

## Runtime topology

```text
Internet
  |
  v
NGINX reverse proxy
  |-- /                 -> React frontend
  |-- /api/auth/        -> Auth service
  |-- /api/profile/     -> Profile service
  |-- /api/data/        -> Data service
  |-- /api/pattern/     -> Pattern service
  `-- /api/orchestration -> Orchestration service
```

Backend containers do not need public host ports. They resolve one another by Docker service name and expose health endpoints for startup and operational checks.

## Service ownership

| Service | Owns | Primary dependencies |
| --- | --- | --- |
| Frontend | User experience and client session state | NGINX API routes |
| Auth | Credentials, OAuth linkage, access and refresh tokens | PostgreSQL, Redis, Profile service |
| Data | Instruments, prices, indicators, news, streaming | Market providers, PostgreSQL, Redis |
| Pattern | Feature transformation and model inference | Model artifacts, MLflow/DVC metadata |
| Profile | User details, cash, positions, transactions, risk | PostgreSQL, Data service |
| Orchestration | Analysis jobs, workflow events, final recommendation | Data, Pattern, Profile, Gemini |

## Analysis request flow

1. The client sends `POST /analyze` to the orchestration route.
2. Orchestration validates the request and creates workflow state.
3. Market data, prediction, and user context are collected through service tools.
4. Signals are normalized and combined into an analysis view.
5. Gemini can produce a structured explanation; deterministic logic remains available as a fallback.
6. Risk checks constrain the result using exposure and volatility context.
7. The response returns the action, confidence, evidence, and trace while analysis events are persisted.

## Component view

![ForesightX component diagram](/img/diagrams/component.jpg)

The component model shows a deliberate separation between presentation, API services, persistence, external providers, MLOps, and deployment infrastructure. This keeps external integrations at service boundaries instead of leaking them into the frontend.

## Reliability strategy

- Redis caches market data and supports revocable session state.
- Services validate external responses before exposing normalized contracts.
- Health checks distinguish a running process from a functioning service.
- Orchestration can degrade to deterministic recommendation logic when AI assistance is unavailable.
- Each service can be restarted or scaled without rebuilding the whole application.

## Data flow

The data service normalizes price, history, indicator, and news information. The pattern service consumes a defined historical feature window rather than reaching directly into another service's database. The profile service supplies only the portfolio and risk context required for analysis. Orchestration combines these API-level outputs and owns the resulting job trace.

This contract-first flow avoids cross-service database access and preserves independent migrations.
