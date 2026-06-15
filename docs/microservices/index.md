---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: Service Catalog
slug: /microservices
description: Responsibilities and contracts of the ForesightX services.
---

# Microservices

ForesightX splits responsibilities by business capability. Each backend service is a FastAPI application with its own configuration, health check, tests, and deployment lifecycle.

| Service | Responsibility | Public route prefix |
| --- | --- | --- |
| [Auth](/docs/microservices/foresightx-auth) | Registration, login, OAuth, JWT lifecycle, session revocation | `/api/auth` |
| [Data](/docs/microservices/foresightx-data) | Quotes, history, indicators, news, search, and streaming | `/api/data` |
| [Pattern](/docs/microservices/foresightx-pattern) | Feature preparation and model inference | `/api/pattern` |
| [Profile](/docs/microservices/foresightx-profile) | User details, portfolio, transactions, and risk | `/api/profile` |
| [Orchestration](/docs/microservices/foresightx-orchestration) | Multi-service analysis and recommendation workflow | `/api/orchestration` |
| [Frontend](/docs/microservices/foresightx-frontend) | Responsive React application and session-aware UI | `/` |

## Boundary rules

- A service owns its tables and migrations.
- Cross-service reads happen through APIs, not shared database credentials.
- External provider formats are normalized by the service that owns that integration.
- The frontend calls same-origin NGINX routes rather than container names.
- Optional integrations should fail explicitly or use documented fallback behavior.

## Common conventions

- `GET /health` for liveness and readiness checks
- Pydantic schemas for request and response validation
- Environment variables for deployment-specific configuration
- Structured HTTP errors at service boundaries
- Pytest for backend verification and Vitest for frontend components
