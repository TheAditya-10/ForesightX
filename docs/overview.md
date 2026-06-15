---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
id: overview
title: Platform Overview
description: What ForesightX solves, how it works, and where to begin.
slug: /overview
---

# ForesightX

ForesightX is a microservice-based intelligent stock analytics platform. It brings market data, technical indicators, short-horizon prediction, portfolio context, and recommendation generation into one workflow.

**Try it:** [Open the live ForesightX application](https://foresightx.apst.me).

The platform is designed as **decision support**, not as a brokerage or autonomous trading system. It helps a user understand a stock before making a personal decision and makes the evidence behind the result visible.

![ForesightX stock analysis screen](/img/screenshots/stock-analysis.png)

## The problem

Stock analysis is usually fragmented across charts, price feeds, screeners, news sources, portfolio tools, and isolated prediction experiments. Users must manually connect those signals, while research models often stop at a notebook and never become a usable product.

ForesightX closes that gap with a structured flow:

1. Retrieve live and historical market data.
2. Compute technical indicators and collect relevant news.
3. Generate next-hour and next-day prediction signals.
4. Add the user's portfolio and risk context.
5. Combine the evidence through an orchestration workflow.
6. Return a recommendation with confidence and supporting reasoning.

## Core capabilities

| Capability | What the platform provides |
| --- | --- |
| Secure access | JWT access and refresh tokens, session revocation, and Google OAuth support |
| Market intelligence | Quotes, historical bars, search, news, indicators, Redis caching, and WebSocket updates |
| Prediction | ML-based short-horizon direction forecasts with confidence signals |
| Portfolio context | User profile, positions, transactions, cash balance, and risk information |
| Recommendation | LangGraph-coordinated analysis with Gemini-assisted summaries and deterministic safeguards |
| Deployment | Containerized services, private service networking, NGINX routing, and health checks |

## System at a glance

The React frontend calls the platform through same-origin `/api/...` routes. NGINX directs requests to the appropriate FastAPI service. Each stateful service owns its data, which reduces coupling and lets prediction, user, and market-data workflows evolve independently.

```text
Browser -> NGINX -> Auth / Profile / Data / Pattern / Orchestration
                           |       |       |          |
                       owned DB  cache   artifacts   analysis jobs
```

For the full interaction model, continue to [System Architecture](./architecture). To see the implemented interface, open [Product Experience](./product-experience).

## Technology stack

- **Frontend:** React, Vite, TypeScript, Tailwind CSS
- **Backend:** Python, FastAPI, Pydantic, SQLAlchemy
- **Data:** PostgreSQL/NeonDB and Redis
- **ML:** PyTorch, scikit-learn, NumPy, Pandas
- **MLOps:** MLflow, DVC, DAGsHub
- **Reasoning:** LangGraph and Gemini
- **Operations:** Docker, Docker Compose, NGINX, GitHub Actions, AWS-oriented deployment

## Design goals

- Keep service responsibilities and persistence boundaries explicit.
- Present prediction as evidence, not certainty.
- Continue returning useful results when an optional dependency is unavailable.
- Make every service independently testable and deployable.
- Preserve a clear path from local Compose deployment to managed cloud orchestration.

:::caution Financial use
ForesightX provides analytical support. Its output is not financial advice and should not be treated as a guaranteed prediction or automated trade instruction.
:::
