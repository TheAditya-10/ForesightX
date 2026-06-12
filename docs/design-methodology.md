---
title: Design & Methodology
description: Component, class, service, prediction, and recommendation design.
---

# Design & Methodology

The implementation follows service ownership, explicit API contracts, and incremental integration. Prediction is treated as one input to a broader recommendation workflow rather than the final answer by itself.

## Domain and class design

![ForesightX design class diagram](/img/diagrams/class.jpg)

The class model separates users and sessions, market entities, prediction records, portfolio state, and analysis jobs. Service-specific repositories and controllers keep transport logic separate from persistence and domain behavior.

## Authentication methodology

![Authentication service workflow](/img/diagrams/auth-workflow.jpg)

The auth service validates credentials, hashes passwords, issues short-lived access tokens, rotates refresh tokens, and supports revocation through Redis-backed state. OAuth users follow the same platform identity flow after provider callback validation.

## Prediction methodology

The Pattern service uses a reproducible offline-to-online pipeline:

1. Collect and validate historical OHLCV data.
2. Clean and normalize the input series.
3. Compute technical features and construct rolling windows.
4. Split data without future leakage.
5. Train and evaluate candidate models.
6. Record experiments with MLflow and version data/artifacts with DVC and DAGsHub.
7. Promote a selected artifact into the inference service.
8. Reapply the saved preprocessing contract for online predictions.

![Offline ML pipeline](/img/diagrams/ml-pipeline.jpg)

![Prediction endpoint workflow](/img/diagrams/prediction-workflow.jpg)

The inference response includes direction and confidence for short horizons. Confidence is communicated as model evidence, not as certainty.

## Data service methodology

![Data service architecture](/img/diagrams/data-service.jpg)

The data service hides provider-specific formats behind stable endpoints. It validates tickers, retrieves quotes and history, computes or serves indicators, caches repeat requests, persists normalized snapshots, and streams selected updates over WebSockets.

## Profile methodology

![Profile service workflow](/img/diagrams/profile-workflow.jpg)

The profile service owns user-facing details and portfolio state. Position updates and transaction history support risk calculations and provide the orchestration service with context such as current exposure and available cash.

## Orchestration methodology

![Orchestration workflow](/img/diagrams/orchestration-workflow.jpg)

Orchestration uses graph state to make multi-service analysis traceable. A typical graph validates the request, fetches evidence, composes signals, generates a recommendation, applies risk checks, and formats the response. Persisted events make failures easier to locate than a single opaque endpoint implementation.

## Recommendation generation

The recommendation combines four classes of evidence:

- **Market evidence:** price movement, volume, technical indicators, and news context
- **Model evidence:** predicted direction, horizon, and confidence
- **User evidence:** holdings, cash, transaction history, and risk context
- **Policy evidence:** deterministic thresholds, exposure constraints, and fallback rules

AI assistance is used to structure and explain the combined evidence. It does not bypass validation or risk rules.
