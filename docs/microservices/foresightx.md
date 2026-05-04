---
title: ForesightX (core)
---

# ForesightX (core service)

Purpose: central API surface and orchestration entrypoints for the platform.

Responsibilities
- Serve core API endpoints consumed by frontends and integrations.
- Route requests to specialized microservices where appropriate.

Tech stack
- Likely Python (FastAPI) or similar, Postgres client, Dockerized.

Input / Output contract
- Exposes REST endpoints that take JSON and return structured JSON payloads.

API endpoints
- Placeholder: actual endpoints live in service code. See the repository folder `ForesightX` for implementation.

Dependencies
- Auth service for auth/token exchange.
- Data service for reads/writes.

Communication
- Synchronous HTTP calls to other services; message-based triggers for long-running jobs.

Error handling
- Standard HTTP error handling and structured error responses; retries for idempotent operations.

Deployment notes
- Containerized with a per-service Dockerfile; images pushed to registry during CI.

Why separate service
- Keeps API surface focused and decoupled from data or ML workloads.
