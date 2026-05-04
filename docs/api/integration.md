---
title: Integration
---

# Integration Patterns

Service-to-service integration is HTTP-based with strict schemas.

## Key integration flows

- Auth → Profile: create profile on successful registration or OAuth login
- Orchestration → Data: price, indicators, news, and instrument search
- Orchestration → Pattern: prediction signals
- Orchestration → Profile: portfolio and risk context
- Frontend → Services: proxy via `/api/*` to avoid CORS fragmentation

## Best practices

- Use consistent JSON schemas and version the API
- Pass correlation IDs for distributed tracing
- Use timeouts and retries for idempotent operations
