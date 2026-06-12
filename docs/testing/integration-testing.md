---
title: Integration Testing
---

# Integration Testing

Integration tests verify the contracts that connect services and the proxy routes exposed to the frontend.

## Critical paths

- Auth creates a platform identity and coordinates profile creation where required.
- Data returns the normalized history expected by Pattern and the frontend.
- Pattern accepts the defined feature window and returns a stable prediction schema.
- Profile returns portfolio and risk context required by Orchestration.
- Orchestration handles successful, partial, and failed dependency responses.
- NGINX preserves path, authorization, forwarding, and WebSocket headers.

## Compose-based verification

```bash
docker compose up -d --build
docker compose ps
curl -f http://localhost/nginx-health
curl -f http://localhost/api/data/health
curl -f http://localhost/api/orchestration/health
```

Tests should use controlled fixtures or provider stubs where an external API would make CI results nondeterministic.
