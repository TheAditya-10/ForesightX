---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: Unit Testing
---

# Unit Testing

Backend services use `pytest`; the React frontend uses Vitest and Testing Library. Unit coverage is organized around business rules and service boundaries rather than framework internals.

```bash
pytest -q
```

Important backend units include token creation and revocation, settings validation, market normalization, cache behavior, feature engineering, model input validation, portfolio calculations, risk rules, and orchestration tool adapters.

Pattern-specific examples:

```bash
pytest tests/test_model.py tests/test_features.py tests/test_api.py
```

Frontend tests cover reusable UI behavior, session utilities, and component states. External providers and inter-service clients should be mocked at their adapters so failures remain deterministic.
