---
title: Unit testing
---

# Unit testing

Guidance for writing and running unit tests for services. Use `pytest` for Python services and keep tests focused on small units.

Local test command example:

```
pytest -q
```

Pattern service tests:

```bash
pytest tests/test_model.py tests/test_features.py tests/test_api.py
```
