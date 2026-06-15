---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: CI/CD
---

# CI/CD

Recommended CI/CD pipeline (GitHub Actions):

1. Run unit tests for each service
2. Build Docker images
3. Push images to registry
4. Deploy to staging and run smoke tests
5. Promote to production

Service-specific notes:

- Pattern service includes ML training/evaluation pipelines (DVC + MLflow)
- Orchestration service depends on `GEMINI_API_KEY` for live decisioning

Example docs workflow exists in `.github/workflows/docs-deploy.yml` and can be extended to build all services.
