---
title: Deployment Flow
---

# Deployment flow

A practical build / deploy flow:

1. Push code to `main` branch
2. CI runs tests, lints, and builds images
3. Images pushed to registry with semantic tags
4. Deploy to staging (automated), run smoke tests
5. Manual or automated promotion to production with tag-based release

Rollback

- Keep previous image tags available for quick rollback.
- Maintain DB migration plans with backwards compatibility (`alembic upgrade`/`downgrade`).
- For Pattern service, keep previous model artifacts available for quick revert.
