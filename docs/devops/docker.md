---
title: Docker
---

# Docker and container strategy

Each microservice includes a service-level `Dockerfile` in its directory. Images should be built in CI and pushed to a registry.

Best practices:

- Multi-stage builds for small images.
- Pin base images for reproducibility.
- Use `.dockerignore` to keep builds small.

Local development:

- Use the root `docker-compose.yml` for a full stack run
- Each service is isolated and can be built independently
- Default local ports: orchestration 8000, data 8001, profile 8002, pattern 8003, auth 8004

Notes for Pattern service:

- Pull model artifacts before starting inference (`dvc pull`)
- Configure `FORESIGHTX_ARTIFACTS_DIR` when using non-default artifact paths
