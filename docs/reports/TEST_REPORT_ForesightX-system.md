# Test Report — ForesightX System

Date: 2026-05-13

## Result
PASS with Docker limitation — service tests, frontend build, Compose render, and Pattern reduced-runtime smoke test passed.

## Commands
```bash
# Auth
../.venv/bin/python -m pytest -q --junitxml=../ForesightX-Docs/docs/reports/junit-foresightx-auth.xml

# Data
../.venv/bin/python -m pytest -q --junitxml=../ForesightX-Docs/docs/reports/junit-foresightx-data.xml

# Profile
../.venv/bin/python -m pytest -q --junitxml=../ForesightX-Docs/docs/reports/junit-foresightx-profile.xml

# Pattern
../.venv/bin/python -m pytest -q --junitxml=../ForesightX-Docs/docs/reports/junit-foresightx-pattern.xml

# Orchestration
../.venv/bin/python -m pytest -q --junitxml=../ForesightX-Docs/docs/reports/junit-foresightx-orchestration.xml

# Frontend
npm test
npm run build

# ForesightX public site and docs site
npm run build

# Deployment config
docker compose config
```

## Summary
- Auth: 13 passed.
- Data: 6 passed.
- Profile: 12 passed.
- Pattern: 5 passed.
- Orchestration: 8 passed.
- Frontend: 5 passed; production build passed.
- ForesightX public site build: PASS.
- ForesightX-Docs build: PASS.
- Docker Compose configuration rendered successfully with nginx, redis, frontend, auth, data, profile, pattern, and orchestration.
- Pattern reduced-runtime smoke test passed from a temporary directory containing only the files copied by the production Dockerfile.

## Limitations
- `docker compose build pattern` and `docker compose build frontend` could not run because the Docker daemon was not reachable at the configured socket.
- Full container-to-container live system testing should be repeated on the EC2 host after Docker is running and production `.env` files are finalized.
