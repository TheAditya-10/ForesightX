# Test Report — ForesightX-Pattern

Date: 2026-05-13

## Result
PASS — 5 tests passed.

## Commands
```bash
# From repo root
../.venv/bin/python -m pytest -q --junitxml=../ForesightX-Docs/docs/reports/junit-foresightx-pattern.xml

# Reduced runtime file-set smoke test
FORESIGHTX_ARTIFACTS_DIR=/tmp/.../app/artifacts/model python -c "from foresightx_pattern.app.main import create_app; create_app()"
```

## Summary
- Test files: 5
- Tests: 5
- Duration: ~12s
- Reduced runtime import smoke test: PASS

## Artifacts
- JUnit XML: `ForesightX-Docs/docs/reports/junit-foresightx-pattern.xml`

## Notes
- Tests cover API prediction path and model/feature utilities at a basic level.
- The deployment Dockerfile now copies only the inference API, required runtime ML modules, `configs/default.yaml`, and the latest model bundle from `artifacts/model`.
- Docker image build could not be executed in this environment because the Docker daemon was not reachable.
