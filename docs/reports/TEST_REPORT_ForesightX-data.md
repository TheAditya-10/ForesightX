# Test Report — ForesightX-data

Date: 2026-05-13

## Result
PASS — 6 tests passed.

## Commands
```bash
# From repo root
.venv/bin/pytest -q --junitxml=../ForesightX-Docs/docs/reports/junit-foresightx-data.xml
```

## Summary
- Test files: 3
- Tests: 6
- Duration: ~4s

## Artifacts
- JUnit XML: `ForesightX-Docs/docs/reports/junit-foresightx-data.xml`

## Notes
- Added `tests/conftest.py` to ensure `app` imports resolve when running `pytest` from the service root.
