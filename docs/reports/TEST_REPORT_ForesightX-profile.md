# Test Report — ForesightX-profile
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team

Date: 2026-05-13

## Result
PASS — 12 tests passed.

## Commands
```bash
# From service root
../.venv/bin/python -m pytest -q --junitxml=../ForesightX-Docs/docs/reports/junit-foresightx-profile.xml
```

## Summary
- Test files: 6
- Tests: 12
- Duration: ~1s

## Artifacts
- JUnit XML: `ForesightX-Docs/docs/reports/junit-foresightx-profile.xml`

## Notes
- Fixed `UpdatePortfolioRequest.quantity` validation (pydantic v2 ignores `Field(ne=0)`; now enforced via validator).
