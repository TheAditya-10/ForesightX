# Test Report — ForesightX-frontend

Date: 2026-05-13

## Result
PASS — 5 tests passed.

## Commands
```bash
npm ci
npm test
npm run build
```

## Summary
- Test files: 3
- Tests: 5
- Runner: Vitest
- Production build: PASS

## Notes
- `npm ci` reported 15 vulnerabilities (3 low, 7 moderate, 5 high). This does not fail tests, but it is a deployment risk and should be addressed via `npm audit` / dependency upgrades in a controlled PR.
- The landing navbar now supports an optional `Documentation` link from `VITE_DOCUMENTATION_URL`; it is omitted when the env value is empty.
