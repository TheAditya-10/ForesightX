---
title: Validation & Evidence
description: Test matrix, debugging strategy, and captured results.
---

# Validation & Evidence

Validation covers individual functions, API contracts, service integration, system workflows, and persistent data. Because the platform is distributed, a passing unit suite is necessary but not sufficient: cross-service response shapes and failure behavior must also be exercised.

## Reported test cases

| ID | Area | Scenario | Expected result |
| --- | --- | --- | --- |
| TC-01 | Auth | Valid registration | User created and token flow available |
| TC-02 | Auth | Invalid credentials | Controlled authentication error |
| TC-03 | Data | Valid ticker | Quote/history returned |
| TC-04 | Pattern | Valid feature window | Direction and confidence returned |
| TC-05 | Profile | Portfolio update | Position and transaction persisted |
| TC-06 | Orchestration | Complete analysis request | Consolidated recommendation returned |
| TC-07 | Health | Service health request | Healthy status response |
| TC-08 | Data | Invalid ticker | Structured error without service crash |

## Captured backend results

![Pattern service test report](/img/screenshots/pattern-tests.png)

![Orchestration service test report](/img/screenshots/orchestration-tests.png)

## Persistent-data checks

Database inspection was used to verify that normalized records and expected fields were written after API operations.

![Database table inspection](/img/screenshots/database-table.png)

![Database record inspection](/img/screenshots/database-record.png)

## Debugging approach

1. Reproduce the request against the owning service.
2. Confirm health status and environment configuration.
3. Validate request and response schemas.
4. Inspect service logs and external dependency responses.
5. Verify database writes or cache state.
6. Repeat the end-to-end request through NGINX.

For ML output, validation also includes feature shape, missing values, preprocessing consistency, artifact availability, and basic prediction sanity checks.
