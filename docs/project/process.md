---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: Development Process
description: Agile incremental model, team structure, and collaboration approach.
---

# Development Process

ForesightX followed an Agile incremental model. The project contained several independent but connected areas: ML experimentation, backend services, frontend design, orchestration, deployment, and documentation. Building in increments allowed each area to be tested before the next integration step.

![Agile incremental development lifecycle](/img/diagrams/agile-lifecycle.jpg)

## Development cycle

1. Define the next requirement and service contract.
2. Design the smallest useful increment.
3. Implement it inside the owning service.
4. Test the service in isolation.
5. Integrate it with its consumers.
6. Review behavior and adjust the next increment.

Repeated integration checks were important because a response-shape change in one service could otherwise break the frontend or orchestration workflow much later.

## Phase-wise development

- **Requirement analysis:** user role, platform scope, service boundaries, and expected outputs
- **Model planning:** features, horizons, artifact strategy, and inference response contract
- **Service implementation:** Pattern, Data, Auth, Profile, and Orchestration APIs
- **Frontend implementation:** responsive pages and API integration
- **System integration:** end-to-end analysis and portfolio paths
- **Deployment:** Docker images, Compose networking, NGINX, and health checks
- **Validation:** unit tests, API checks, database inspection, and report preparation

## Team structure

The project report records a two-member team:

| Member | Primary responsibility |
| --- | --- |
| Aditya Pratap Singh Tomar | Architecture, backend APIs, ML workflow, orchestration, and deployment organization |
| Stuti Jain | Frontend design, interface implementation, integration checks, and documentation support |

The team used an architecture-first collaboration pattern: agree on the next contract, implement in parallel where possible, review the integrated result, and then continue to the next increment.
