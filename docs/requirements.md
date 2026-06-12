---
title: Requirements
description: Functional, non-functional, hardware, and software requirements.
---

# Requirements

The Software Requirement Specification defines measurable platform behavior and the constraints created by live data, distributed services, prediction latency, and user-facing responsiveness.

## Functional requirements

- Register and sign in users securely.
- Generate and refresh JWT access tokens, revoke sessions, and support Google OAuth.
- Search instruments and retrieve live and historical market data.
- Compute technical indicators such as moving averages and RSI.
- Generate next-hour and next-day forecasts with confidence and trend direction.
- Maintain profile, portfolio, transaction, cash, and risk information.
- Aggregate service outputs into an explainable recommendation.
- Stream live updates over WebSockets.
- Expose health endpoints and OpenAPI documentation for service verification.

## Non-functional requirements

| Area | Requirement | Current target or approach |
| --- | --- | --- |
| Performance | One-year chart retrieval | Approximately 3-4 seconds in the reported environment |
| Prediction latency | Generate a forecast | Within a few seconds |
| Scalability | Scale workloads independently | Service isolation and stateless API containers where possible |
| Reliability | Contain failures | One service failure should not crash the entire platform |
| Security | Protect user sessions | JWT, refresh-token controls, password hashing, and private networking |
| Maintainability | Limit change impact | Independent codebases and owned data models |
| Availability | Reproduce execution | Containerized services and health checks |
| Extensibility | Support cloud growth | Compose design that maps to ECS or Kubernetes concepts |

Caching in the data service reduces repeated network-bound fetches. The prediction workflow uses explicit feature contracts and artifact versioning because the ML layer is expected to evolve faster than the user interface.

## Development environment

| Component | Recommended specification |
| --- | --- |
| Processor | Intel Core i5 / AMD Ryzen 5 or above |
| Memory | 16 GB RAM |
| Storage | 512 GB SSD |
| Operating system | Ubuntu Linux or Windows 11 |
| GPU | Optional NVIDIA GPU for model training |
| Browser | Current Chrome or Firefox |

## Deployment environment

The implemented deployment model uses Docker Compose and one public NGINX reverse proxy. AWS EC2 is the current cloud target. ECR/ECS or Kubernetes on EKS are future scaling paths after the single-host deployment is stable.

## Use cases

![ForesightX use case diagram](/img/diagrams/use-case.jpg)

Primary actors can create an account, authenticate, browse instruments, review analytics, request prediction, inspect recommendations, and maintain portfolio context. External systems provide market data, OAuth identity, model artifacts, and AI assistance.
