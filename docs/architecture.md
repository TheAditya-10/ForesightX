---
id: architecture
title: Architecture
---

# System Architecture

This page summarizes the high-level architecture for ForesightX. Full UML source files and larger rendered diagrams are available on the [UML Diagrams](./diagrams) page.

## High-level structure

- Microservice architecture with service-owned databases
- HTTP contracts between services and a frontend proxy for unified routing
- ML artifacts managed via DVC + MLflow

## Request flow (analyze)

1. Client calls `POST /analyze` on orchestration service
2. Orchestration validates request and executes a LangGraph workflow
3. Data fetch node calls data + pattern + sentiment tools in parallel
4. Analysis node combines signals with portfolio context from profile service
5. Decision node uses Gemini for a structured decision; falls back to deterministic logic
6. Risk check node enforces exposure and volatility limits
7. Response node returns action + trace, and persists job + events

## Data flow

- Data service fetches external prices/news, caches in Redis, and persists normalized records
- Pattern service builds features from historical data and serves predictions
- Profile service stores user profiles, cash balance, and portfolio positions

## Containerization & deployment

- Each service is containerized with a service-specific Dockerfile
- CI builds images and pushes to a registry (ECR)
- AWS targets: ECS/Fargate + RDS + S3; CloudFront for frontend

## Architecture diagram

![ForesightX system architecture](/img/diagrams/system-architecture.svg)

## Request flow diagram

![ForesightX trading analysis sequence](/img/diagrams/trading-analysis-sequence.svg)

## Collaboration diagram

![ForesightX service collaboration](/img/diagrams/service-collaboration.svg)
