---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
title: DB Design Decisions
---

# Database Design Decisions

This page captures rationale for normalization, indexing, and storage choices.

- Service-owned schemas prevent cross-service coupling and keep ownership clear
- Normalization for market observations avoids repeated raw payloads
- Partitioning is planned for time-series tables at scale
- Large model files stay in DVC/MLflow (or S3); DB stores metadata only
- Job trace tables are append-only for auditability
