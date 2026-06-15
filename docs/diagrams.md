---
last_update:
  date: 2026-06-12T12:00:00.000Z
  author: ForesightX Team
id: diagrams
title: UML Diagrams
---

# UML Diagrams

These diagrams document the ForesightX microservice trading system. The rendered SVGs are stored in `static/img/diagrams`, and the editable PlantUML sources are stored in `docs/uml`.

## System architecture

![ForesightX system architecture](/img/diagrams/system-architecture.svg)

Editable source: [`docs/uml/system-architecture.puml`](./uml/system-architecture.puml)

## Trading analysis sequence

This sequence follows the primary runtime path: authentication, instrument search, `POST /analyze`, parallel data collection, ML prediction, portfolio/risk lookup, decisioning, trace persistence, and optional portfolio update.

![ForesightX trading analysis sequence](/img/diagrams/trading-analysis-sequence.svg)

Editable source: [`docs/uml/trading-analysis-sequence.puml`](./uml/trading-analysis-sequence.puml)

## Service collaboration

This UML communication diagram shows the same system from a collaboration perspective. Numbered links represent the order of responsibility handoff across services and external providers.

![ForesightX service collaboration](/img/diagrams/service-collaboration.svg)

Editable source: [`docs/uml/service-collaboration.puml`](./uml/service-collaboration.puml)
