---
title: Getting Started
description: Run the documentation or the complete ForesightX stack locally.
---

# Getting Started

This page covers the two common entry points: previewing this documentation site and running the application stack.

## Documentation site

From the documentation repository:

```bash
npm ci
npm run start
```

Docusaurus starts a local development server with live reload. Validate a production build with:

```bash
npm run build
npm run serve
```

## Application prerequisites

- Docker Engine with the Compose plugin
- Git
- Service-specific `.env` files
- PostgreSQL connection strings for stateful services
- Required external credentials, such as Gemini and OAuth keys, when those integrations are enabled

## Configure the stack

At the application repository root, create the deployment environment file and one environment file per service:

```bash
cp .env.example .env
cp ForesightX-auth/.env.example ForesightX-auth/.env
cp ForesightX-profile/.env.example ForesightX-profile/.env
cp ForesightX-data/.env.example ForesightX-data/.env
cp ForesightX-orchestration/.env.example ForesightX-orchestration/.env
cp ForesightX-Pattern/.env.example ForesightX-Pattern/.env
```

Use separate database URLs for services that own persistent data. Do not commit populated `.env` files.

## Start and inspect

```bash
./scripts/start.sh
docker compose ps
docker compose logs -f nginx
```

Only NGINX should publish a host port in the production-shaped Compose setup. Backend services communicate through Docker DNS names on the private network.

## Health endpoints

Every service exposes `GET /health`. NGINX additionally exposes `GET /nginx-health`. A healthy process should be both running and responding to its health check.

## Where to go next

- [Product Experience](./product-experience) for the user-facing workflows
- [System Architecture](./architecture) for service boundaries
- [API Endpoints](./api/endpoints) for the HTTP and WebSocket surface
- [Docker Deployment](./devops/docker) for production-oriented container details
