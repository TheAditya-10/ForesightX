---
title: ForesightX-auth
---

# ForesightX-auth

## Purpose

Central authentication and identity service for the platform. It issues JWT access/refresh tokens, handles Google OAuth, and bootstraps user profiles.

## Responsibilities

- User registration and login
- JWT issuance, refresh rotation, and blacklist handling
- OAuth login via Google
- Profile bootstrap on first registration/login

## Tech stack

- FastAPI + async SQLAlchemy
- PostgreSQL (users + identity records)
- Redis (refresh sessions + token blacklist)
- Authlib + JWT utilities

## API endpoints

- `POST /auth/sign-up`
- `POST /auth/sign-in`
- `POST /auth/token/refresh`
- `POST /auth/sign-out`
- `GET /auth/me`
- `GET /oauth/google/authorize`
- `GET /oauth/google/callback`
- `GET /health`
- `GET /docs` (OpenAPI UI)

## Input / Output contract

- Access token: short-lived (default 15 minutes)
- Refresh token: rotating, stored in Redis with session metadata
- Users: email + password (bcrypt) or Google OAuth subject

## Dependencies

- Postgres (service-owned)
- Redis (session store)
- Profile service (`POST /profiles` for profile bootstrap)

## Communication

On sign-up or first OAuth login, the service calls the profile service:

```http
POST {PROFILE_SERVICE_URL}{PROFILE_CREATE_PATH}
Content-Type: application/json

{
	"user_id": "<uuid>",
	"email": "user@example.com"
}
```

## Error handling

- `401/403` for token validation errors and disabled accounts
- `409` on duplicate registration
- Revoked token JTIs are blacklisted in Redis

## Deployment notes

- Configure via `.env` in the service root
- Key env vars: `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `PROFILE_SERVICE_URL`
- Default port: `8004`

:::tip
OAuth is optional and only enabled when `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are set.
:::
