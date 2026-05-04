---
id: security
title: Security
---

# Security

Security best practices and considerations for the platform:

- Secure service-to-service communication (mTLS or trusted network layers where needed).
- Use principle of least privilege for IAM roles and database users.
- Rotate secrets and use a secret manager.
- Auth service hashes passwords (bcrypt) and rotates refresh tokens.
- Redis maintains token blacklists to revoke compromised sessions.
- OAuth credentials must be stored in Secrets Manager and rotated.
