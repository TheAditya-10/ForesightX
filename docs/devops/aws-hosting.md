---
title: AWS Hosting
---

# AWS Hosting Guidance

Recommended AWS components for hosting:

- ECR for container registry
- ECS / Fargate or EKS for service orchestration
- RDS (Postgres) per service database
- ElastiCache (Redis) for auth/data caching
- S3 for artifacts and static site hosting
- CloudFront for frontend distribution

Security and secrets

- Use AWS Secrets Manager or Parameter Store for secrets; never hardcode credentials.
- Rotate JWT secrets and OAuth credentials regularly.
