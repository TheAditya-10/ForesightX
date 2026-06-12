module.exports = {
  docs: [
    {
      type: 'category',
      label: 'Start Here',
      collapsed: false,
      items: ['overview', 'getting-started', 'product-experience', 'requirements'],
    },
    {
      type: 'category',
      label: 'Architecture & Design',
      collapsed: false,
      items: ['architecture', 'design-methodology', 'system-flows'],
    },
    {
      type: 'category',
      label: 'Microservices',
      items: [
        'microservices/index',
        'microservices/foresightx-auth',
        'microservices/foresightx-data',
        'microservices/foresightx-pattern',
        'microservices/foresightx-profile',
        'microservices/foresightx-orchestration',
        'microservices/foresightx-frontend',
      ],
    },
    {
      type: 'category',
      label: 'Data & API',
      items: [
        'api/endpoints',
        'api/integration',
        'database/schema',
        'database/er-diagram',
        'database/db-design-decisions',
      ],
    },
    {
      type: 'category',
      label: 'Testing & Security',
      items: [
        'testing/unit-testing',
        'testing/integration-testing',
        'testing/validation',
        'security',
      ],
    },
    {
      type: 'category',
      label: 'Deployment',
      items: ['devops/docker', 'devops/deployment-flow', 'devops/cicd', 'devops/aws-hosting'],
    },
    {
      type: 'category',
      label: 'Project Report',
      items: ['project/process', 'project/execution', 'project/outcomes', 'roadmap', 'resources'],
    },
  ],
};
