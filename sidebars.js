module.exports = {
  docs: [
    'overview',
    'architecture',
    {
      type: 'category',
      label: 'Microservices',
      items: [
        'microservices/index',
        'microservices/foresightx',
        'microservices/foresightx-auth',
        'microservices/foresightx-data',
        'microservices/foresightx-orchestration',
        'microservices/foresightx-pattern',
        'microservices/foresightx-profile',
        'microservices/foresightx-trading-hub',
        'microservices/foresightx-frontend'
      ]
    },
    {
      type: 'category',
      label: 'Database',
      items: ['database/schema', 'database/er-diagram', 'database/db-design-decisions']
    },
    {
      type: 'category',
      label: 'DevOps',
      items: ['devops/docker', 'devops/cicd', 'devops/aws-hosting', 'devops/deployment-flow']
    },
    {
      type: 'category',
      label: 'API',
      items: ['api/endpoints', 'api/integration']
    },
    {
      type: 'category',
      label: 'Testing',
      items: ['testing/unit-testing', 'testing/integration-testing', 'testing/validation']
    },
    'security',
    'roadmap',
    'resources'
  ]
};
