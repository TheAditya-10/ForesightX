const githubUrl = 'https://github.com/TheAditya-10/ForesightX';

module.exports = {
  title: 'ForesightX',
  tagline: 'Intelligent stock analytics, prediction, and explainable recommendations',
  url: process.env.SITE_URL || 'https://theaditya-10.github.io',
  baseUrl: process.env.BASE_URL || '/',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  favicon: 'img/logo.svg',
  organizationName: 'TheAditya-10',
  projectName: 'ForesightX',
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: `${githubUrl}/edit/main/`,
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  themeConfig: {
    image: 'img/screenshots/landing.png',
    metadata: [
      {name: 'theme-color', content: '#0d1211'},
      {name: 'description', content: 'Technical documentation for the ForesightX microservice-based intelligent stock analytics platform.'},
    ],
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'ForesightX',
      hideOnScroll: true,
      logo: {alt: 'ForesightX logo', src: 'img/logo.svg'},
      items: [
        {to: '/docs/overview', label: 'Guide', position: 'left'},
        {to: '/docs/product-experience', label: 'Product', position: 'left'},
        {to: '/docs/architecture', label: 'Architecture', position: 'left'},
        {to: '/docs/microservices', label: 'Services', position: 'left'},
        {to: '/docs/api/endpoints', label: 'API', position: 'left'},
        {href: githubUrl, label: 'GitHub', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Explore',
          items: [
            {label: 'Product overview', to: '/docs/overview'},
            {label: 'System architecture', to: '/docs/architecture'},
            {label: 'Application gallery', to: '/docs/product-experience'},
          ],
        },
        {
          title: 'Build',
          items: [
            {label: 'Microservices', to: '/docs/microservices'},
            {label: 'API reference', to: '/docs/api/endpoints'},
            {label: 'Deployment', to: '/docs/devops/docker'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'Requirements', to: '/docs/requirements'},
            {label: 'Testing', to: '/docs/testing/validation'},
            {label: 'GitHub', href: githubUrl},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} ForesightX. Built with Docusaurus.`,
    },
    prism: {
      additionalLanguages: ['bash', 'json', 'python'],
    },
  },
};
