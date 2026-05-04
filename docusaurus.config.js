/**
 * Docusaurus configuration for ForesightX documentation site.
 * Replace placeholders like [GITHUB_REPO_LINK] and [LIVE_DEMO_LINK] when available.
 */

module.exports = {
  title: 'ForesightX',
  tagline: 'Predictive analytics platform for financial foresight',
  // Replace with your production URL when ready.
  url: 'https://example.com',
  baseUrl: '/',
  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: '[GITHUB_ORG]',
  projectName: '[GITHUB_REPO]',
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://example.com'
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css')
        }
      }
    ]
  ],
  themeConfig: {
    navbar: {
      title: 'ForesightX',
      logo: {
        alt: 'ForesightX Logo',
        src: 'img/logo.svg'
      },
      items: [
        { to: 'docs/overview', label: 'Docs', position: 'left' },
        { to: 'docs/microservices', label: 'Microservices', position: 'left' },
        { to: 'docs/architecture', label: 'Architecture', position: 'left' },
        { href: 'https://example.com', label: 'GitHub', position: 'right' }
      ]
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Platform',
          items: [
            { label: 'Overview', to: 'docs/overview' },
            { label: 'Architecture', to: 'docs/architecture' }
          ]
        },
        {
          title: 'Resources',
          items: [
            { label: 'Roadmap', to: 'docs/roadmap' },
            { label: 'API', to: 'docs/api/endpoints' }
          ]
        }
      ],
      copyright: `Copyright © ${new Date().getFullYear()} ForesightX`
    }
  }
};
