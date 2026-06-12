# ForesightX Documentation

Docusaurus site for the ForesightX product guide, architecture, service reference, API surface, testing evidence, and deployment documentation.

## Local development

```bash
npm ci
npm run start
```

## Production build

```bash
npm run build
npm run serve
```

The default production path is `/ForesightX/` for GitHub Pages. Override deployment values when needed:

```bash
SITE_URL=https://docs.example.com BASE_URL=/ npm run build
```

Source documents live in `docs/`, homepage code in `src/pages/index.mdx`, theme rules in `src/css/custom.css`, and extracted report assets in `static/img/diagrams` and `static/img/screenshots`.
