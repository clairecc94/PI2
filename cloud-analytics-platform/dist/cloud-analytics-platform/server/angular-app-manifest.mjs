
export default {
  bootstrap: () => import('./main.server.mjs').then(m => m.default),
  inlineCriticalCss: true,
  baseHref: '/',
  locale: undefined,
  routes: [
  {
    "renderMode": 2,
    "redirectTo": "/dashboard",
    "route": "/"
  },
  {
    "renderMode": 2,
    "route": "/dashboard"
  },
  {
    "renderMode": 2,
    "route": "/real-time-analysis"
  },
  {
    "renderMode": 2,
    "route": "/scan-history"
  }
],
  entryPointToBrowserMapping: undefined,
  assets: {
    'index.csr.html': {size: 679, hash: '015dbd9c56d2918740588b882947f6ab76fa58b6835594e3cf6313a93e1e147d', text: () => import('./assets-chunks/index_csr_html.mjs').then(m => m.default)},
    'index.server.html': {size: 1192, hash: 'a22fa571e6f3de6ee262fb71b71cc3e9fd4447439e10be88f02ff59e18f4ffcc', text: () => import('./assets-chunks/index_server_html.mjs').then(m => m.default)},
    'real-time-analysis/index.html': {size: 4990, hash: 'ee0d97703b636addfde88f06154f94e0565adee5681d8ce19ffe3fe53b711a60', text: () => import('./assets-chunks/real-time-analysis_index_html.mjs').then(m => m.default)},
    'dashboard/index.html': {size: 6723, hash: 'ce405e56e121a5942b9cdf90764c1a77bedcf0d60a7b0b17c761a6e2e2ebf94f', text: () => import('./assets-chunks/dashboard_index_html.mjs').then(m => m.default)},
    'scan-history/index.html': {size: 4815, hash: '56306d40a1769a791c5a0aac5b1a183397b5ca2fbd1629f04395f2709a20c59e', text: () => import('./assets-chunks/scan-history_index_html.mjs').then(m => m.default)},
    'styles-5INURTSO.css': {size: 0, hash: 'menYUTfbRu8', text: () => import('./assets-chunks/styles-5INURTSO_css.mjs').then(m => m.default)}
  },
};
