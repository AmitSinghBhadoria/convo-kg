import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

// Brand palette (from frontend/index.html): gold accent #C8A24E on a dark canvas.
export default withMermaid(
  defineConfig({
    title: 'Atyx Convo-KG',
    description:
      'Noisy multi-party Hinglish audio → grounded, queryable knowledge graph — on a local open-weight LLM.',
    lang: 'en-US',
    cleanUrls: true,
    lastUpdated: true,

    // The docs/ tree also holds internal build plans/specs + the demo script —
    // keep those out of the published site.
    srcExclude: ['superpowers/**', 'demo_script.md', 'README.md'],

    themeConfig: {
      siteTitle: 'Atyx Convo-KG',

      nav: [
        { text: 'Home', link: '/' },
        { text: 'Product', link: '/product-overview' },
        {
          text: 'Architecture',
          items: [
            { text: 'System Architecture', link: '/system-architecture' },
            { text: 'Entity Relationship', link: '/entity-relationship' },
            { text: 'Sequence Diagrams', link: '/sequence-diagrams' },
            { text: 'Deployment Guide', link: '/deployment-guide' },
          ],
        },
        { text: 'API', link: '/api-specification' },
      ],

      sidebar: [
        {
          text: 'Overview',
          collapsed: false,
          items: [
            { text: 'Documentation Home', link: '/' },
            { text: 'Product Overview', link: '/product-overview' },
          ],
        },
        {
          text: 'Architecture & Data',
          collapsed: false,
          items: [
            { text: 'System Architecture', link: '/system-architecture' },
            { text: 'Entity Relationship', link: '/entity-relationship' },
            { text: 'Sequence Diagrams', link: '/sequence-diagrams' },
          ],
        },
        {
          text: 'Product Experience',
          collapsed: false,
          items: [
            { text: 'User Stories', link: '/user-stories' },
            { text: 'Wireflows', link: '/wireflows' },
            { text: 'Wireframes', link: '/wireframes' },
          ],
        },
        {
          text: 'API & Operations',
          collapsed: false,
          items: [
            { text: 'API Specification', link: '/api-specification' },
            { text: 'Deployment Guide', link: '/deployment-guide' },
          ],
        },
      ],

      search: { provider: 'local' },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/AmitSinghBhadoria/convo-kg' },
      ],

      outline: { level: [2, 3] },

      footer: {
        message: 'Atyx Convo-KG — conversational knowledge graph on a local LLM.',
        copyright: 'Local research prototype · no auth · single-user.',
      },
    },

    // Brand-tuned Mermaid theme (dark nodes + gold accents read on light & dark pages).
    mermaid: {
      theme: 'base',
      themeVariables: {
        // NOTE: do not set a custom fontFamily here — Mermaid measures label
        // width at render time; a web font that loads later than the measure
        // pass makes nodes too narrow and clips labels. Mermaid's default
        // system-font stack is available immediately and sizes correctly.
        primaryColor: '#1c1b21',
        primaryTextColor: '#ECE8E1',
        primaryBorderColor: '#C8A24E',
        secondaryColor: '#141318',
        secondaryTextColor: '#ECE8E1',
        secondaryBorderColor: '#8A8579',
        tertiaryColor: '#0f0e12',
        tertiaryTextColor: '#ECE8E1',
        tertiaryBorderColor: '#3a3942',
        lineColor: '#C8A24E',
        textColor: '#ECE8E1',
        mainBkg: '#1c1b21',
        nodeBorder: '#C8A24E',
        clusterBkg: '#141318',
        clusterBorder: '#3a3942',
        titleColor: '#C8A24E',
        edgeLabelBackground: '#0A0A0C',
        actorBkg: '#1c1b21',
        actorBorder: '#C8A24E',
        actorTextColor: '#ECE8E1',
        signalColor: '#ECE8E1',
        signalTextColor: '#ECE8E1',
        labelBoxBkgColor: '#141318',
        labelBoxBorderColor: '#C8A24E',
        labelTextColor: '#ECE8E1',
        noteBkgColor: '#2a2410',
        noteTextColor: '#ECE8E1',
        noteBorderColor: '#C8A24E',
      },
    },

    // Tag rendered diagrams with a stable class for the zoom layer + CSS.
    mermaidPlugin: {
      class: 'mermaid-diagram',
    },
  })
)
