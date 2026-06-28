import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h } from 'vue'
import DiagramZoom from './components/DiagramZoom.vue'
import './custom.css'

// Extend the default VitePress theme and mount the global diagram zoom/pan layer
// once (via the layout-bottom slot). DiagramZoom scans the page for rendered
// Mermaid diagrams, makes them click-to-fullscreen, and owns the overlay.
export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(DiagramZoom),
    })
  },
} satisfies Theme
