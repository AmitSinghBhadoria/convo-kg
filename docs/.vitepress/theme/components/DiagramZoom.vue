<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onBeforeUnmount } from 'vue'

// ── Overlay state ────────────────────────────────────────────────────────────
const open = ref(false)
const scale = ref(1)
const translateX = ref(0)
const translateY = ref(0)
const isDragging = ref(false)
const justDragged = ref(false)
const stageEl = ref<HTMLElement | null>(null)

let dragStartX = 0
let dragStartY = 0
let originX = 0
let originY = 0

const MIN = 0.5
const MAX = 3
const STEP = 0.2
const PAN_KEY = 50

const scalePct = computed(() => Math.round(scale.value * 100))

const cursor = computed(() => {
  if (scale.value <= 1) return 'zoom-out'
  return isDragging.value ? 'grabbing' : 'grab'
})

const transform = computed(
  () => `translate(${translateX.value}px, ${translateY.value}px) scale(${scale.value})`,
)

// ── Open / close ─────────────────────────────────────────────────────────────
// Clone the rendered SVG and size it to fit the viewport, preserving aspect.
//
// Three things make this fiddly, each handled below:
//   1. The clone shares every id with the in-page original. SVG `clip-path` and
//      `marker` references (`url(#id)`) then resolve to the ORIGINAL's elements,
//      which mis-clips the clone and ghosts the diagram. Fix: namespace the
//      clone's ids. Mermaid prefixes every internal id with the root diagram id,
//      so one string-replace rewrites ids, `url(#…)` refs, and the scoped
//      `<style>` together. This also means Mermaid's post-render pass can no
//      longer match the clone (so it won't wipe our sizing) and the internal
//      edge/border styles still apply (they're now scoped to the new id).
//   2. Mermaid writes a `max-width` onto the SVG's style; clear it so the SVG
//      can fill our wrapper.
//   3. Put the fit-size on a WRAPPER div and let the SVG fill it at 100% with a
//      preserved aspect ratio.
let cloneSeq = 0
function mountClone(svg: SVGElement) {
  const host = stageEl.value
  if (!host) return
  host.innerHTML = ''
  const rect = svg.getBoundingClientRect()
  let ratio = rect.width > 0 ? rect.height / rect.width : 0
  if (!ratio || !isFinite(ratio)) {
    const vb = (svg.getAttribute('viewBox') || '').split(/[\s,]+/).map(Number)
    ratio = vb.length === 4 && vb[2] ? vb[3] / vb[2] : 0.6
  }
  const maxW = window.innerWidth * 0.92
  const maxH = window.innerHeight * 0.82
  let w = maxW
  let h = w * ratio
  if (h > maxH) {
    h = maxH
    w = ratio ? h / ratio : maxW
  }

  // Build a clone with namespaced ids (see note above); fall back to a plain
  // deep clone if serialization/parsing fails for any reason.
  let clone: SVGElement
  const rootId = svg.getAttribute('id')
  try {
    let markup = new XMLSerializer().serializeToString(svg)
    if (rootId) markup = markup.split(rootId).join(`${rootId}-dz${cloneSeq++}`)
    const parsed = new DOMParser().parseFromString(markup, 'image/svg+xml')
    if (parsed.querySelector('parsererror') || !parsed.documentElement) {
      throw new Error('svg parse failed')
    }
    clone = document.importNode(parsed.documentElement, true) as unknown as SVGElement
  } catch {
    clone = svg.cloneNode(true) as SVGElement
  }
  clone.setAttribute('width', '100%')
  clone.setAttribute('height', '100%')
  clone.setAttribute('preserveAspectRatio', 'xMidYMid meet')
  clone.style.maxWidth = 'none'
  clone.style.maxHeight = 'none'

  const wrap = document.createElement('div')
  wrap.style.cssText =
    `width:${Math.round(w)}px;height:${Math.round(h)}px;` +
    'display:flex;align-items:center;justify-content:center;'
  wrap.appendChild(clone)
  host.appendChild(wrap)
}

function openOverlay(svg: SVGElement) {
  scale.value = 1
  translateX.value = 0
  translateY.value = 0
  open.value = true
  document.documentElement.style.overflow = 'hidden'
  window.addEventListener('keydown', onKeydown)
  nextTick(() => mountClone(svg))
}

function close() {
  open.value = false
  if (stageEl.value) stageEl.value.innerHTML = ''
  document.documentElement.style.overflow = ''
  window.removeEventListener('keydown', onKeydown)
  window.removeEventListener('mousemove', onDragMove)
  window.removeEventListener('mouseup', onDragEnd)
  window.removeEventListener('touchmove', onTouchMove)
  window.removeEventListener('touchend', onDragEnd)
}

// ── Zoom ─────────────────────────────────────────────────────────────────────
function clampPanToZoom() {
  if (scale.value <= 1) {
    translateX.value = 0
    translateY.value = 0
  }
}

function zoomIn() {
  scale.value = Math.min(MAX, Math.round((scale.value + STEP) * 100) / 100)
  clampPanToZoom()
}

function zoomOut() {
  scale.value = Math.max(MIN, Math.round((scale.value - STEP) * 100) / 100)
  clampPanToZoom()
}

function reset() {
  scale.value = 1
  translateX.value = 0
  translateY.value = 0
}

// ── Mouse drag-to-pan ────────────────────────────────────────────────────────
function onDragStart(e: MouseEvent) {
  if (scale.value <= 1) return
  e.preventDefault()
  isDragging.value = true
  dragStartX = e.clientX
  dragStartY = e.clientY
  originX = translateX.value
  originY = translateY.value
  window.addEventListener('mousemove', onDragMove)
  window.addEventListener('mouseup', onDragEnd)
}

function onDragMove(e: MouseEvent) {
  if (!isDragging.value) return
  translateX.value = originX + (e.clientX - dragStartX)
  translateY.value = originY + (e.clientY - dragStartY)
}

function onDragEnd() {
  if (!isDragging.value) return
  isDragging.value = false
  window.removeEventListener('mousemove', onDragMove)
  window.removeEventListener('mouseup', onDragEnd)
  window.removeEventListener('touchmove', onTouchMove)
  window.removeEventListener('touchend', onDragEnd)
  // Guard the backdrop click so releasing a drag never closes the overlay.
  justDragged.value = true
  setTimeout(() => (justDragged.value = false), 100)
}

// ── Touch drag-to-pan (single finger) ────────────────────────────────────────
function onTouchStart(e: TouchEvent) {
  if (scale.value <= 1 || e.touches.length !== 1) return
  const t = e.touches[0]
  isDragging.value = true
  dragStartX = t.clientX
  dragStartY = t.clientY
  originX = translateX.value
  originY = translateY.value
  window.addEventListener('touchmove', onTouchMove, { passive: false })
  window.addEventListener('touchend', onDragEnd)
}

function onTouchMove(e: TouchEvent) {
  if (!isDragging.value || e.touches.length !== 1) return
  e.preventDefault()
  const t = e.touches[0]
  translateX.value = originX + (t.clientX - dragStartX)
  translateY.value = originY + (t.clientY - dragStartY)
}

// ── Keyboard ─────────────────────────────────────────────────────────────────
function onKeydown(e: KeyboardEvent) {
  if (!open.value) return
  switch (e.key) {
    case 'Escape':
      close()
      break
    case '+':
    case '=':
      e.preventDefault()
      zoomIn()
      break
    case '-':
    case '_':
      e.preventDefault()
      zoomOut()
      break
    case '0':
      e.preventDefault()
      reset()
      break
    case 'ArrowUp':
      if (scale.value > 1) { e.preventDefault(); translateY.value += PAN_KEY }
      break
    case 'ArrowDown':
      if (scale.value > 1) { e.preventDefault(); translateY.value -= PAN_KEY }
      break
    case 'ArrowLeft':
      if (scale.value > 1) { e.preventDefault(); translateX.value += PAN_KEY }
      break
    case 'ArrowRight':
      if (scale.value > 1) { e.preventDefault(); translateX.value -= PAN_KEY }
      break
  }
}

function onBackdropClick() {
  if (isDragging.value || justDragged.value) return
  close()
}

// ── Discover rendered Mermaid diagrams and wire click-to-fullscreen ───────────
function svgContainer(svg: SVGElement): HTMLElement {
  return (svg.closest('.mermaid, .mermaid-diagram') as HTMLElement) ||
    (svg.parentElement as HTMLElement)
}

function enhance() {
  const svgs = document.querySelectorAll<SVGElement>(
    '.vp-doc svg[id^="mermaid"], .vp-doc .mermaid svg, .vp-doc .mermaid-diagram svg',
  )
  svgs.forEach((svg) => {
    const box = svgContainer(svg)
    if (!box || box.dataset.zoomReady === '1') return
    box.dataset.zoomReady = '1'
    box.classList.add('zoom-enabled')
    box.addEventListener('click', () => openOverlay(svg))
  })
}

let observer: MutationObserver | null = null
let scanTimer: ReturnType<typeof setTimeout> | null = null
function scheduleScan() {
  if (scanTimer) clearTimeout(scanTimer)
  scanTimer = setTimeout(enhance, 120)
}

onMounted(() => {
  enhance()
  observer = new MutationObserver(scheduleScan)
  observer.observe(document.body, { childList: true, subtree: true })
})

onBeforeUnmount(() => {
  observer?.disconnect()
  close()
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="diagram-zoom-overlay"
      @click.self="onBackdropClick"
    >
      <div class="diagram-zoom-toolbar" @click.stop>
        <button class="dz-btn" title="Zoom out (-)" @click="zoomOut">−</button>
        <span class="dz-level">{{ scalePct }}%</span>
        <button class="dz-btn" title="Zoom in (+)" @click="zoomIn">+</button>
        <button class="dz-btn dz-reset" title="Reset (0)" @click="reset">Reset</button>
        <button class="dz-btn dz-close" title="Close (Esc)" @click="close">✕</button>
      </div>

      <div
        class="diagram-zoom-stage"
        :style="{ cursor }"
        @mousedown="onDragStart"
        @touchstart="onTouchStart"
        @click.self="onBackdropClick"
      >
        <div class="diagram-zoom-content" ref="stageEl" :style="{ transform }" />
      </div>

      <div class="diagram-zoom-hint">
        Scroll-free zoom: <kbd>+</kbd>/<kbd>−</kbd> · <kbd>0</kbd> reset ·
        <kbd>drag</kbd>/<kbd>↑↓←→</kbd> pan · <kbd>Esc</kbd> close
      </div>
    </div>
  </Teleport>
</template>
