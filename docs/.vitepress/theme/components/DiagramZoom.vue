<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'

// ── Overlay state ────────────────────────────────────────────────────────────
const open = ref(false)
const svgHtml = ref('')
const scale = ref(1)
const translateX = ref(0)
const translateY = ref(0)
const isDragging = ref(false)
const justDragged = ref(false)

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
function openOverlay(svg: SVGElement) {
  svgHtml.value = svg.outerHTML
  scale.value = 1
  translateX.value = 0
  translateY.value = 0
  open.value = true
  document.documentElement.style.overflow = 'hidden'
  window.addEventListener('keydown', onKeydown)
}

function close() {
  open.value = false
  svgHtml.value = ''
  document.documentElement.style.overflow = ''
  window.removeEventListener('keydown', onKeydown)
  window.removeEventListener('mousemove', onDragMove)
  window.removeEventListener('mouseup', onDragEnd)
  window.removeEventListener('touchmove', onTouchMove)
  window.removeEventListener('touchend', onDragEnd)
}

// ── Zoom ─────────────────────────────────────────────────────────────────────
function clampPanToZoom() {
  // Panning only makes sense above 100%; snap back to centre at/below 100%.
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

// Click on the dark backdrop closes — unless we just finished dragging.
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
        <div class="diagram-zoom-content" :style="{ transform }" v-html="svgHtml" />
      </div>

      <div class="diagram-zoom-hint">
        Scroll-free zoom: <kbd>+</kbd>/<kbd>−</kbd> · <kbd>0</kbd> reset ·
        <kbd>drag</kbd>/<kbd>↑↓←→</kbd> pan · <kbd>Esc</kbd> close
      </div>
    </div>
  </Teleport>
</template>
