---
name: Tiny AI Foundry
description: Key architecture and pattern decisions for the nn-sandbox premium LLM browser app.
---

## Training Lab layout (PR 3 refactor)
Tab-based training dashboard was replaced with a permanent dual-column layout inside `trainStep === "training"`:
- **Left panel** (`w-[268px]`, `bg-[#020202]`): Model Architecture sliders, Train Speed slider, Data Feed Engine, Library (SharingHub). All config is always visible.
- **Right panel** (`flex-1`, `bg-[#000000]`): MatrixPulseCore SVG viz → status label → 2×2 TelemetryCard grid → live sample → LLMStats.

**Why:** The old tabs (arch/dataset/explorer/status) forced context-switching. The dual-column lets users tune and monitor simultaneously without navigation.

## MatrixPulseCore visualization
SVG neural network diagram: 4 layers [3, 5, 5, 3], node positions computed at render time. CSS animations `node-pulse` and `edge-pulse` (defined in index.css) applied with staggered `animation-delay` per node/edge. When `active=false`, all elements are static dim.

**How to apply:** Pass `active={llmPlaying}` prop. Add new keyframes to `index.css` if you need different animation names.

## ChatView multi-thread sessions
ChatView owns ALL session state internally (Thread[] array with id/name/messages/modelId). The `messages`/`setMessages` props from App.tsx are used for initial value and to sync the active thread back to App.tsx (for LLMStats messageCount). Never lift thread state to App.tsx — it would bloat the parent.

**Why:** Keeps App.tsx clean. The `setMessages` callback acts as a one-way sync channel, not the source of truth.

## Automated Data Feeder / Token Importance Ranking
`tokenImportanceRank(corpus, keepFraction=0.8)` scores each line by vocab density (60% weight) + bigram variety (40% weight), returns top-80% lines. This is a **simulation** — the actual worker training corpus is not modified. The UI shows "X/Y lines" stats when token ranking is enabled.

**Why:** Changing the actual training corpus risks breaking the worker contract. Simulating it in the UI gives the UX value without risk.

## Loss history tracking
`lossHistory: number[]` state in App.tsx is a rolling buffer (max 80 entries) populated in the worker snapshot handler. Passed to SparkLine (inline SVG polyline). When `lossHistory.length < 2`, SparkLine returns null.

## Sub-components in App.tsx
Bottom of App.tsx has: `SlimNavItem`, `MobileNavButton` (kept), plus new: `PanelLabel`, `SliderRow`, `ToggleSwitch`, `TelemetryCard`, `SparkLine`, `MatrixPulseCore`. `MiniBarTab` and `MetricCell` were removed.

## WebGPU detection
`useEffect` on mount calls `navigator.gpu?.requestAdapter()`. Shows "WebGPU Active" (green, Monitor icon) when adapter is found, "Parallel Thread Core" (muted, Server icon) otherwise. Badge only appears once `webGpuAvailable !== null`.
