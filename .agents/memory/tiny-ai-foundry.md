---
name: Tiny AI Foundry
description: Key architecture and pattern decisions for the nn-sandbox premium LLM browser app.
---

## Training Lab layout (PR 3 refactor)
Tab-based training dashboard was replaced with a permanent dual-column layout inside `trainStep === "training"`:
- **Left panel** (`w-[268px]`, `bg-[#020202]`): Model Architecture sliders, Train Speed slider, Data Feed Engine, Library (SharingHub). All config is always visible.
- **Right panel** (`flex-1 overflow-hidden`, `bg-[#000000]`): MatrixPulseCore SVG viz → status label → 2×3 TelemetryCard grid → live sample. Panel is `overflow-hidden` — no scrolling.

**Why:** The old tabs forced context-switching. The dual-column lets users tune and monitor simultaneously without navigation.

## MatrixPulseCore visualization
SVG neural network pipeline: 4 layers [3, 5, 5, 3], cubic bezier paths between layers, layer labels at bottom. Uses `stroke-dashoffset` CSS animation for flowing particles (blue forward pass `flow-fwd`, orange backward pass `flow-bwd` keyframes in index.css). When `active=false`, all elements are static dim blueprint.

**How to apply:** Pass `active={llmPlaying}`. The `flow-fwd`/`flow-bwd` keyframes are in `index.css`. Paths use `M x1,y1 C mx,y1 mx,y2 x2,y2` where mx = midpoint of adjacent layer x values.

## Automated Data Feeder wiring
`effectiveCorpus` (useMemo, defined AFTER `llmModels` in derived state) computes the corpus actually sent to the worker:
- `feederEnabled=false` → uses `llmConfig.corpus` unchanged
- `feederEnabled=true` → merges all active datasets; applies `tokenImportanceRank()` if `tokenRankingEnabled=true`

`rebuildLLM()` and `saveNewModel()` both use `effectiveCorpus` (not `llmConfig.corpus`) when posting `reset` to the worker. This wires the UI toggle to the training pipeline without touching worker math.

**Why:** Changing the actual training corpus risks breaking the worker contract. The feeder is pure main-thread preprocessing — the worker receives a clean filtered corpus string, no protocol change.

**Note:** `effectiveCorpus` is defined after `rebuildLLM` in render order. This is safe because `rebuildLLM` is only called in event handlers (never during render), by which point `effectiveCorpus` is already initialized.

## ChatView multi-thread sessions
ChatView owns ALL session state internally (Thread[] array with id/name/messages/modelId). The `messages`/`setMessages` props from App.tsx are used for initial value and to sync the active thread back to App.tsx (for LLMStats messageCount). Never lift thread state to App.tsx — it would bloat the parent.

**Why:** Keeps App.tsx clean. The `setMessages` callback acts as a one-way sync channel, not the source of truth.

## Token Importance Ranking
`tokenImportanceRank(corpus, keepFraction=0.8)` scores each line by vocab density (60% weight) + bigram variety (40% weight), returns top-80% lines. This is pure main-thread preprocessing — the actual worker training corpus is what gets filtered.

## Loss history tracking
`lossHistory: number[]` state in App.tsx is a rolling buffer (max 80 entries) populated in the worker snapshot handler. Passed to SparkLine (inline SVG polyline). When `lossHistory.length < 2`, SparkLine returns null.

## Hidden neurons slider bounds
Both the Setup page and Control Matrix sliders for Hidden Neurons are `min={4} max={1024} step={8}`. With word LM (vocab ~5000) and ctx=20, hidden=1024 reaches ~102M params — exceeds the 50M ceiling. MAX_PARAMS_LLM guard at 50_000_000 prevents building models that exceed this.

## Telemetry cards (2×3 grid)
Six cards: Global Loss (with SparkLine + converging indicator), Epoch Counter, Throughput (tok/s), RAM Overhead (MB + param count), Vocab Coverage (unique token types), Bits/Char (loss ÷ ln(2), cross-entropy bits).

## WebGPU detection
`useEffect` on mount calls `navigator.gpu?.requestAdapter()`. Shows "WebGPU Active" (green, Monitor icon) when adapter is found, "Parallel Thread Core" (muted, Server icon) otherwise. Badge only appears once `webGpuAvailable !== null`.

## Sub-components in App.tsx
Bottom of App.tsx has: `SlimNavItem`, `MobileNavButton` (kept), plus: `PanelLabel`, `SliderRow`, `ToggleSwitch`, `TelemetryCard`, `SparkLine`, `MatrixPulseCore`. `MiniBarTab` and `MetricCell` were removed.
