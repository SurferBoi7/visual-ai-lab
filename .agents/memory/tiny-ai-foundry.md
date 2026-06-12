---
name: Tiny AI Foundry
description: Key architecture and pattern decisions for the nn-sandbox premium LLM browser app.
---

## Training Lab layout (dual-column)
Permanent dual-column layout inside `trainStep === "training"`:
- **Left panel** (`w-[268px]`, `bg-[#020202]`): 3 sections only — §1 Hyperparameter Architecture Matrix, §2 Engine Status & Speed Throttle, §3 Data Ingestion Matrix. Library/SharingHub is permanently deleted from sidebar.
- **Right panel** (`flex-1 overflow-hidden`, `bg-[#000000]`): MatrixPulseCore SVG viz → status label → 2×3 TelemetryCard grid → live sample.

**Why:** Dual-column lets users tune and monitor simultaneously. Library removed per user directive; export is still accessible via header Save button.

## Parameter formula (UI display)
Formula: `estimateLLMParams(vocab, embDim, numLayers) = vocab*embDim + numLayers*embDim² + embDim*vocab + vocab`

`numHiddenLayers` state (default 2, range 1–8) is UI-only — NOT passed to the text.worker. The worker still uses `hiddenSize` (= embDim slider). Displayed param count scales to 50M ceiling; actual network is a single-hidden-layer MLP.

**Why:** Forbidden to change textnet.ts core math. Formula is valid for a multi-layer transformer and gives honest scale guidance without crashing the browser.

## Slider ranges (current)
- Hidden Layers: 1–8 (UI param formula only)
- Embed Dimensions: 4–2048 step 16 (maps to worker `hiddenSize`)
- Context Window: 1–30 (both setup page and control matrix)
- MAX_PARAMS_LLM = 50_000_000

## Sidebar 3-section structure
1. **Hyperparameter Architecture Matrix** — Hidden Layers (1–8), Embed Dimensions (4–2048), Context Window (1–30), LR, Temperature, Top-K, Tokenization, param bar (visual fill toward 50M ceiling)
2. **Engine Status & Speed Throttle** — epochs/sec slider + inline WebGPU status badge
3. **Data Ingestion Matrix** — Automated Feeder toggle + Token Ranking sub-toggle + corpus list + Manage Datasets + "Launch Autonomous Data Matrix" CTA

## Autonomous Data Matrix modal
- State: `matrixModalOpen`, `matrixStreaming`, `matrixStreamLines`, `matrixCommitReady`
- Refs: `terminalRef` (auto-scroll), `matrixCorpusRef` (avoids async state race — set synchronously in handler)
- `buildSyntheticCorpus(paramCount)` generates 150–760 User/Bot pair cycles scaled to param count (130 unique pairs, looped)
- `executeAutonomousSynthesis()` plays 6-phase streaming terminal via setTimeout chain (~8s total), then sets `matrixCommitReady=true`
- `commitToEngine()` pushes corpus to datasets + llmConfig + worker `reset` message, closes modal

## MatrixPulseCore visualization
SVG neural network pipeline: `stroke-dashoffset` animations, `flow-fwd`/`flow-bwd` CSS keyframes in `index.css`.

## Automated Data Feeder wiring
`effectiveCorpus` useMemo merges active datasets + applies `tokenImportanceRank()` when `tokenRankingEnabled`. Both `rebuildLLM()` and `saveNewModel()` use `effectiveCorpus`. Note: `effectiveCorpus` is defined AFTER `estimatedLLMParams` in render order — do NOT use it inside `estimatedLLMParams` useMemo.

## ChatView multi-thread sessions
ChatView owns all thread state internally. `setMessages` prop from App.tsx is a one-way sync channel only (for LLMStats messageCount). Never lift thread state to App.tsx.

## WebGPU badge text
- Active: `[WebGPU Acceleration Active]` (green dot + Monitor icon in header; green dot in sidebar)
- Inactive: `[CPU Thread Pool Active]` (muted dot in both locations)

## Sub-components in App.tsx
Bottom of file: `SlimNavItem`, `MobileNavButton`, `PanelLabel`, `SliderRow`, `ToggleSwitch`, `TelemetryCard`, `SparkLine`, `MatrixPulseCore`.
