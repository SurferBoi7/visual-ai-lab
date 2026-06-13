/// <reference lib="webworker" />

import {
  TextNetwork,
  buildVocab,
  makeWindows,
  generateTokens,
  joinTokens,
  tokenize,
  sampleFromProbs,
  PAD_TOKEN,
  EOS_TOKEN,
  // Re-exported only because makeSample still uses generateTokens for the
  // training-tick "live dream" preview where stop-sequencing isn't desired.
  type Tokenization,
} from "./textnet";

// Inference stop sequence — when the model "starts speaking as the user", we
// cut generation off so the chat reply doesn't run into the next turn.
const STOP_SEQUENCES = ["user", " user"];

// Normalize raw text: lowercase, strip punctuation, collapse whitespace.
// Preserves <PAD> and <EOS> special tokens.
export function normalizeText(text: string): string {
  const cleanChunk = (s: string) =>
    s
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]+/gu, " ")
      .replace(/\s+/g, " ")
      .trim();
  if (!text.includes(PAD_TOKEN) && !text.includes(EOS_TOKEN)) {
    return cleanChunk(text);
  }
  const parts = text.split(/(<PAD>|<EOS>)/g);
  return parts
    .map((p) => (p === PAD_TOKEN || p === EOS_TOKEN ? ` ${p} ` : cleanChunk(p)))
    .join("")
    .replace(/\s+/g, " ")
    .trim();
}

function findStopCut(text: string): number {
  let earliest = -1;
  for (const s of STOP_SEQUENCES) {
    const idx = text.indexOf(s);
    if (idx !== -1 && (earliest === -1 || idx < earliest)) earliest = idx;
  }
  return earliest;
}

interface InitOpts {
  corpus: string;
  // Optional per-dataset corpora for interleaved continual learning.
  // When provided (and length > 1), trainEpoch round-robins across datasets
  // so no single dataset dominates gradient updates (anti-catastrophic-forgetting).
  corpusDatasets?: string[];
  contextSize: number;
  hiddenSize: number;
  learningRate: number;
  temperature: number;
  tokenization: Tokenization;
  topK?: number;
}

function inferTokenizationFromVocab(vocabArr: string[]): Tokenization {
  for (const t of vocabArr) {
    if (t === PAD_TOKEN || t === EOS_TOKEN) continue;
    if (typeof t === "string" && t.length > 1) return "word";
  }
  return "char";
}

function applyTopK(probs: Float32Array, k: number): Float32Array {
  if (!Number.isFinite(k) || k <= 0 || k >= probs.length) return probs;
  const indexed: { i: number; p: number }[] = new Array(probs.length);
  for (let i = 0; i < probs.length; i++) indexed[i] = { i, p: probs[i] };
  indexed.sort((a, b) => b.p - a.p);
  const out = new Float32Array(probs.length);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const { i: idx, p } = indexed[i];
    out[idx] = p;
    sum += p;
  }
  if (sum <= 0) return probs;
  const inv = 1 / sum;
  for (let i = 0; i < out.length; i++) out[i] *= inv;
  return out;
}

// ── WebGPU lifecycle handshake ──────────────────────────────────────────────
let gpuActive = false;
(async () => {
  try {
    const gpu = (self as unknown as { navigator?: { gpu?: { requestAdapter: () => Promise<unknown> } } }).navigator?.gpu;
    if (gpu) {
      const adapter = await gpu.requestAdapter();
      gpuActive = adapter !== null;
    }
  } catch {
    gpuActive = false;
  }
  postMessage({ type: "gpuStatus", active: gpuActive });
})();

// ── Per-dataset window set for interleaved training ─────────────────────────
// Each entry holds the windows extracted from one source dataset, plus its
// own shuffle-cursor so interleaving is truly round-robin rather than
// accidentally serializing datasets.
interface DatasetWindowSet {
  inputs: number[][];
  targets: number[];
  order: number[];
  cursor: number;
}

let datasetWindowSets: DatasetWindowSet[] = [];

function shuffleSet(ds: DatasetWindowSet) {
  ds.order = ds.inputs.map((_, i) => i);
  for (let i = ds.order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [ds.order[i], ds.order[j]] = [ds.order[j], ds.order[i]];
  }
  ds.cursor = 0;
}

function buildDatasetWindowSet(
  rawCorpus: string,
  contextSize: number,
): DatasetWindowSet | null {
  const normalized = normalizeText(rawCorpus);
  const tokens = tokenize(normalized.length > 0 ? normalized : " ", tokenization);
  if (tokens.length === 0) return null;
  // Build windows using the shared global vocab (stoi) so all datasets map to
  // the same token ids — the vocab must have been built from the combined
  // corpus before calling this function.
  const w = makeWindows(tokens, stoi, contextSize);
  if (w.inputs.length === 0) return null;
  const ds: DatasetWindowSet = { inputs: w.inputs, targets: w.targets, order: [], cursor: 0 };
  shuffleSet(ds);
  return ds;
}

// ── Global training state ───────────────────────────────────────────────────
let net: TextNetwork | null = null;
let inputs: number[][] = [];
let targets: number[] = [];
let vocab: string[] = [];
let stoi: Record<string, number> = {};
let corpus = "";
let corpusTokens: string[] = [];
let tokenization: Tokenization = "char";
let temperature = 0.8;
let topK = 0;

let epoch = 0;
let lossEMA = 0;
let trainedSamples = 0;
let trainStartedAt = 0;
let timer: ReturnType<typeof setInterval> | null = null;

let order: number[] = [];
let cursor = 0;

function shuffleOrder() {
  order = inputs.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  cursor = 0;
}

function init(opts: InitOpts) {
  tokenization = opts.tokenization ?? "char";
  temperature = opts.temperature;
  if (typeof opts.topK === "number") topK = opts.topK;

  const rawDatasets: string[] = (opts.corpusDatasets && opts.corpusDatasets.length > 0)
    ? opts.corpusDatasets
    : [opts.corpus];

  // Build vocab from the FULL combined corpus so all per-dataset windows
  // share the same token-id space.
  const combinedRaw = rawDatasets.join(" ");
  const normalized = normalizeText(combinedRaw);
  corpus = normalized.length > 0 ? normalized : " ";
  corpusTokens = tokenize(corpus, tokenization);
  if (corpusTokens.length === 0) corpusTokens = [" "];

  const v = buildVocab(corpusTokens);
  vocab = v.chars;
  stoi = v.stoi;

  // Full combined windows (used for single-dataset fallback and snapshots).
  const w = makeWindows(corpusTokens, stoi, opts.contextSize);
  inputs = w.inputs;
  targets = w.targets;

  // Build per-dataset window sets when multiple corpora are given.
  if (rawDatasets.length > 1) {
    datasetWindowSets = rawDatasets
      .map((raw) => buildDatasetWindowSet(raw, opts.contextSize))
      .filter((ds): ds is DatasetWindowSet => ds !== null);
  } else {
    datasetWindowSets = [];
  }

  net = new TextNetwork({
    vocabSize: vocab.length,
    contextSize: opts.contextSize,
    hiddenSize: opts.hiddenSize,
    learningRate: opts.learningRate,
  });

  epoch = 0;
  lossEMA = Math.log(Math.max(2, vocab.length));
  trainedSamples = 0;
  trainStartedAt = 0;
  shuffleOrder();
  emitSnapshot();
}

// ── Training epoch ──────────────────────────────────────────────────────────
// When datasetWindowSets has > 1 entry (multiple corpora), we interleave
// one step from each dataset per inner iteration.  This implements the
// "jigsaw-puzzle" training strategy that prevents Catastrophic Forgetting:
// each gradient update sees all knowledge sources simultaneously, so the
// model cannot overwrite one domain's weights while learning another.
//
// When only one dataset exists (or the model was loaded from weights), the
// original shuffle-cursor approach is used unchanged.
function trainEpoch() {
  if (!net) return;
  if (trainStartedAt === 0) trainStartedAt = performance.now();

  if (datasetWindowSets.length > 1) {
    // Interleaved multi-dataset path.
    // We compute steps as the average dataset size so every epoch is
    // roughly the same cost regardless of dataset count.
    const totalInputs = datasetWindowSets.reduce((acc, ds) => acc + ds.inputs.length, 0);
    const avgSize = Math.max(1, Math.ceil(totalInputs / datasetWindowSets.length));
    let total = 0;
    let count = 0;

    for (let step = 0; step < avgSize; step++) {
      for (const ds of datasetWindowSets) {
        if (ds.cursor >= ds.order.length) shuffleSet(ds);
        const idx = ds.order[ds.cursor++];
        total += net.trainStep(ds.inputs[idx], ds.targets[idx]);
        trainedSamples++;
        count++;
      }
    }

    const avg = count > 0 ? total / count : 0;
    lossEMA = epoch === 0 ? avg : lossEMA * 0.7 + avg * 0.3;
    epoch++;
  } else {
    // Original single-dataset shuffle path.
    if (inputs.length === 0) return;
    let total = 0;
    for (let i = 0; i < inputs.length; i++) {
      if (cursor >= order.length) shuffleOrder();
      const idx = order[cursor++];
      total += net.trainStep(inputs[idx], targets[idx]);
      trainedSamples++;
    }
    const avg = total / inputs.length;
    lossEMA = epoch === 0 ? avg : lossEMA * 0.7 + avg * 0.3;
    epoch++;
  }
}

function pickSeedTokens(): string[] {
  if (!net) return [];
  const ctxSize = net.config.contextSize;
  if (corpusTokens.length <= ctxSize) return corpusTokens.slice();
  const max = corpusTokens.length - ctxSize;
  const start = Math.floor(Math.random() * (max + 1));
  return corpusTokens.slice(start, start + ctxSize);
}

function makeSample(length = 32): string {
  if (!net) return "";
  const seed = pickSeedTokens();
  const generated = generateTokens(net, vocab, stoi, seed, length, temperature);
  return joinTokens([...seed, ...generated], tokenization);
}

function tokensPerSecond(): number {
  if (trainStartedAt === 0) return 0;
  const elapsed = (performance.now() - trainStartedAt) / 1000;
  if (elapsed < 0.05) return 0;
  return trainedSamples / elapsed;
}

function emitSnapshot() {
  if (!net) return;
  postMessage({
    type: "snapshot",
    epoch,
    loss: lossEMA,
    paramCount: net.paramCount(),
    vocabSize: vocab.length,
    contextSize: net.config.contextSize,
    hiddenSize: net.config.hiddenSize,
    sample: makeSample(32),
    tokensPerSecond: tokensPerSecond(),
    trainedSamples,
  });
}

function play(epochsPerSecond: number) {
  pause();
  const intervalMs = Math.max(8, 1000 / Math.max(1, epochsPerSecond));
  timer = setInterval(() => {
    trainEpoch();
    emitSnapshot();
  }, intervalMs);
}

function pause() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

self.onmessage = (e: MessageEvent) => {
  const msg = e.data;
  switch (msg.type) {
    case "init":
      init(msg.opts);
      break;
    case "reset":
      pause();
      init(msg.opts);
      break;
    case "play":
      play(msg.epochsPerSecond ?? 5);
      break;
    case "pause":
      pause();
      break;
    case "config":
      if (msg.partial?.learningRate != null && net) {
        net.config.learningRate = msg.partial.learningRate;
      }
      if (msg.partial?.temperature != null) {
        temperature = msg.partial.temperature;
      }
      if (msg.partial?.topK != null) {
        topK = msg.partial.topK;
      }
      break;
    case "generateStream": {
      if (!net) {
        postMessage({ type: "streamToken", id: msg.id, token: "", done: true, fullText: "(model not ready — try training first)" });
        break;
      }
      const streamSeedTokens = tokenize(normalizeText(msg.seed ?? ""), tokenization);
      const streamLength = msg.length ?? 300;
      const streamTemp = msg.temperature ?? temperature;
      const streamCtxSize = net.config.contextSize;
      const streamPadId = stoi[PAD_TOKEN] ?? stoi[" "] ?? 0;
      const streamEosId = stoi[EOS_TOKEN];
      const streamSeedIds = streamSeedTokens.map((t) => stoi[t] !== undefined ? stoi[t] : streamPadId);
      let streamCtx: number[];
      if (streamSeedIds.length >= streamCtxSize) {
        streamCtx = streamSeedIds.slice(-streamCtxSize);
      } else {
        streamCtx = new Array(streamCtxSize - streamSeedIds.length).fill(streamPadId).concat(streamSeedIds);
      }
      const streamGenerated: string[] = [];
      let streamText = "";
      for (let i = 0; i < streamLength; i++) {
        const { probs } = net.forward(streamCtx);
        const filtered = applyTopK(probs, topK);
        const next = sampleFromProbs(filtered, streamTemp);
        if (streamEosId !== undefined && next === streamEosId) break;
        const tok = vocab[next];
        streamCtx = streamCtx.slice(1).concat(next);
        if (tok === PAD_TOKEN) continue;
        streamGenerated.push(tok);
        streamText = joinTokens(streamGenerated, tokenization);
        const streamCut = findStopCut(streamText);
        if (streamCut !== -1) {
          streamText = streamText.slice(0, streamCut).replace(/\s+$/, "");
          break;
        }
        const displayTok = tokenization === "char" ? tok : tok + " ";
        postMessage({ type: "streamToken", id: msg.id, token: displayTok, done: false, fullText: "" });
      }
      const streamCleaned = streamText.replace(/^bot\s+/i, "").trim();
      postMessage({ type: "streamToken", id: msg.id, token: "", done: true, fullText: streamCleaned || "(silence)" });
      break;
    }
    case "generate": {
      if (!net) {
        postMessage({ type: "generation", id: msg.id, text: "(model not ready — try training first)" });
        break;
      }
      const seedTokens = tokenize(normalizeText(msg.seed ?? ""), tokenization);
      const length = msg.length ?? 300;
      const temp = msg.temperature ?? temperature;
      const ctxSize = net.config.contextSize;
      const padId = stoi[PAD_TOKEN] ?? stoi[" "] ?? 0;
      const eosId = stoi[EOS_TOKEN];
      const seedIds = seedTokens.map((t) => stoi[t] !== undefined ? stoi[t] : padId);
      let ctx: number[];
      if (seedIds.length >= ctxSize) {
        ctx = seedIds.slice(-ctxSize);
      } else {
        ctx = new Array(ctxSize - seedIds.length).fill(padId).concat(seedIds);
      }
      const generated: string[] = [];
      let text = "";
      for (let i = 0; i < length; i++) {
        const { probs } = net.forward(ctx);
        const filtered = applyTopK(probs, topK);
        const next = sampleFromProbs(filtered, temp);
        if (eosId !== undefined && next === eosId) break;
        const tok = vocab[next];
        ctx = ctx.slice(1).concat(next);
        if (tok === PAD_TOKEN) continue;
        generated.push(tok);
        text = joinTokens(generated, tokenization);
        const cut = findStopCut(text);
        if (cut !== -1) {
          text = text.slice(0, cut).replace(/\s+$/, "");
          break;
        }
      }
      const cleaned = text.replace(/^bot\s+/i, "").trim();
      postMessage({ type: "generation", id: msg.id, text: cleaned });
      break;
    }
    case "loadWeights": {
      pause();
      const w = msg.payload;
      if (!w || !w.config || !w.weights || !w.vocab) break;
      if (w.tokenization === "word" || w.tokenization === "char") {
        tokenization = w.tokenization;
      } else {
        tokenization = inferTokenizationFromVocab(w.vocab as string[]);
      }
      const restoredCorpus =
        typeof w.corpus === "string" && w.corpus.length > 0
          ? w.corpus
          : (w.vocab as string[]).join(tokenization === "word" ? " " : "");
      corpus = restoredCorpus;
      corpusTokens = tokenize(corpus, tokenization);
      if (corpusTokens.length === 0) corpusTokens = [" "];
      vocab = w.vocab as string[];
      stoi = {};
      for (let i = 0; i < vocab.length; i++) stoi[vocab[i]] = i;
      const wins = makeWindows(corpusTokens, stoi, w.config.contextSize);
      inputs = wins.inputs;
      targets = wins.targets;
      // Loaded models always use the single-dataset path for training.
      // When the user adds more datasets, the worker will be reset with
      // per-dataset corpora via the "reset" message.
      datasetWindowSets = [];
      net = new TextNetwork({
        vocabSize: w.config.vocabSize,
        contextSize: w.config.contextSize,
        hiddenSize: w.config.hiddenSize,
        learningRate: w.config.learningRate,
      });
      for (let j = 0; j < net.W1.length; j++) {
        net.W1[j] = Float32Array.from(w.weights.W1[j]);
      }
      net.b1 = Float32Array.from(w.weights.b1);
      for (let i = 0; i < net.W2.length; i++) {
        net.W2[i] = Float32Array.from(w.weights.W2[i]);
      }
      net.b2 = Float32Array.from(w.weights.b2);
      if (typeof w.temperature === "number") temperature = w.temperature;
      epoch = typeof w.epoch === "number" ? w.epoch : 0;
      lossEMA = typeof w.loss === "number" ? w.loss : 0;
      trainedSamples = 0;
      trainStartedAt = 0;
      shuffleOrder();
      emitSnapshot();
      break;
    }
    case "exportModel": {
      if (!net) {
        postMessage({ type: "exportModel", id: msg.id, payload: null });
        break;
      }
      const payload = {
        kind: "char-lm",
        vocab,
        tokenization,
        config: net.config,
        epoch,
        loss: lossEMA,
        weights: {
          W1: net.W1.map((r) => Array.from(r)),
          b1: Array.from(net.b1),
          W2: net.W2.map((r) => Array.from(r)),
          b2: Array.from(net.b2),
        },
      };
      postMessage({ type: "exportModel", id: msg.id, payload });
      break;
    }
  }
};
