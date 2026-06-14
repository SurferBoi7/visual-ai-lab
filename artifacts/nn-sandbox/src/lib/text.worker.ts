/// <reference lib="webworker" />

import {
  TextNetwork,
  buildVocab,
  generateTokens,
  joinTokens,
  tokenize,
  sampleFromProbs,
  PAD_TOKEN,
  EOS_TOKEN,
  type Tokenization,
} from "./textnet";

// Inference stop sequence — when the model "starts speaking as the user", we
// cut generation off so the chat reply doesn't run into the next turn.
const STOP_SEQUENCES = ["user", " user"];

// Number of training steps to process before yielding control back to the
// browser's event loop.  Keeping this at 64 keeps each micro-batch well under
// 16 ms on a 50M-param model while still amortising the setTimeout overhead.
const CHUNK_SIZE = 64;

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

// ── Lazy-batch helpers ───────────────────────────────────────────────────────
//
// MEMORY MODEL: we never pre-compute the full sliding-window matrix.
// Instead every dataset is stored as a single flat Uint16Array of token IDs.
// Windows are sliced on-the-fly inside trainEpoch:
//
//   input  = ids.subarray(start, start + contextSize)  → Array.from(...)
//   target = ids[start + contextSize]
//
// This drops allocation from O(N × contextSize) down to O(N), so a 1.5 MB
// corpus at contextSize=512 goes from ~600 MB pre-allocated to ~3 MB.

function tokensToIds(
  tokens: string[],
  stoi: Record<string, number>,
  padId: number,
): Uint16Array {
  const ids = new Uint16Array(tokens.length);
  for (let i = 0; i < tokens.length; i++) {
    const id = stoi[tokens[i]];
    ids[i] = id !== undefined ? id : padId;
  }
  return ids;
}

// Fisher-Yates shuffle over a Uint32Array of window-start indices.
function fisherYates(arr: Uint32Array, len: number): void {
  for (let i = len - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

function makeShuffledOrder(numWindows: number): Uint32Array {
  const order = new Uint32Array(numWindows);
  for (let i = 0; i < numWindows; i++) order[i] = i;
  fisherYates(order, numWindows);
  return order;
}

// ── Per-dataset lazy state ────────────────────────────────────────────────────
// ids         — flat token-ID array for the dataset corpus
// numWindows  — ids.length - contextSize (number of valid stride-1 windows)
// order       — shuffled Uint32Array of window-start indices [0, numWindows)
// cursor      — next position in order to consume
interface DatasetLazy {
  ids: Uint16Array;
  numWindows: number;
  order: Uint32Array;
  cursor: number;
}

let datasetWindowSets: DatasetLazy[] = [];

function reshuffleDataset(ds: DatasetLazy): void {
  fisherYates(ds.order, ds.numWindows);
  ds.cursor = 0;
}

function buildDatasetLazy(
  rawCorpus: string,
  contextSize: number,
): DatasetLazy | null {
  const normalized = normalizeText(rawCorpus);
  const tokens = tokenize(normalized.length > 0 ? normalized : " ", tokenization);
  if (tokens.length <= contextSize) return null;
  const padId = stoi[PAD_TOKEN] ?? 0;
  const ids = tokensToIds(tokens, stoi, padId);
  const numWindows = ids.length - contextSize;
  if (numWindows <= 0) return null;
  return { ids, numWindows, order: makeShuffledOrder(numWindows), cursor: 0 };
}

// ── Global training state ────────────────────────────────────────────────────
let net: TextNetwork | null = null;

// Lazy corpus — single flat Uint16Array, no pre-computed window matrix.
let corpusIds: Uint16Array = new Uint16Array(0);
let numWindows = 0;
let order: Uint32Array = new Uint32Array(0);
let cursor = 0;

// corpusTokens is kept only for pickSeedTokens (generation seeding).
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

// ── Async training loop control ──────────────────────────────────────────────
let playing = false;
let generation = 0;

function shuffleOrder(): void {
  fisherYates(order, numWindows);
  cursor = 0;
}

function init(opts: InitOpts) {
  tokenization = opts.tokenization ?? "char";
  temperature = opts.temperature;
  if (typeof opts.topK === "number") topK = opts.topK;

  const rawDatasets: string[] =
    opts.corpusDatasets && opts.corpusDatasets.length > 0
      ? opts.corpusDatasets
      : [opts.corpus];

  // Build vocab from the FULL combined corpus so all datasets share the same
  // token-id space.
  const combinedRaw = rawDatasets.join(" ");
  const normalized = normalizeText(combinedRaw);
  corpus = normalized.length > 0 ? normalized : " ";
  corpusTokens = tokenize(corpus, tokenization);
  if (corpusTokens.length === 0) corpusTokens = [" "];

  const v = buildVocab(corpusTokens);
  vocab = v.chars;
  stoi = v.stoi;

  const padId = stoi[PAD_TOKEN] ?? stoi[" "] ?? 0;

  // ── Lazy corpus IDs — O(N), not O(N × ctx) ────────────────────────────────
  corpusIds = tokensToIds(corpusTokens, stoi, padId);
  numWindows = Math.max(0, corpusIds.length - opts.contextSize);
  order = makeShuffledOrder(numWindows);
  cursor = 0;

  // Build per-dataset lazy sets when multiple corpora are given.
  if (rawDatasets.length > 1) {
    datasetWindowSets = rawDatasets
      .map((raw) => buildDatasetLazy(raw, opts.contextSize))
      .filter((ds): ds is DatasetLazy => ds !== null);
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
  generation++;
  emitSnapshot();
}

// ── Async chunked training epoch ─────────────────────────────────────────────
// Windows are generated on-the-fly from the flat corpusIds / dataset ids arrays.
// Each iteration allocates only a tiny contextSize-length Array — the GC reclaims
// it immediately after backprop, so heap stays flat regardless of corpus size.
async function trainEpoch(): Promise<void> {
  if (!net) return;
  if (trainStartedAt === 0) trainStartedAt = performance.now();

  const myGen = generation;
  const contextSize = net.config.contextSize;

  const yieldToEventLoop = (): Promise<void> =>
    new Promise<void>((resolve) => setTimeout(resolve, 0));

  if (datasetWindowSets.length > 1) {
    // ── Interleaved multi-dataset path ────────────────────────────────────
    const totalInputs = datasetWindowSets.reduce(
      (acc, ds) => acc + ds.numWindows,
      0,
    );
    const avgSize = Math.max(
      1,
      Math.ceil(totalInputs / datasetWindowSets.length),
    );
    let total = 0;
    let count = 0;
    let chunkCount = 0;

    for (let step = 0; step < avgSize; step++) {
      for (const ds of datasetWindowSets) {
        if (ds.cursor >= ds.numWindows) reshuffleDataset(ds);
        const start = ds.order[ds.cursor++];
        // On-the-fly window slice — allocated here, GC'd after trainStep.
        const ctx = new Array<number>(contextSize);
        for (let k = 0; k < contextSize; k++) ctx[k] = ds.ids[start + k];
        const tgt = ds.ids[start + contextSize];
        total += net.trainStep(ctx, tgt);
        trainedSamples++;
        count++;
        chunkCount++;

        if (chunkCount >= CHUNK_SIZE) {
          chunkCount = 0;
          await yieldToEventLoop();
          if (generation !== myGen || !net) return;
        }
      }
    }

    const avg = count > 0 ? total / count : 0;
    lossEMA = epoch === 0 ? avg : lossEMA * 0.7 + avg * 0.3;
    epoch++;
  } else {
    // ── Single-dataset path ───────────────────────────────────────────────
    if (numWindows === 0) return;
    let total = 0;
    let chunkCount = 0;

    for (let i = 0; i < numWindows; i++) {
      if (cursor >= numWindows) shuffleOrder();
      const start = order[cursor++];
      // On-the-fly window slice.
      const ctx = new Array<number>(contextSize);
      for (let k = 0; k < contextSize; k++) ctx[k] = corpusIds[start + k];
      const tgt = corpusIds[start + contextSize];
      total += net.trainStep(ctx, tgt);
      trainedSamples++;
      chunkCount++;

      if (chunkCount >= CHUNK_SIZE) {
        chunkCount = 0;
        await yieldToEventLoop();
        if (generation !== myGen || !net) return;
      }
    }

    const avg = total / numWindows;
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

// ── Async training loop ───────────────────────────────────────────────────────
function play(epochsPerSecond: number) {
  pause();
  playing = true;
  const targetMs = Math.max(8, 1000 / Math.max(1, epochsPerSecond));

  const loop = async () => {
    while (playing) {
      const t0 = performance.now();
      await trainEpoch();
      if (!playing) break;
      emitSnapshot();
      const elapsed = performance.now() - t0;
      const remaining = targetMs - elapsed;
      if (remaining > 1) {
        await new Promise<void>((resolve) => setTimeout(resolve, remaining));
      }
    }
  };

  loop().catch(() => {});
}

function pause() {
  playing = false;
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
      const streamSeedIds = streamSeedTokens.map((t) =>
        stoi[t] !== undefined ? stoi[t] : streamPadId,
      );
      let streamCtx: number[];
      if (streamSeedIds.length >= streamCtxSize) {
        streamCtx = streamSeedIds.slice(-streamCtxSize);
      } else {
        streamCtx = new Array(streamCtxSize - streamSeedIds.length)
          .fill(streamPadId)
          .concat(streamSeedIds);
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
      const seedIds = seedTokens.map((t) =>
        stoi[t] !== undefined ? stoi[t] : padId,
      );
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

      // ── Lazy IDs — no makeWindows, no pre-allocated window matrix ─────────
      const padId = stoi[PAD_TOKEN] ?? stoi[" "] ?? 0;
      corpusIds = tokensToIds(corpusTokens, stoi, padId);
      numWindows = Math.max(0, corpusIds.length - w.config.contextSize);
      order = makeShuffledOrder(numWindows);
      cursor = 0;
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
      generation++;
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
