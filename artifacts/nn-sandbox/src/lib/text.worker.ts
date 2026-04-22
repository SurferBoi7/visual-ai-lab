/// <reference lib="webworker" />

import {
  TextNetwork,
  buildVocab,
  makeWindows,
  generateText,
} from "./textnet";

interface InitOpts {
  corpus: string;
  contextSize: number;
  hiddenSize: number;
  learningRate: number;
  temperature: number;
}

let net: TextNetwork | null = null;
let inputs: number[][] = [];
let targets: number[] = [];
let vocab: string[] = [];
let stoi: Record<string, number> = {};
let corpus = "";
let temperature = 0.8;

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
  corpus = opts.corpus.length > 0 ? opts.corpus : " ";
  temperature = opts.temperature;
  const v = buildVocab(corpus);
  vocab = v.chars;
  stoi = v.stoi;
  const w = makeWindows(corpus, stoi, opts.contextSize);
  inputs = w.inputs;
  targets = w.targets;
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

function trainEpoch() {
  if (!net || inputs.length === 0) return;
  if (trainStartedAt === 0) trainStartedAt = performance.now();
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

function pickSeed(): string {
  // Pick a seed of length >= contextSize from the corpus so we always
  // bootstrap with a "real" prefix.
  if (!net) return " ";
  const ctxSize = net.config.contextSize;
  const max = Math.max(0, corpus.length - ctxSize);
  const start = Math.floor(Math.random() * (max + 1));
  return corpus.slice(start, start + ctxSize);
}

function makeSample(length = 32): string {
  if (!net) return "";
  const seed = pickSeed();
  return seed + generateText(net, vocab, stoi, seed, length, temperature);
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
      break;
    case "generate": {
      if (!net) {
        postMessage({
          type: "generation",
          id: msg.id,
          text: "(model not ready — try training first)",
        });
        break;
      }
      const text = generateText(
        net,
        vocab,
        stoi,
        msg.seed ?? "",
        msg.length ?? 50,
        msg.temperature ?? temperature,
      );
      postMessage({ type: "generation", id: msg.id, text });
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
