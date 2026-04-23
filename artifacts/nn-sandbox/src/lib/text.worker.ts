/// <reference lib="webworker" />

import {
  TextNetwork,
  buildVocab,
  makeWindows,
  generateTokens,
  joinTokens,
  tokenize,
  type Tokenization,
} from "./textnet";

interface InitOpts {
  corpus: string;
  contextSize: number;
  hiddenSize: number;
  learningRate: number;
  temperature: number;
  tokenization: Tokenization;
}

let net: TextNetwork | null = null;
let inputs: number[][] = [];
let targets: number[] = [];
let vocab: string[] = [];
let stoi: Record<string, number> = {};
let corpus = "";
let corpusTokens: string[] = [];
let tokenization: Tokenization = "char";
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
  tokenization = opts.tokenization ?? "char";
  temperature = opts.temperature;
  corpusTokens = tokenize(corpus, tokenization);
  if (corpusTokens.length === 0) corpusTokens = [" "];
  const v = buildVocab(corpusTokens);
  vocab = v.chars;
  stoi = v.stoi;
  const w = makeWindows(corpusTokens, stoi, opts.contextSize);
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
  const generated = generateTokens(
    net,
    vocab,
    stoi,
    seed,
    length,
    temperature,
  );
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
      const seedTokens = tokenize(msg.seed ?? "", tokenization);
      const generated = generateTokens(
        net,
        vocab,
        stoi,
        seedTokens,
        msg.length ?? 50,
        msg.temperature ?? temperature,
      );
      const text = joinTokens(generated, tokenization);
      postMessage({ type: "generation", id: msg.id, text });
      break;
    }
    case "loadWeights": {
      pause();
      const w = msg.payload;
      if (!w || !w.config || !w.weights || !w.vocab) break;
      // Restore the tokenization mode first so the corpus is split the same
      // way it was when this model was originally trained.
      tokenization = w.tokenization === "word" ? "word" : "char";
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
      net = new TextNetwork({
        vocabSize: w.config.vocabSize,
        contextSize: w.config.contextSize,
        hiddenSize: w.config.hiddenSize,
        learningRate: w.config.learningRate,
      });
      // Overwrite freshly-randomised parameters with the saved ones.
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
