// Pure-TypeScript character-level language model.
// Architecture: one-hot encoded context window -> tanh hidden layer -> softmax over vocab.
// Optimizer: SGD. Loss: cross-entropy.

export interface TextNetConfig {
  vocabSize: number;
  contextSize: number;
  hiddenSize: number;
  learningRate: number;
}

function randn(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class TextNetwork {
  config: TextNetConfig;
  W1: Float32Array[]; // [hiddenSize][inputDim]
  b1: Float32Array;
  W2: Float32Array[]; // [vocabSize][hiddenSize]
  b2: Float32Array;
  inputDim: number;

  constructor(cfg: TextNetConfig) {
    this.config = cfg;
    this.inputDim = cfg.contextSize * cfg.vocabSize;
    const s1 = Math.sqrt(2 / this.inputDim);
    const s2 = Math.sqrt(2 / cfg.hiddenSize);
    this.W1 = Array.from({ length: cfg.hiddenSize }, () => {
      const row = new Float32Array(this.inputDim);
      for (let i = 0; i < this.inputDim; i++) row[i] = randn() * s1;
      return row;
    });
    this.b1 = new Float32Array(cfg.hiddenSize);
    this.W2 = Array.from({ length: cfg.vocabSize }, () => {
      const row = new Float32Array(cfg.hiddenSize);
      for (let i = 0; i < cfg.hiddenSize; i++) row[i] = randn() * s2;
      return row;
    });
    this.b2 = new Float32Array(cfg.vocabSize);
  }

  paramCount(): number {
    const c = this.config;
    return (
      this.inputDim * c.hiddenSize +
      c.hiddenSize +
      c.hiddenSize * c.vocabSize +
      c.vocabSize
    );
  }

  forward(ctx: number[]): {
    h: Float32Array;
    probs: Float32Array;
  } {
    const { hiddenSize, vocabSize } = this.config;
    const h = new Float32Array(hiddenSize);
    // Hidden pre-activation (sparse: input is one-hot, so sum the active columns).
    for (let j = 0; j < hiddenSize; j++) {
      let s = this.b1[j];
      const row = this.W1[j];
      for (let k = 0; k < ctx.length; k++) {
        s += row[k * vocabSize + ctx[k]];
      }
      h[j] = Math.tanh(s);
    }
    // Output logits.
    const logits = new Float32Array(vocabSize);
    let max = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      let s = this.b2[i];
      const row = this.W2[i];
      for (let j = 0; j < hiddenSize; j++) s += row[j] * h[j];
      logits[i] = s;
      if (s > max) max = s;
    }
    // Softmax (numerically stable).
    let sum = 0;
    const probs = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) {
      probs[i] = Math.exp(logits[i] - max);
      sum += probs[i];
    }
    const inv = 1 / sum;
    for (let i = 0; i < vocabSize; i++) probs[i] *= inv;
    return { h, probs };
  }

  trainStep(ctx: number[], target: number): number {
    const { hiddenSize, vocabSize, learningRate: lr } = this.config;
    const { h, probs } = this.forward(ctx);

    // d/dlogits cross-entropy with softmax = probs - onehot(target).
    const dlogits = new Float32Array(vocabSize);
    for (let i = 0; i < vocabSize; i++) dlogits[i] = probs[i];
    dlogits[target] -= 1;

    // Backprop through output layer; accumulate dh; update W2, b2.
    const dh = new Float32Array(hiddenSize);
    for (let i = 0; i < vocabSize; i++) {
      const dl = dlogits[i];
      const row = this.W2[i];
      for (let j = 0; j < hiddenSize; j++) {
        dh[j] += dl * row[j];
        row[j] -= lr * dl * h[j];
      }
      this.b2[i] -= lr * dl;
    }

    // Backprop through tanh and update W1, b1 (sparse: only active one-hot columns).
    for (let j = 0; j < hiddenSize; j++) {
      const dpre = dh[j] * (1 - h[j] * h[j]);
      const row = this.W1[j];
      for (let k = 0; k < ctx.length; k++) {
        const col = k * vocabSize + ctx[k];
        row[col] -= lr * dpre;
      }
      this.b1[j] -= lr * dpre;
    }

    return -Math.log(Math.max(1e-9, probs[target]));
  }
}

export type Tokenization = "char" | "word";

// Split a corpus into tokens. Char-mode keeps every character (whitespace and
// punctuation included). Word-mode treats whitespace-separated runs of word
// characters as tokens and emits each non-word character (punctuation) as its
// own token, which keeps the vocabulary compact while still letting the model
// learn punctuation.
export function tokenize(text: string, mode: Tokenization): string[] {
  if (mode === "char") return text.split("");
  const matches = text.match(/[A-Za-z0-9_']+|[^\sA-Za-z0-9_']/g);
  return matches ?? [];
}

// Join a sequence of tokens back into a display string. Word-mode inserts a
// space between word tokens but keeps punctuation flush with the previous
// token, mirroring normal English typography.
export function joinTokens(tokens: string[], mode: Tokenization): string {
  if (mode === "char") return tokens.join("");
  let out = "";
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    const isPunct = /^[^\sA-Za-z0-9_']$/.test(t);
    const noSpaceBefore =
      i === 0 || isPunct || /^['(\[{]$/.test(tokens[i - 1] ?? "");
    out += (noSpaceBefore ? "" : " ") + t;
  }
  return out;
}

export function buildVocab(tokens: string[]): {
  chars: string[];
  stoi: Record<string, number>;
} {
  const set = new Set(tokens);
  const chars = Array.from(set).sort();
  const stoi: Record<string, number> = {};
  chars.forEach((c, i) => (stoi[c] = i));
  return { chars, stoi };
}

export function makeWindows(
  tokens: string[],
  stoi: Record<string, number>,
  contextSize: number,
): { inputs: number[][]; targets: number[] } {
  const inputs: number[][] = [];
  const targets: number[] = [];
  for (let i = 0; i + contextSize < tokens.length; i++) {
    const ctx: number[] = [];
    let ok = true;
    for (let k = 0; k < contextSize; k++) {
      const id = stoi[tokens[i + k]];
      if (id === undefined) {
        ok = false;
        break;
      }
      ctx.push(id);
    }
    const tgt = stoi[tokens[i + contextSize]];
    if (!ok || tgt === undefined) continue;
    inputs.push(ctx);
    targets.push(tgt);
  }
  return { inputs, targets };
}

export function sampleFromProbs(
  probs: Float32Array,
  temperature: number,
): number {
  if (temperature <= 0.001) {
    let best = 0;
    let bestP = -Infinity;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > bestP) {
        bestP = probs[i];
        best = i;
      }
    }
    return best;
  }
  const adjusted = new Float32Array(probs.length);
  let sum = 0;
  const invT = 1 / temperature;
  for (let i = 0; i < probs.length; i++) {
    const v = Math.pow(Math.max(1e-9, probs[i]), invT);
    adjusted[i] = v;
    sum += v;
  }
  const r = Math.random() * sum;
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += adjusted[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

// Generate a sequence of tokens from a tokenized seed. The caller is
// responsible for joining the returned tokens with `joinTokens(...)`.
export function generateTokens(
  net: TextNetwork,
  itos: string[],
  stoi: Record<string, number>,
  seedTokens: string[],
  length: number,
  temperature: number,
): string[] {
  const ctxSize = net.config.contextSize;
  const padId = stoi[" "] ?? 0;
  const seedIds = seedTokens.map((t) =>
    stoi[t] !== undefined ? stoi[t] : padId,
  );
  let ctx: number[];
  if (seedIds.length >= ctxSize) {
    ctx = seedIds.slice(-ctxSize);
  } else {
    ctx = new Array(ctxSize - seedIds.length).fill(padId).concat(seedIds);
  }
  const out: string[] = [];
  for (let i = 0; i < length; i++) {
    const { probs } = net.forward(ctx);
    const next = sampleFromProbs(probs, temperature);
    out.push(itos[next]);
    ctx = ctx.slice(1).concat(next);
  }
  return out;
}
