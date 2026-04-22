export type Activation = "relu" | "sigmoid" | "tanh";

export interface NetworkConfig {
  inputSize: number;
  hiddenLayers: number[];
  outputSize: number;
  activation: Activation;
  learningRate: number;
}

export interface NetworkSnapshot {
  layers: number[];
  weights: number[][][];
  biases: number[][];
  activations: number[][];
  epoch: number;
  loss: number;
  accuracy: number;
  paramCount: number;
}

const act = {
  relu: (x: number) => (x > 0 ? x : 0),
  reluD: (y: number) => (y > 0 ? 1 : 0),
  sigmoid: (x: number) => 1 / (1 + Math.exp(-x)),
  sigmoidD: (y: number) => y * (1 - y),
  tanh: (x: number) => Math.tanh(x),
  tanhD: (y: number) => 1 - y * y,
};

function applyActivation(x: number, fn: Activation): number {
  if (fn === "relu") return act.relu(x);
  if (fn === "sigmoid") return act.sigmoid(x);
  return act.tanh(x);
}

function activationDerivative(y: number, fn: Activation): number {
  if (fn === "relu") return act.reluD(y);
  if (fn === "sigmoid") return act.sigmoidD(y);
  return act.tanhD(y);
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export class NeuralNetwork {
  config: NetworkConfig;
  layers: number[];
  weights: number[][][]; // weights[layer][to][from]
  biases: number[][]; // biases[layer][to]
  activationsCache: number[][]; // activations per layer (including input)

  constructor(config: NetworkConfig) {
    this.config = { ...config };
    this.layers = [config.inputSize, ...config.hiddenLayers, config.outputSize];
    this.weights = [];
    this.biases = [];
    this.activationsCache = this.layers.map((n) => new Array(n).fill(0));
    this.initializeWeights();
  }

  initializeWeights() {
    this.weights = [];
    this.biases = [];
    for (let l = 1; l < this.layers.length; l++) {
      const fanIn = this.layers[l - 1];
      const fanOut = this.layers[l];
      // He / Xavier-ish init
      const scale =
        this.config.activation === "relu"
          ? Math.sqrt(2 / fanIn)
          : Math.sqrt(1 / fanIn);
      const layerW: number[][] = [];
      const layerB: number[] = [];
      for (let j = 0; j < fanOut; j++) {
        const row: number[] = [];
        for (let i = 0; i < fanIn; i++) {
          row.push((Math.random() * 2 - 1) * scale);
        }
        layerW.push(row);
        layerB.push(0);
      }
      this.weights.push(layerW);
      this.biases.push(layerB);
    }
  }

  paramCount(): number {
    let total = 0;
    for (let l = 0; l < this.weights.length; l++) {
      total += this.weights[l].length * this.weights[l][0].length;
      total += this.biases[l].length;
    }
    return total;
  }

  forward(input: number[]): number[] {
    this.activationsCache[0] = input.slice();
    let current = input;
    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const isLast = l === this.weights.length - 1;
      const z: number[] = new Array(W.length);
      for (let j = 0; j < W.length; j++) {
        let sum = b[j];
        const row = W[j];
        for (let i = 0; i < current.length; i++) {
          sum += row[i] * current[i];
        }
        z[j] = sum;
      }
      let next: number[];
      if (isLast && this.config.outputSize > 1) {
        next = softmax(z);
      } else if (isLast && this.config.outputSize === 1) {
        next = z.map((v) => act.sigmoid(v));
      } else {
        next = z.map((v) => applyActivation(v, this.config.activation));
      }
      this.activationsCache[l + 1] = next;
      current = next;
    }
    return current;
  }

  /** One SGD step on a single sample. Returns loss for that sample. */
  trainSample(input: number[], target: number[]): number {
    const output = this.forward(input);
    const lr = this.config.learningRate;

    // Compute output layer delta (cross-entropy with softmax/sigmoid → simple form)
    let delta: number[] = output.map((o, i) => o - target[i]);

    // Loss: cross-entropy
    let loss = 0;
    for (let i = 0; i < output.length; i++) {
      const o = Math.min(Math.max(output[i], 1e-9), 1 - 1e-9);
      loss += -(target[i] * Math.log(o) + (1 - target[i]) * Math.log(1 - o));
    }

    // Backpropagate
    for (let l = this.weights.length - 1; l >= 0; l--) {
      const prevAct = this.activationsCache[l];
      const W = this.weights[l];
      const b = this.biases[l];

      // gradient w.r.t. weights and biases
      const newDelta: number[] = new Array(prevAct.length).fill(0);
      for (let j = 0; j < W.length; j++) {
        const d = delta[j];
        for (let i = 0; i < prevAct.length; i++) {
          newDelta[i] += W[j][i] * d;
          W[j][i] -= lr * d * prevAct[i];
        }
        b[j] -= lr * d;
      }

      if (l > 0) {
        const layerOutput = this.activationsCache[l];
        for (let i = 0; i < newDelta.length; i++) {
          newDelta[i] *= activationDerivative(
            layerOutput[i],
            this.config.activation,
          );
        }
        delta = newDelta;
      }
    }
    return loss;
  }

  predict(input: number[]): number[] {
    return this.forward(input);
  }

  snapshot(): NetworkSnapshot {
    return {
      layers: this.layers.slice(),
      weights: this.weights.map((m) => m.map((r) => r.slice())),
      biases: this.biases.map((b) => b.slice()),
      activations: this.activationsCache.map((a) => a.slice()),
      epoch: 0,
      loss: 0,
      accuracy: 0,
      paramCount: this.paramCount(),
    };
  }
}

export interface DataPoint {
  x: number;
  y: number;
  label: number; // 0 or 1
}

export function generateSpiral(n = 100): DataPoint[] {
  const points: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    const r = i / n;
    const t = ((1.75 * i) / n) * 2 * Math.PI + Math.random() * 0.2;
    points.push({
      x: r * Math.sin(t),
      y: r * Math.cos(t),
      label: 0,
    });
    points.push({
      x: -r * Math.sin(t),
      y: -r * Math.cos(t),
      label: 1,
    });
  }
  return points;
}

export function generateCircle(n = 200): DataPoint[] {
  const points: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    const r = Math.sqrt(x * x + y * y);
    points.push({ x, y, label: r < 0.5 ? 0 : 1 });
  }
  return points;
}

export function generateXor(n = 200): DataPoint[] {
  const points: DataPoint[] = [];
  for (let i = 0; i < n; i++) {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    points.push({ x, y, label: x * y > 0 ? 0 : 1 });
  }
  return points;
}
