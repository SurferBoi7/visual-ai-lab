/// <reference lib="webworker" />
import {
  NeuralNetwork,
  type NetworkConfig,
  type DataPoint,
  generateSpiral,
  generateCircle,
  generateXor,
} from "./nn";

type DatasetKind = "spiral" | "circle" | "xor";

interface InitMsg {
  type: "init";
  config: NetworkConfig;
  dataset: DatasetKind;
}
interface PlayMsg {
  type: "play";
  epochsPerSecond: number;
}
interface PauseMsg {
  type: "pause";
}
interface StepMsg {
  type: "step";
  epochs: number;
}
interface ResetMsg {
  type: "reset";
  config: NetworkConfig;
  dataset: DatasetKind;
}
interface ConfigMsg {
  type: "config";
  partial: Partial<NetworkConfig>;
}
interface DatasetMsg {
  type: "dataset";
  dataset: DatasetKind;
}

type IncomingMsg =
  | InitMsg
  | PlayMsg
  | PauseMsg
  | StepMsg
  | ResetMsg
  | ConfigMsg
  | DatasetMsg;

let net: NeuralNetwork | null = null;
let data: DataPoint[] = [];
let dataset: DatasetKind = "circle";
let epoch = 0;
let interval: ReturnType<typeof setInterval> | null = null;
let lastLoss = 0;
let lastAcc = 0;

function buildDataset(kind: DatasetKind): DataPoint[] {
  if (kind === "spiral") return generateSpiral(80);
  if (kind === "xor") return generateXor(200);
  return generateCircle(200);
}

function trainOneEpoch() {
  if (!net) return;
  // Shuffle indices
  const order: number[] = [];
  for (let i = 0; i < data.length; i++) order.push(i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  let totalLoss = 0;
  let correct = 0;
  for (const idx of order) {
    const p = data[idx];
    const target = [p.label];
    totalLoss += net.trainSample([p.x, p.y], target);
    const out = net.activationsCache[net.activationsCache.length - 1][0];
    const pred = out > 0.5 ? 1 : 0;
    if (pred === p.label) correct++;
  }
  epoch++;
  lastLoss = totalLoss / data.length;
  lastAcc = correct / data.length;
}

function buildDecisionGrid(res: number): Float32Array {
  const grid = new Float32Array(res * res);
  if (!net) return grid;
  for (let yi = 0; yi < res; yi++) {
    for (let xi = 0; xi < res; xi++) {
      const x = (xi / (res - 1)) * 2 - 1;
      const y = (yi / (res - 1)) * 2 - 1;
      const out = net.predict([x, y]);
      grid[yi * res + xi] = out[0];
    }
  }
  return grid;
}

function postSnapshot() {
  if (!net) return;
  const snap = net.snapshot();
  const grid = buildDecisionGrid(28);
  const w = self as unknown as Worker;
  w.postMessage({
    type: "snapshot",
    epoch,
    loss: lastLoss,
    accuracy: lastAcc,
    paramCount: snap.paramCount,
    layers: snap.layers,
    weights: snap.weights,
    biases: snap.biases,
    data,
    grid,
    gridRes: 28,
  });
}

function startLoop(epochsPerSecond: number) {
  if (interval) clearInterval(interval);
  // Run a training tick at ~30Hz, doing the right number of epochs per tick.
  const tickHz = 30;
  const epochsPerTick = Math.max(1, Math.round(epochsPerSecond / tickHz));
  const intervalMs = 1000 / tickHz;
  interval = setInterval(() => {
    if (!net) return;
    for (let i = 0; i < epochsPerTick; i++) trainOneEpoch();
    postSnapshot();
  }, intervalMs);
}

function stopLoop() {
  if (interval) {
    clearInterval(interval);
    interval = null;
  }
}

self.onmessage = (e: MessageEvent<IncomingMsg>) => {
  const msg = e.data;
  switch (msg.type) {
    case "init":
    case "reset": {
      stopLoop();
      dataset = msg.dataset;
      data = buildDataset(dataset);
      net = new NeuralNetwork(msg.config);
      epoch = 0;
      lastLoss = 0;
      lastAcc = 0;
      postSnapshot();
      break;
    }
    case "config": {
      if (!net) return;
      // Updating learning rate is safe in-place.
      if (msg.partial.learningRate !== undefined) {
        net.config.learningRate = msg.partial.learningRate;
      }
      // Architecture / activation changes require a rebuild — caller should send "reset".
      break;
    }
    case "dataset": {
      stopLoop();
      dataset = msg.dataset;
      data = buildDataset(dataset);
      epoch = 0;
      lastLoss = 0;
      lastAcc = 0;
      postSnapshot();
      break;
    }
    case "play": {
      startLoop(msg.epochsPerSecond);
      break;
    }
    case "pause": {
      stopLoop();
      break;
    }
    case "step": {
      if (!net) return;
      for (let i = 0; i < msg.epochs; i++) trainOneEpoch();
      postSnapshot();
      break;
    }
  }
};

export {};
