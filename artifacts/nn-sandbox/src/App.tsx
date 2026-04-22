import { useEffect, useMemo, useRef, useState } from "react";
import {
  Play,
  Pause,
  RefreshCcw,
  Save,
  Plus,
  Minus,
  Brain,
  Zap,
  Layers,
  Activity,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Card } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Toaster } from "@/components/ui/toaster";
import { useToast } from "@/hooks/use-toast";
import { NetworkCanvas } from "@/components/NetworkCanvas";
import { DataView } from "@/components/DataView";
import type { Activation, NetworkConfig, DataPoint } from "@/lib/nn";

type DatasetKind = "spiral" | "circle" | "xor";

interface SnapshotMsg {
  type: "snapshot";
  epoch: number;
  loss: number;
  accuracy: number;
  paramCount: number;
  layers: number[];
  weights: number[][][];
  biases: number[][];
  data: DataPoint[];
  grid: Float32Array;
  gridRes: number;
}

const MAX_PARAMS = 1000;

function estimateParams(layers: number[]): number {
  let p = 0;
  for (let i = 1; i < layers.length; i++) {
    p += layers[i - 1] * layers[i] + layers[i];
  }
  return p;
}

export default function App() {
  const { toast } = useToast();
  const workerRef = useRef<Worker | null>(null);

  const [hidden, setHidden] = useState<number[]>([6, 4]);
  const [activation, setActivation] = useState<Activation>("tanh");
  const [learningRate, setLearningRate] = useState(0.05);
  const [dataset, setDataset] = useState<DatasetKind>("circle");
  const [playing, setPlaying] = useState(false);
  const [epochsPerSecond, setEpochsPerSecond] = useState(10);

  const [snap, setSnap] = useState<SnapshotMsg | null>(null);

  const layers = useMemo(() => [2, ...hidden, 1], [hidden]);
  const estimatedParams = useMemo(() => estimateParams(layers), [layers]);

  // Initialize / re-init worker on architecture changes.
  useEffect(() => {
    const worker = new Worker(
      new URL("./lib/trainer.worker.ts", import.meta.url),
      { type: "module" },
    );
    workerRef.current = worker;
    worker.onmessage = (e: MessageEvent<SnapshotMsg>) => {
      if (e.data.type === "snapshot") setSnap(e.data);
    };
    const config: NetworkConfig = {
      inputSize: 2,
      hiddenLayers: hidden,
      outputSize: 1,
      activation,
      learningRate,
    };
    worker.postMessage({ type: "init", config, dataset });
    return () => {
      worker.terminate();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const reinit = (
    nextHidden = hidden,
    nextActivation = activation,
    nextDataset = dataset,
    nextLR = learningRate,
  ) => {
    setPlaying(false);
    const worker = workerRef.current;
    if (!worker) return;
    const config: NetworkConfig = {
      inputSize: 2,
      hiddenLayers: nextHidden,
      outputSize: 1,
      activation: nextActivation,
      learningRate: nextLR,
    };
    worker.postMessage({ type: "reset", config, dataset: nextDataset });
  };

  const handleAddLayer = () => {
    if (hidden.length >= 4) {
      toast({ title: "Max 4 hidden layers", description: "Keep it simple." });
      return;
    }
    const next = [...hidden, 4];
    if (estimateParams([2, ...next, 1]) > MAX_PARAMS) {
      toast({
        title: "Parameter budget exceeded",
        description: `Network would exceed ${MAX_PARAMS} parameters.`,
      });
      return;
    }
    setHidden(next);
    reinit(next);
  };

  const handleRemoveLayer = () => {
    if (hidden.length <= 1) return;
    const next = hidden.slice(0, -1);
    setHidden(next);
    reinit(next);
  };

  const handleNeuronChange = (idx: number, delta: number) => {
    const next = hidden.slice();
    next[idx] = Math.max(1, Math.min(12, next[idx] + delta));
    if (estimateParams([2, ...next, 1]) > MAX_PARAMS) {
      toast({
        title: "Parameter budget exceeded",
        description: `Cannot exceed ${MAX_PARAMS} parameters.`,
      });
      return;
    }
    setHidden(next);
    reinit(next);
  };

  const handleActivation = (v: string) => {
    const a = v as Activation;
    setActivation(a);
    reinit(hidden, a);
  };

  const handleDataset = (v: string) => {
    const d = v as DatasetKind;
    setDataset(d);
    reinit(hidden, activation, d);
  };

  const handleLR = (vals: number[]) => {
    const v = vals[0];
    setLearningRate(v);
    workerRef.current?.postMessage({
      type: "config",
      partial: { learningRate: v },
    });
  };

  const handleSpeed = (vals: number[]) => {
    const v = vals[0];
    setEpochsPerSecond(v);
    if (playing) {
      workerRef.current?.postMessage({ type: "play", epochsPerSecond: v });
    }
  };

  const handlePlay = () => {
    if (playing) {
      workerRef.current?.postMessage({ type: "pause" });
      setPlaying(false);
    } else {
      workerRef.current?.postMessage({ type: "play", epochsPerSecond });
      setPlaying(true);
    }
  };

  const handleReset = () => {
    reinit();
  };

  const handleSave = () => {
    if (!snap) return;
    const blob = new Blob(
      [
        JSON.stringify(
          {
            architecture: { layers, activation },
            learningRate,
            epoch: snap.epoch,
            loss: snap.loss,
            accuracy: snap.accuracy,
            weights: snap.weights,
            biases: snap.biases,
          },
          null,
          2,
        ),
      ],
      { type: "application/json" },
    );
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `nn-sandbox-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: "Network saved", description: "Weights exported as JSON." });
  };

  const overBudget = estimatedParams > MAX_PARAMS;

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Top bar */}
      <header className="border-b border-border/60 bg-card/40 backdrop-blur sticky top-0 z-10">
        <div className="px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="size-8 rounded-md bg-gradient-to-br from-sky-500 to-emerald-500 flex items-center justify-center">
              <Brain className="size-4 text-slate-900" />
            </div>
            <div>
              <div className="text-sm font-semibold tracking-tight">
                Visual Neural Network Sandbox
              </div>
              <div className="text-[11px] text-muted-foreground">
                Build, train, and watch an MLP learn — entirely in the browser.
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="secondary"
              onClick={handleReset}
              className="gap-1.5"
            >
              <RefreshCcw className="size-3.5" />
              Reset
            </Button>
            <Button size="sm" variant="secondary" onClick={handleSave} className="gap-1.5">
              <Save className="size-3.5" />
              Save
            </Button>
            <Button
              size="sm"
              onClick={handlePlay}
              className="gap-1.5 min-w-[88px]"
            >
              {playing ? (
                <>
                  <Pause className="size-3.5" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="size-3.5" />
                  Train
                </>
              )}
            </Button>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-12 gap-4 p-4">
        {/* Sidebar */}
        <aside className="col-span-12 lg:col-span-3 space-y-3">
          <Card className="p-4 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Layers className="size-4 text-sky-400" />
                <span className="text-sm font-semibold">Architecture</span>
              </div>
              <span
                className={`text-[11px] tabular-nums ${
                  overBudget ? "text-red-400" : "text-muted-foreground"
                }`}
              >
                {estimatedParams} / {MAX_PARAMS} params
              </span>
            </div>

            <div className="space-y-2">
              {hidden.map((n, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded-md border border-border/60 bg-background/60 px-3 py-2"
                >
                  <span className="text-xs text-muted-foreground">
                    Hidden Layer {i + 1}
                  </span>
                  <div className="flex items-center gap-1.5">
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-6"
                      onClick={() => handleNeuronChange(i, -1)}
                    >
                      <Minus className="size-3" />
                    </Button>
                    <span className="text-sm font-medium w-6 text-center tabular-nums">
                      {n}
                    </span>
                    <Button
                      size="icon"
                      variant="ghost"
                      className="size-6"
                      onClick={() => handleNeuronChange(i, +1)}
                    >
                      <Plus className="size-3" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex gap-2">
              <Button
                size="sm"
                variant="secondary"
                onClick={handleAddLayer}
                className="flex-1 gap-1.5"
              >
                <Plus className="size-3.5" /> Add Layer
              </Button>
              <Button
                size="sm"
                variant="secondary"
                onClick={handleRemoveLayer}
                className="flex-1 gap-1.5"
                disabled={hidden.length <= 1}
              >
                <Minus className="size-3.5" /> Remove
              </Button>
            </div>
          </Card>

          <Card className="p-4 space-y-4">
            <div className="flex items-center gap-2">
              <Zap className="size-4 text-amber-400" />
              <span className="text-sm font-semibold">Hyperparameters</span>
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between">
                <span className="text-xs text-muted-foreground">
                  Learning Rate
                </span>
                <span className="text-xs tabular-nums">
                  {learningRate.toFixed(3)}
                </span>
              </div>
              <Slider
                min={0.001}
                max={0.5}
                step={0.001}
                value={[learningRate]}
                onValueChange={handleLR}
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between">
                <span className="text-xs text-muted-foreground">
                  Train Speed (epochs/sec)
                </span>
                <span className="text-xs tabular-nums">{epochsPerSecond}</span>
              </div>
              <Slider
                min={1}
                max={120}
                step={1}
                value={[epochsPerSecond]}
                onValueChange={handleSpeed}
              />
            </div>

            <div className="space-y-1.5">
              <span className="text-xs text-muted-foreground">
                Activation Function
              </span>
              <Select value={activation} onValueChange={handleActivation}>
                <SelectTrigger className="h-9">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="tanh">Tanh</SelectItem>
                  <SelectItem value="relu">ReLU</SelectItem>
                  <SelectItem value="sigmoid">Sigmoid</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <span className="text-xs text-muted-foreground">Dataset</span>
              <Select value={dataset} onValueChange={handleDataset}>
                <SelectTrigger className="h-9">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="circle">Circle</SelectItem>
                  <SelectItem value="xor">XOR Quadrants</SelectItem>
                  <SelectItem value="spiral">Spiral</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </Card>

          <Card className="p-4 space-y-3">
            <div className="flex items-center gap-2">
              <Activity className="size-4 text-emerald-400" />
              <span className="text-sm font-semibold">Training Stats</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              <Stat label="Epoch" value={snap?.epoch ?? 0} />
              <Stat
                label="Loss"
                value={snap ? snap.loss.toFixed(3) : "—"}
              />
              <Stat
                label="Acc"
                value={snap ? `${(snap.accuracy * 100).toFixed(0)}%` : "—"}
              />
            </div>
          </Card>
        </aside>

        {/* Center stage */}
        <main className="col-span-12 lg:col-span-9 space-y-4">
          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-sm font-semibold">Network Topology</div>
                <div className="text-[11px] text-muted-foreground">
                  Blue = positive weight · Red = negative · Thickness = magnitude
                </div>
              </div>
              <div className="text-[11px] text-muted-foreground">
                {layers.join(" → ")} · {snap?.paramCount ?? estimatedParams}{" "}
                params
              </div>
            </div>
            <div className="rounded-lg bg-[#0b1220] border border-border/60 overflow-hidden">
              <NetworkCanvas
                layers={snap?.layers ?? layers}
                weights={snap?.weights ?? []}
                width={920}
                height={360}
              />
            </div>
          </Card>

          <Card className="p-4">
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="text-sm font-semibold">
                  2D Point Classifier
                </div>
                <div className="text-[11px] text-muted-foreground">
                  The network learns to color the plane based on (x, y) inputs.
                </div>
              </div>
              <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <span className="size-2.5 rounded-full bg-sky-400" /> Class A
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="size-2.5 rounded-full bg-emerald-400" />{" "}
                  Class B
                </span>
              </div>
            </div>
            <div className="flex justify-center">
              <DataView
                data={snap?.data ?? []}
                grid={snap?.grid ?? null}
                gridRes={snap?.gridRes ?? 28}
                size={420}
              />
            </div>
          </Card>
        </main>
      </div>
      <Toaster />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border border-border/60 bg-background/60 px-2 py-2 text-center">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="text-sm font-semibold tabular-nums">{value}</div>
    </div>
  );
}
