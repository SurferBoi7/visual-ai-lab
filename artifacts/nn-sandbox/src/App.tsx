import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  SlidersHorizontal,
  Network,
  LineChart,
  MessageSquare,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
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
import { SharingHub } from "@/components/SharingHub";
import { ModeToggle, type AppMode } from "@/components/ModeToggle";
import {
  LLMArchitect,
  type LLMConfig,
} from "@/components/llm/LLMArchitect";
import { ChatView, type ChatMessage } from "@/components/llm/ChatView";
import { LLMStats } from "@/components/llm/LLMStats";
import type { Activation, NetworkConfig, DataPoint } from "@/lib/nn";

type DatasetKind = "spiral" | "circle" | "xor";
type TabKey = "architect" | "brain" | "output";

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

interface TextSnapshot {
  epoch: number;
  loss: number;
  paramCount: number;
  vocabSize: number;
  contextSize: number;
  hiddenSize: number;
  sample: string;
  tokensPerSecond: number;
  trainedSamples: number;
}

const MAX_PARAMS_MLP = 1000;
const MAX_PARAMS_LLM = 5000;

const DEFAULT_CORPUS =
  "hello world. hello friend. how are you? i am good. i am a bot. i love you. you are my friend. how do you do? hello world. how are you doing today? i am a friendly bot. ";

function estimateMLPParams(layers: number[]): number {
  let p = 0;
  for (let i = 1; i < layers.length; i++) {
    p += layers[i - 1] * layers[i] + layers[i];
  }
  return p;
}

function estimateLLMParams(
  vocab: number,
  ctx: number,
  hidden: number,
): number {
  return vocab * ctx * hidden + hidden + hidden * vocab + vocab;
}

function uniqueChars(s: string): number {
  return new Set(s).size;
}

function Card({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`bg-slate-800/50 rounded-2xl border border-slate-700 backdrop-blur-md ${className}`}
    >
      {children}
    </div>
  );
}

export default function App() {
  const { toast } = useToast();
  const workerRef = useRef<Worker | null>(null);
  const textWorkerRef = useRef<Worker | null>(null);
  const pendingGenRef = useRef<
    Map<string, (text: string) => void>
  >(new Map());

  const [mode, setMode] = useState<AppMode>("mlp");

  // ---- MLP state ----
  const [hidden, setHidden] = useState<number[]>([6, 4]);
  const [activation, setActivation] = useState<Activation>("tanh");
  const [learningRate, setLearningRate] = useState(0.05);
  const [dataset, setDataset] = useState<DatasetKind>("circle");
  const [playing, setPlaying] = useState(false);
  const [epochsPerSecond, setEpochsPerSecond] = useState(10);
  const [snap, setSnap] = useState<SnapshotMsg | null>(null);

  // ---- LLM state ----
  const [llmConfig, setLLMConfig] = useState<LLMConfig>({
    corpus: DEFAULT_CORPUS,
    contextSize: 3,
    hiddenSize: 24,
    learningRate: 0.1,
    temperature: 0.6,
  });
  const [textSnap, setTextSnap] = useState<TextSnapshot>({
    epoch: 0,
    loss: 0,
    paramCount: 0,
    vocabSize: uniqueChars(DEFAULT_CORPUS),
    contextSize: 3,
    hiddenSize: 24,
    sample: "",
    tokensPerSecond: 0,
    trainedSamples: 0,
  });
  const [llmPlaying, setLLMPlaying] = useState(false);
  const [llmEpochsPerSecond, setLLMEpochsPerSecond] = useState(20);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const correctionCount = useMemo(
    () => messages.filter((m) => m.corrected).length,
    [messages],
  );

  const [tab, setTab] = useState<TabKey>("brain");

  const layers = useMemo(() => [2, ...hidden, 1], [hidden]);
  const estimatedMLPParams = useMemo(
    () => estimateMLPParams(layers),
    [layers],
  );
  const estimatedLLMParams = useMemo(
    () =>
      estimateLLMParams(
        Math.max(2, uniqueChars(llmConfig.corpus)),
        llmConfig.contextSize,
        llmConfig.hiddenSize,
      ),
    [llmConfig.corpus, llmConfig.contextSize, llmConfig.hiddenSize],
  );

  // ---- MLP worker ----
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
    return () => worker.terminate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- LLM worker ----
  useEffect(() => {
    const worker = new Worker(
      new URL("./lib/text.worker.ts", import.meta.url),
      { type: "module" },
    );
    textWorkerRef.current = worker;
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg.type === "snapshot") {
        setTextSnap({
          epoch: msg.epoch,
          loss: msg.loss,
          paramCount: msg.paramCount,
          vocabSize: msg.vocabSize,
          contextSize: msg.contextSize,
          hiddenSize: msg.hiddenSize,
          sample: msg.sample,
          tokensPerSecond: msg.tokensPerSecond,
          trainedSamples: msg.trainedSamples,
        });
      } else if (msg.type === "generation") {
        const cb = pendingGenRef.current.get(msg.id);
        if (cb) {
          pendingGenRef.current.delete(msg.id);
          cb(msg.text);
        }
      } else if (msg.type === "exportModel") {
        const cb = pendingGenRef.current.get(msg.id);
        if (cb) {
          pendingGenRef.current.delete(msg.id);
          cb(JSON.stringify(msg.payload));
        }
      }
    };
    worker.postMessage({
      type: "init",
      opts: {
        corpus: llmConfig.corpus,
        contextSize: llmConfig.contextSize,
        hiddenSize: llmConfig.hiddenSize,
        learningRate: llmConfig.learningRate,
        temperature: llmConfig.temperature,
      },
    });
    return () => worker.terminate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push live config tweaks (lr/temperature) to the LLM worker without resetting weights.
  useEffect(() => {
    textWorkerRef.current?.postMessage({
      type: "config",
      partial: {
        learningRate: llmConfig.learningRate,
        temperature: llmConfig.temperature,
      },
    });
  }, [llmConfig.learningRate, llmConfig.temperature]);

  const reinitMLP = (
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

  const rebuildLLM = () => {
    setLLMPlaying(false);
    textWorkerRef.current?.postMessage({
      type: "reset",
      opts: {
        corpus: llmConfig.corpus,
        contextSize: llmConfig.contextSize,
        hiddenSize: llmConfig.hiddenSize,
        learningRate: llmConfig.learningRate,
        temperature: llmConfig.temperature,
      },
    });
    toast({ title: "Model rebuilt", description: "Weights reset to random." });
  };

  // ---- MLP handlers ----
  const handleAddLayer = () => {
    if (hidden.length >= 4) {
      toast({ title: "Max 4 hidden layers", description: "Keep it simple." });
      return;
    }
    const next = [...hidden, 4];
    if (estimateMLPParams([2, ...next, 1]) > MAX_PARAMS_MLP) {
      toast({
        title: "Parameter budget exceeded",
        description: `Network would exceed ${MAX_PARAMS_MLP} parameters.`,
      });
      return;
    }
    setHidden(next);
    reinitMLP(next);
  };

  const handleRemoveLayer = () => {
    if (hidden.length <= 1) return;
    const next = hidden.slice(0, -1);
    setHidden(next);
    reinitMLP(next);
  };

  const handleNeuronChange = (idx: number, delta: number) => {
    const next = hidden.slice();
    next[idx] = Math.max(1, Math.min(12, next[idx] + delta));
    if (estimateMLPParams([2, ...next, 1]) > MAX_PARAMS_MLP) {
      toast({
        title: "Parameter budget exceeded",
        description: `Cannot exceed ${MAX_PARAMS_MLP} parameters.`,
      });
      return;
    }
    setHidden(next);
    reinitMLP(next);
  };

  const handleActivation = (v: string) => {
    const a = v as Activation;
    setActivation(a);
    reinitMLP(hidden, a);
  };

  const handleDataset = (v: string) => {
    const d = v as DatasetKind;
    setDataset(d);
    reinitMLP(hidden, activation, d);
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

  const handleLLMSpeed = (vals: number[]) => {
    const v = vals[0];
    setLLMEpochsPerSecond(v);
    if (llmPlaying) {
      textWorkerRef.current?.postMessage({
        type: "play",
        epochsPerSecond: v,
      });
    }
  };

  const handlePlay = () => {
    if (mode === "mlp") {
      if (playing) {
        workerRef.current?.postMessage({ type: "pause" });
        setPlaying(false);
      } else {
        workerRef.current?.postMessage({ type: "play", epochsPerSecond });
        setPlaying(true);
      }
    } else {
      if (llmPlaying) {
        textWorkerRef.current?.postMessage({ type: "pause" });
        setLLMPlaying(false);
      } else {
        textWorkerRef.current?.postMessage({
          type: "play",
          epochsPerSecond: llmEpochsPerSecond,
        });
        setLLMPlaying(true);
      }
    }
  };

  const handleReset = () => {
    if (mode === "mlp") {
      reinitMLP();
    } else {
      rebuildLLM();
      setMessages([]);
    }
  };

  const generateFromWorker = useCallback(
    (seed: string): Promise<string> =>
      new Promise((resolve) => {
        const worker = textWorkerRef.current;
        if (!worker) {
          resolve("(model not ready)");
          return;
        }
        const id = `g-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        pendingGenRef.current.set(id, resolve);
        worker.postMessage({
          type: "generate",
          id,
          seed,
          length: 50,
          temperature: llmConfig.temperature,
        });
      }),
    [llmConfig.temperature],
  );

  const handleSave = async () => {
    if (mode === "mlp") {
      if (!snap) return;
      const payload = {
        kind: "mlp",
        architecture: { layers, activation },
        learningRate,
        epoch: snap.epoch,
        loss: snap.loss,
        accuracy: snap.accuracy,
        weights: snap.weights,
        biases: snap.biases,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `weights.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast({
        title: "weights.json saved",
        description: "Your trained model is ready to ship.",
      });
    } else {
      // Ask the worker to dump the trained char-LM weights.
      const worker = textWorkerRef.current;
      if (!worker) return;
      const id = `e-${Date.now()}`;
      const json = await new Promise<string>((resolve) => {
        pendingGenRef.current.set(id, resolve);
        worker.postMessage({ type: "exportModel", id });
      });
      if (!json || json === "null") {
        toast({
          title: "Nothing to export",
          description: "Train the model first.",
        });
        return;
      }
      const pretty = JSON.stringify(JSON.parse(json), null, 2);
      const blob = new Blob([pretty], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `char-lm.json`;
      a.click();
      URL.revokeObjectURL(url);
      toast({
        title: "char-lm.json saved",
        description: "Vocabulary, config, and weights exported.",
      });
    }
  };

  const handleModeChange = (m: AppMode) => {
    if (m === mode) return;
    setMode(m);
    setTab("brain");
    if (m === "llm" && playing) {
      workerRef.current?.postMessage({ type: "pause" });
      setPlaying(false);
    }
    if (m === "mlp" && llmPlaying) {
      textWorkerRef.current?.postMessage({ type: "pause" });
      setLLMPlaying(false);
    }
  };

  const overBudgetMLP = estimatedMLPParams > MAX_PARAMS_MLP;
  const llmHasModel = textSnap.epoch > 0;
  const isPlaying = mode === "mlp" ? playing : llmPlaying;
  const playLabel = mode === "mlp" ? "Train" : "Train";

  const llmModelLabel = `Char-LM · ${llmConfig.contextSize}→${llmConfig.hiddenSize}→${textSnap.vocabSize}`;

  // ===== MLP VIEWS =====
  const MLPArchitect = (
    <div className="space-y-4">
      <Card className="p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="size-4 text-sky-400" />
            <span className="text-sm font-semibold text-slate-100">
              Architecture
            </span>
          </div>
          <span
            className={`text-[11px] tabular-nums ${
              overBudgetMLP ? "text-red-400" : "text-slate-400"
            }`}
          >
            {estimatedMLPParams} / {MAX_PARAMS_MLP} params
          </span>
        </div>

        <div className="space-y-2">
          {hidden.map((n, i) => (
            <div
              key={i}
              className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-900/60 px-3 min-h-[52px]"
            >
              <span className="text-xs text-slate-400">
                Hidden Layer {i + 1}
              </span>
              <div className="flex items-center gap-1">
                <Button
                  size="icon"
                  variant="ghost"
                  className="size-9 rounded-lg"
                  onClick={() => handleNeuronChange(i, -1)}
                >
                  <Minus className="size-4" />
                </Button>
                <span className="text-sm font-semibold w-7 text-center tabular-nums text-slate-100">
                  {n}
                </span>
                <Button
                  size="icon"
                  variant="ghost"
                  className="size-9 rounded-lg"
                  onClick={() => handleNeuronChange(i, +1)}
                >
                  <Plus className="size-4" />
                </Button>
              </div>
            </div>
          ))}
        </div>

        <div className="flex gap-2">
          <Button
            variant="secondary"
            onClick={handleAddLayer}
            className="flex-1 gap-1.5 min-h-[44px] rounded-xl"
          >
            <Plus className="size-4" /> Add Layer
          </Button>
          <Button
            variant="secondary"
            onClick={handleRemoveLayer}
            className="flex-1 gap-1.5 min-h-[44px] rounded-xl"
            disabled={hidden.length <= 1}
          >
            <Minus className="size-4" /> Remove
          </Button>
        </div>
      </Card>

      <Card className="p-5 space-y-5">
        <div className="flex items-center gap-2">
          <Zap className="size-4 text-amber-400" />
          <span className="text-sm font-semibold text-slate-100">
            Hyperparameters
          </span>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Learning Rate</span>
            <span className="text-xs tabular-nums text-slate-200">
              {learningRate.toFixed(3)}
            </span>
          </div>
          <Slider
            min={0.001}
            max={0.5}
            step={0.001}
            value={[learningRate]}
            onValueChange={handleLR}
            className="py-2"
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">
              Train Speed (epochs/sec)
            </span>
            <span className="text-xs tabular-nums text-slate-200">
              {epochsPerSecond}
            </span>
          </div>
          <Slider
            min={1}
            max={120}
            step={1}
            value={[epochsPerSecond]}
            onValueChange={handleSpeed}
            className="py-2"
          />
        </div>

        <div className="space-y-2">
          <span className="text-xs text-slate-400">Activation Function</span>
          <Select value={activation} onValueChange={handleActivation}>
            <SelectTrigger className="min-h-[44px] rounded-xl">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="tanh">Tanh</SelectItem>
              <SelectItem value="relu">ReLU</SelectItem>
              <SelectItem value="sigmoid">Sigmoid</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <span className="text-xs text-slate-400">Dataset</span>
          <Select value={dataset} onValueChange={handleDataset}>
            <SelectTrigger className="min-h-[44px] rounded-xl">
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
    </div>
  );

  const MLPBrain = (
    <Card className="p-4 sm:p-5">
      <div className="flex items-center justify-between mb-3 gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-slate-100">
            Network Topology
          </div>
          <div className="text-[11px] text-slate-400 truncate">
            Blue = positive · Red = negative · Thickness = magnitude
          </div>
        </div>
        <div className="text-[11px] text-slate-400 tabular-nums whitespace-nowrap">
          {layers.join(" → ")} · {snap?.paramCount ?? estimatedMLPParams}p
        </div>
      </div>
      <div className="rounded-xl bg-slate-950/60 border border-slate-800 overflow-hidden">
        <NetworkCanvas
          layers={snap?.layers ?? layers}
          weights={snap?.weights ?? []}
          height={420}
        />
      </div>
    </Card>
  );

  const MLPOutput = (
    <div className="space-y-4">
      <Card className="p-5">
        <div className="flex items-center justify-between mb-3 gap-3">
          <div className="min-w-0">
            <div className="text-sm font-semibold text-slate-100">
              2D Point Classifier
            </div>
            <div className="text-[11px] text-slate-400 truncate">
              The network learns to color the plane based on (x, y).
            </div>
          </div>
          <div className="hidden sm:flex items-center gap-3 text-[11px] text-slate-400">
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-full bg-sky-400" /> Class A
            </span>
            <span className="flex items-center gap-1.5">
              <span className="size-2.5 rounded-full bg-emerald-400" /> Class B
            </span>
          </div>
        </div>
        <div className="flex justify-center">
          <div className="w-full max-w-[420px] aspect-square">
            <ResponsiveDataView snap={snap} />
          </div>
        </div>
      </Card>

      <Card className="p-5">
        <div className="flex items-center gap-2 mb-3">
          <Activity className="size-4 text-emerald-400" />
          <span className="text-sm font-semibold text-slate-100">
            Training Stats
          </span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <Stat label="Epoch" value={snap?.epoch ?? 0} />
          <Stat label="Loss" value={snap ? snap.loss.toFixed(3) : "—"} />
          <Stat
            label="Acc"
            value={snap ? `${(snap.accuracy * 100).toFixed(0)}%` : "—"}
          />
        </div>
      </Card>

      <SharingHub mode="mlp" onDownload={handleSave} hasModel={!!snap} />
    </div>
  );

  // ===== LLM VIEWS =====
  const LLMArchitectView = (
    <div className="space-y-4">
      <LLMArchitect
        config={llmConfig}
        onChange={setLLMConfig}
        onApply={rebuildLLM}
        paramCount={estimatedLLMParams}
        vocabSize={Math.max(2, uniqueChars(llmConfig.corpus))}
        maxParams={MAX_PARAMS_LLM}
      />
      <Card className="p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Zap className="size-4 text-amber-400" />
          <span className="text-sm font-semibold text-slate-100">
            Train Speed
          </span>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Epochs / sec</span>
            <span className="text-xs tabular-nums text-slate-200">
              {llmEpochsPerSecond}
            </span>
          </div>
          <Slider
            min={1}
            max={60}
            step={1}
            value={[llmEpochsPerSecond]}
            onValueChange={handleLLMSpeed}
            className="py-2"
          />
        </div>
      </Card>
    </div>
  );

  const LLMBrain = (
    <ChatView
      modelLabel={llmModelLabel}
      messages={messages}
      setMessages={setMessages}
      loading={chatLoading}
      setLoading={setChatLoading}
      onCorrection={() =>
        toast({
          title: "Correction recorded",
          description: "Added to your fine-tune set.",
        })
      }
      generate={generateFromWorker}
      liveSample={textSnap.sample}
      epoch={textSnap.epoch}
      loss={textSnap.loss}
      isTraining={llmPlaying}
    />
  );

  const LLMOutput = (
    <div className="space-y-4">
      <LLMStats
        modelLabel={llmModelLabel}
        epoch={textSnap.epoch}
        loss={textSnap.loss}
        paramCount={textSnap.paramCount}
        vocabSize={textSnap.vocabSize}
        contextSize={textSnap.contextSize}
        hiddenSize={textSnap.hiddenSize}
        tokensPerSecond={textSnap.tokensPerSecond}
        trainedSamples={textSnap.trainedSamples}
        messageCount={messages.length}
        correctionCount={correctionCount}
        liveSample={textSnap.sample}
      />
      <SharingHub mode="llm" onDownload={handleSave} hasModel={llmHasModel} />
    </div>
  );

  const ArchitectContent = mode === "mlp" ? MLPArchitect : LLMArchitectView;
  const BrainContent = mode === "mlp" ? MLPBrain : LLMBrain;
  const OutputContent = mode === "mlp" ? MLPOutput : LLMOutput;

  const tabs: { key: TabKey; label: string; icon: typeof Brain }[] = [
    { key: "architect", label: "Architect", icon: SlidersHorizontal },
    {
      key: "brain",
      label: mode === "mlp" ? "Brain" : "Chat",
      icon: mode === "mlp" ? Network : MessageSquare,
    },
    { key: "output", label: "Output", icon: LineChart },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900 text-slate-100">
      <header className="sticky top-0 z-20 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
        <div className="px-3 sm:px-6 h-14 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="size-8 rounded-lg bg-gradient-to-br from-sky-500 to-emerald-500 flex items-center justify-center shrink-0">
              <Brain className="size-4 text-slate-900" />
            </div>
            <div className="min-w-0 hidden sm:block">
              <div className="text-sm font-semibold tracking-tight truncate">
                AI Sandbox
              </div>
              <div className="text-[11px] text-slate-400 hidden sm:block truncate">
                Build, train, and ship tiny models — entirely in the browser.
              </div>
            </div>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <ModeToggle mode={mode} onChange={handleModeChange} />
            <div className="w-px h-6 bg-slate-800 mx-0.5 hidden sm:block" />
            <Button
              size="icon"
              variant="secondary"
              onClick={handleReset}
              aria-label="Reset"
              className="sm:hidden size-10 rounded-xl"
            >
              <RefreshCcw className="size-4" />
            </Button>
            <Button
              size="icon"
              variant="secondary"
              onClick={handleSave}
              aria-label="Save"
              className="sm:hidden size-10 rounded-xl"
            >
              <Save className="size-4" />
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={handleReset}
              className="hidden sm:inline-flex gap-1.5 min-h-[40px] rounded-xl"
            >
              <RefreshCcw className="size-3.5" />
              Reset
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={handleSave}
              className="hidden sm:inline-flex gap-1.5 min-h-[40px] rounded-xl"
            >
              <Save className="size-3.5" />
              Save
            </Button>
            <Button
              onClick={handlePlay}
              className="gap-1.5 min-w-[88px] sm:min-w-[96px] min-h-[40px] rounded-xl"
            >
              {isPlaying ? (
                <>
                  <Pause className="size-4" />
                  Pause
                </>
              ) : (
                <>
                  <Play className="size-4" />
                  {playLabel}
                </>
              )}
            </Button>
          </div>
        </div>
      </header>

      <main className="md:hidden px-3 pt-4 pb-28">
        {tab === "architect" && ArchitectContent}
        {tab === "brain" && BrainContent}
        {tab === "output" && OutputContent}
      </main>

      <main className="hidden md:grid grid-cols-12 gap-4 p-4 lg:p-6">
        <aside className="col-span-4 lg:col-span-3">{ArchitectContent}</aside>
        <section className="col-span-8 lg:col-span-9 space-y-4">
          {BrainContent}
          {mode === "mlp" ? (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <div className="space-y-4">
                <Card className="p-5">
                  <div className="flex items-center justify-between mb-3 gap-3">
                    <div className="min-w-0">
                      <div className="text-sm font-semibold text-slate-100">
                        2D Point Classifier
                      </div>
                      <div className="text-[11px] text-slate-400 truncate">
                        The network learns to color the plane based on (x, y).
                      </div>
                    </div>
                    <div className="flex items-center gap-3 text-[11px] text-slate-400">
                      <span className="flex items-center gap-1.5">
                        <span className="size-2.5 rounded-full bg-sky-400" /> A
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="size-2.5 rounded-full bg-emerald-400" />{" "}
                        B
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-center">
                    <div className="w-full max-w-[380px] aspect-square">
                      <ResponsiveDataView snap={snap} />
                    </div>
                  </div>
                </Card>
                <Card className="p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Activity className="size-4 text-emerald-400" />
                    <span className="text-sm font-semibold text-slate-100">
                      Training Stats
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    <Stat label="Epoch" value={snap?.epoch ?? 0} />
                    <Stat
                      label="Loss"
                      value={snap ? snap.loss.toFixed(3) : "—"}
                    />
                    <Stat
                      label="Acc"
                      value={
                        snap ? `${(snap.accuracy * 100).toFixed(0)}%` : "—"
                      }
                    />
                  </div>
                </Card>
              </div>
              <SharingHub
                mode="mlp"
                onDownload={handleSave}
                hasModel={!!snap}
              />
            </div>
          ) : (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <LLMStats
                modelLabel={llmModelLabel}
                epoch={textSnap.epoch}
                loss={textSnap.loss}
                paramCount={textSnap.paramCount}
                vocabSize={textSnap.vocabSize}
                contextSize={textSnap.contextSize}
                hiddenSize={textSnap.hiddenSize}
                tokensPerSecond={textSnap.tokensPerSecond}
                trainedSamples={textSnap.trainedSamples}
                messageCount={messages.length}
                correctionCount={correctionCount}
                liveSample={textSnap.sample}
              />
              <SharingHub
                mode="llm"
                onDownload={handleSave}
                hasModel={llmHasModel}
              />
            </div>
          )}
        </section>
      </main>

      <nav className="md:hidden fixed bottom-0 inset-x-0 z-20 border-t border-slate-800 bg-slate-950/90 backdrop-blur-lg">
        <div className="grid grid-cols-3">
          {tabs.map((t) => {
            const Icon = t.icon;
            const active = tab === t.key;
            return (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`flex flex-col items-center justify-center gap-1 py-2.5 min-h-[60px] transition-colors ${
                  active
                    ? "text-sky-400"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Icon
                  className={`size-5 ${active ? "drop-shadow-[0_0_8px_rgba(56,189,248,0.6)]" : ""}`}
                />
                <span className="text-[10px] font-medium tracking-wide uppercase">
                  {t.label}
                </span>
              </button>
            );
          })}
        </div>
        <div
          className="absolute top-0 h-0.5 bg-sky-400 transition-all duration-300"
          style={{
            width: "33.333%",
            left:
              tab === "architect"
                ? "0%"
                : tab === "brain"
                  ? "33.333%"
                  : "66.666%",
          }}
        />
      </nav>

      <Toaster />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 px-2 py-3 text-center">
      <div className="text-[10px] uppercase tracking-wider text-slate-400">
        {label}
      </div>
      <div className="text-sm font-semibold tabular-nums text-slate-100 mt-0.5">
        {value}
      </div>
    </div>
  );
}

function ResponsiveDataView({ snap }: { snap: SnapshotMsg | null }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState(320);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      for (const e of entries) {
        const w = e.contentRect.width;
        if (w > 0) setSize(Math.floor(w));
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return (
    <div ref={ref} className="w-full">
      <DataView
        data={snap?.data ?? []}
        grid={snap?.grid ?? null}
        gridRes={snap?.gridRes ?? 28}
        size={size}
      />
    </div>
  );
}
