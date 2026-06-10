import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Play,
  Pause,
  RefreshCcw,
  Save,
  Brain,
  Zap,
  SlidersHorizontal,
  MessageSquare,
  Plus,
  Trash2,
  ArrowLeft,
  Cpu,
  Database,
  Activity,
  ScrollText,
} from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Toaster } from "@/components/ui/toaster";
import { useToast } from "@/hooks/use-toast";
import { SharingHub } from "@/components/SharingHub";
import {
  LLMArchitect,
  type LLMConfig,
  type Dataset,
} from "@/components/llm/LLMArchitect";
import { ChatView, type ChatMessage } from "@/components/llm/ChatView";
import { LLMStats } from "@/components/llm/LLMStats";
import { SaveModal } from "@/components/SaveModal";
import { DeployHub } from "@/components/llm/DeployHub";
import {
  getModels,
  saveModel,
  deleteModel,
  makeId,
  type SavedModel,
  type CharLMWeights,
} from "@/lib/storage";
import { tokenize, type Tokenization } from "@/lib/textnet";

type TabKey = "chat" | "train" | "deploy";
type TrainStep = "fleet" | "setup" | "training";
type TrainTab = "arch" | "dataset" | "explorer" | "status";

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

const MAX_PARAMS_LLM = 5_000_000;

const DEFAULT_CORPUS = [
  "User: hello",
  "Bot: hi there how can i help you",
  "User: how are you",
  "Bot: i am a small neural network and i am doing great",
  "User: what is your name",
  "Bot: i am a tiny language model trained in your browser",
  "User: tell me a joke",
  "Bot: why did the neuron cross the layer to get to the other bias",
  "User: goodbye",
  "Bot: goodbye have a nice day",
  "User: thanks",
  "Bot: you are welcome",
].join("\n") + "\n";

function estimateLLMParams(vocab: number, ctx: number, hidden: number): number {
  return vocab * ctx * hidden + hidden + hidden * vocab + vocab;
}

function normalizePromptForLLM(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const OOV_STOP_WORDS = new Set(["tell", "me", "about", "what", "is", "a", "the"]);
const OOV_FALLBACK_REPLY = "I don't know the answer to that. I haven't been trained on this topic yet.";

function levenshteinDistance(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = new Array<number>(n + 1);
  let curr = new Array<number>(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    const ac = a.charCodeAt(i - 1);
    for (let j = 1; j <= n; j++) {
      const cost = ac === b.charCodeAt(j - 1) ? 0 : 1;
      curr[j] = Math.min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n];
}

function uniqueChars(s: string): number {
  return new Set(s).size;
}

function vocabSizeFor(corpus: string, mode: Tokenization): number {
  const normalized = normalizePromptForLLM(corpus);
  if (mode === "char") return uniqueChars(normalized);
  return new Set(tokenize(normalized, "word")).size;
}

function slug(s: string): string {
  return (
    s
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 40) || "model"
  );
}

function defaultSaveName(modeLabel: string, epoch: number, tokenization?: "char" | "word"): string {
  const tag = tokenization === "word" ? "Word-LM" : "Char-LM";
  return `${tag} · ${modeLabel} · ep ${epoch}`;
}

export default function App() {
  const { toast } = useToast();
  const textWorkerRef = useRef<Worker | null>(null);
  const pendingGenRef = useRef<Map<string, (text: string) => void>>(new Map());

  // ── LLM state ──────────────────────────────────────────────────────────────
  const [datasets, setDatasets] = useState<Dataset[]>([
    { id: 1, name: "Base Training", text: DEFAULT_CORPUS, active: true },
  ]);
  const [llmConfig, setLLMConfig] = useState<LLMConfig>({
    corpus: DEFAULT_CORPUS,
    contextSize: 3,
    hiddenSize: 24,
    learningRate: 0.1,
    temperature: 0.6,
    tokenization: "char",
    topK: 5,
    systemPrompt: "",
  });
  const [textSnap, setTextSnap] = useState<TextSnapshot>({
    epoch: 0,
    loss: 0,
    paramCount: 0,
    vocabSize: vocabSizeFor(DEFAULT_CORPUS, "char"),
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

  // ── Navigation state ───────────────────────────────────────────────────────
  const [tab, setTab] = useState<TabKey>("train");
  const [trainStep, setTrainStep] = useState<TrainStep>("fleet");
  const [trainTab, setTrainTab] = useState<TrainTab>("status");
  const [newModelName, setNewModelName] = useState("");
  const [saveOpen, setSaveOpen] = useState(false);
  const [libraryRefresh, setLibraryRefresh] = useState(0);
  const [savedModels, setSavedModels] = useState<SavedModel[]>([]);

  useEffect(() => {
    getModels().then(setSavedModels).catch(() => {});
  }, [libraryRefresh]);

  // ── LLM worker ─────────────────────────────────────────────────────────────
  useEffect(() => {
    const worker = new Worker(new URL("./lib/text.worker.ts", import.meta.url), { type: "module" });
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
        if (cb) { pendingGenRef.current.delete(msg.id); cb(msg.text); }
      } else if (msg.type === "exportModel") {
        const cb = pendingGenRef.current.get(msg.id);
        if (cb) { pendingGenRef.current.delete(msg.id); cb(JSON.stringify(msg.payload)); }
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
        tokenization: llmConfig.tokenization,
        topK: llmConfig.topK,
      },
    });
    return () => worker.terminate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    textWorkerRef.current?.postMessage({
      type: "config",
      partial: { learningRate: llmConfig.learningRate, temperature: llmConfig.temperature, topK: llmConfig.topK },
    });
  }, [llmConfig.learningRate, llmConfig.temperature, llmConfig.topK]);

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
        tokenization: llmConfig.tokenization,
        topK: llmConfig.topK,
      },
    });
    toast({ title: "Model rebuilt", description: "Weights reset to random." });
  };

  const handleLLMSpeed = (vals: number[]) => {
    const v = vals[0];
    setLLMEpochsPerSecond(v);
    if (llmPlaying) textWorkerRef.current?.postMessage({ type: "play", epochsPerSecond: v });
  };

  const handlePlay = () => {
    if (llmPlaying) {
      textWorkerRef.current?.postMessage({ type: "pause" });
      setLLMPlaying(false);
    } else {
      textWorkerRef.current?.postMessage({ type: "play", epochsPerSecond: llmEpochsPerSecond });
      setLLMPlaying(true);
    }
  };

  const handleReset = () => { rebuildLLM(); setMessages([]); };

  const generateFromWorker = useCallback(
    (seed: string): Promise<string> =>
      new Promise((resolve) => {
        const worker = textWorkerRef.current;
        if (!worker) { resolve("(model not ready)"); return; }

        let normalizedInput = normalizePromptForLLM(seed);

        if (llmConfig.tokenization === "word") {
          const corpusVocab = new Set(tokenize(normalizePromptForLLM(llmConfig.corpus), "word"));
          const inputWords = normalizedInput.split(" ").filter(Boolean);
          const AUTOCORRECT_MAX_DIST = 1;
          const correctedWords = inputWords.map((w) => {
            if (corpusVocab.has(w)) return w;
            let best = w; let bestDist = Infinity;
            for (const v of corpusVocab) { const d = levenshteinDistance(w, v); if (d < bestDist) { bestDist = d; best = v; } }
            return bestDist <= AUTOCORRECT_MAX_DIST ? best : w;
          });
          const meaningful = correctedWords.filter((w) => !OOV_STOP_WORDS.has(w));
          if (meaningful.length > 0 && meaningful.every((w) => !corpusVocab.has(w))) { resolve(OOV_FALLBACK_REPLY); return; }
          let routedPrompt: string | null = null;
          if (meaningful.length > 0) {
            const meaningfulSet = new Set(meaningful);
            const knownPrompts = llmConfig.corpus.split("\n")
              .filter((l) => /^\s*User:\s/i.test(l))
              .map((l) => l.replace(/^\s*User:\s*/i, "").replace(/\bBot:\s.*$/i, "").trim());
            let bestScore = 0;
            for (const kp of knownPrompts) {
              const kpCore = new Set(normalizePromptForLLM(kp).split(" ").filter((w) => w && !OOV_STOP_WORDS.has(w)));
              let overlap = 0;
              for (const w of kpCore) { if (meaningfulSet.has(w)) overlap++; }
              if (overlap > bestScore) { bestScore = overlap; routedPrompt = normalizePromptForLLM(kp); }
            }
          }
          normalizedInput = routedPrompt ?? correctedWords.join(" ");
        }

        const id = `g-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        pendingGenRef.current.set(id, resolve);
        const systemPart = llmConfig.systemPrompt ? normalizePromptForLLM(llmConfig.systemPrompt) + " " : "";
        const formattedSeed = `${systemPart}user ${normalizedInput} bot `;
        worker.postMessage({ type: "generate", id, seed: formattedSeed, length: 300, temperature: llmConfig.temperature });
      }),
    [llmConfig.temperature, llmConfig.systemPrompt, llmConfig.corpus, llmConfig.tokenization],
  );

  const requestLLMExport = useCallback((): Promise<CharLMWeights | null> => {
    return new Promise((resolve) => {
      const worker = textWorkerRef.current;
      if (!worker) return resolve(null);
      const id = `e-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      pendingGenRef.current.set(id, (json: string) => {
        if (!json || json === "null") return resolve(null);
        try {
          const parsed = JSON.parse(json);
          parsed.corpus = llmConfig.corpus;
          parsed.temperature = llmConfig.temperature;
          parsed.tokenization = llmConfig.tokenization;
          parsed.datasets = datasets.map(({ name, text, active }) => ({ name, text, active }));
          resolve(parsed as CharLMWeights & { corpus: string });
        } catch { resolve(null); }
      });
      worker.postMessage({ type: "exportModel", id });
    });
  }, [llmConfig.corpus, llmConfig.temperature, llmConfig.tokenization, datasets]);

  const downloadJSON = (filename: string, payload: unknown) => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  };

  const handleOpenSave = () => {
    if (textSnap.epoch === 0) { toast({ title: "Train first", description: "Run a few epochs before saving." }); return; }
    setSaveOpen(true);
  };

  const handleExportFile = async (name: string) => {
    const w = await requestLLMExport();
    if (!w) { toast({ title: "Nothing to export", description: "Train first." }); return; }
    downloadJSON(`${slug(name)}.json`, { kind: "char-lm", name, ...w });
    toast({ title: "File exported", description: `${name}.json saved.` });
  };

  const handleSaveToLibrary = async (name: string) => {
    const w = await requestLLMExport();
    if (!w) { toast({ title: "Nothing to save", description: "Train first." }); return; }
    const model: SavedModel = {
      id: makeId(), name, type: "Char-LM", date: Date.now(),
      paramsCount: textSnap.paramCount, loss: textSnap.loss, epoch: textSnap.epoch, weights: w,
    };
    await saveModel(model);
    setLibraryRefresh((v) => v + 1);
    toast({ title: "Saved to Library", description: `${name} is in your model library.` });
  };

  // ── The Persistence Handle ─────────────────────────────────────────────────
  // Saves the newly configured model to the library BEFORE navigating to the
  // training dashboard, so the model is always persisted from the moment it
  // is created — not only after the user manually hits Save.
  const saveNewModel = async () => {
    const name = newModelName.trim() || defaultSaveName(
      llmConfig.tokenization === "word" ? "Word-LM" : "Char-LM",
      0,
      llmConfig.tokenization,
    );

    // 1. Reset the worker with the new config (queued first)
    setLLMPlaying(false);
    textWorkerRef.current?.postMessage({
      type: "reset",
      opts: {
        corpus: llmConfig.corpus,
        contextSize: llmConfig.contextSize,
        hiddenSize: llmConfig.hiddenSize,
        learningRate: llmConfig.learningRate,
        temperature: llmConfig.temperature,
        tokenization: llmConfig.tokenization,
        topK: llmConfig.topK,
      },
    });

    // 2. Export immediately after — worker processes messages in order,
    //    so the reset will have completed before exportModel fires.
    const w = await requestLLMExport();
    if (!w) {
      toast({ title: "Initialization failed", description: "Could not initialize model weights." });
      return;
    }

    // 3. Persist to IndexedDB
    const model: SavedModel = {
      id: makeId(), name, type: "Char-LM", date: Date.now(),
      paramsCount: estimatedLLMParams, loss: 0, epoch: 0, weights: w,
    };
    await saveModel(model);
    setLibraryRefresh((v) => v + 1);

    // 4. Navigate to Training Dashboard
    setMessages([]);
    setNewModelName("");
    setTrainTab("status");
    setTrainStep("training");
    toast({ title: "Model created", description: `"${name}" saved to your library.` });
  };

  const handleLoadModel = (model: SavedModel) => {
    if (llmPlaying) { textWorkerRef.current?.postMessage({ type: "pause" }); setLLMPlaying(false); }
    const w = model.weights as CharLMWeights;
    const restoredTok: Tokenization =
      w.tokenization === "word" || w.tokenization === "char"
        ? w.tokenization
        : (w.vocab.some((t) => typeof t === "string" && t.length > 1) ? "word" : "char");
    setLLMConfig((prev) => ({
      ...prev,
      corpus: w.corpus ?? w.vocab.join(restoredTok === "word" ? " " : ""),
      contextSize: w.config.contextSize,
      hiddenSize: w.config.hiddenSize,
      learningRate: w.config.learningRate,
      temperature: typeof w.temperature === "number" ? w.temperature : 0.6,
      tokenization: restoredTok,
    }));
    if (Array.isArray(w.datasets) && w.datasets.length > 0) {
      const baseId = Date.now();
      setDatasets(
        w.datasets
          .filter((d): d is { name: string; text: string; active: boolean } =>
            !!d && typeof d === "object" &&
            typeof (d as { name?: unknown }).name === "string" &&
            typeof (d as { text?: unknown }).text === "string" &&
            typeof (d as { active?: unknown }).active === "boolean",
          )
          .map((d, i) => ({ id: baseId + i, name: d.name, text: d.text, active: d.active })),
      );
    }
    textWorkerRef.current?.postMessage({ type: "loadWeights", payload: { ...w, epoch: model.epoch, loss: model.loss } });
    setMessages([]);
    toast({ title: "Model loaded", description: `${model.name} is now active.` });
  };

  const handleDeleteModel = async (id: string) => {
    await deleteModel(id);
    setLibraryRefresh((v) => v + 1);
    toast({ title: "Deleted", description: "Model removed from library." });
  };

  const handleImportModel = (json: string) => {
    try {
      const parsed = JSON.parse(json) as Record<string, unknown>;
      if (!parsed || typeof parsed !== "object") { toast({ title: "Import failed", description: "Invalid file." }); return; }
      const kind = parsed.kind as string | undefined;
      if (kind === "char-lm") {
        const model: SavedModel = {
          id: makeId(), name: (parsed.name as string | undefined) ?? "Imported Model",
          type: "Char-LM", date: Date.now(), paramsCount: 0,
          loss: (parsed.loss as number | undefined) ?? 0,
          epoch: (parsed.epoch as number | undefined) ?? 0,
          weights: parsed as unknown as CharLMWeights,
        };
        handleLoadModel(model);
      } else {
        toast({ title: "Import failed", description: "Only Char-LM models are supported." });
      }
    } catch { toast({ title: "Import failed", description: "Could not parse the JSON file." }); }
  };

  // ── Derived state ──────────────────────────────────────────────────────────
  const estimatedLLMParams = useMemo(
    () => estimateLLMParams(
      Math.max(2, vocabSizeFor(llmConfig.corpus, llmConfig.tokenization)),
      llmConfig.contextSize,
      llmConfig.hiddenSize,
    ),
    [llmConfig.corpus, llmConfig.contextSize, llmConfig.hiddenSize, llmConfig.tokenization],
  );

  const llmHasModel = textSnap.epoch > 0;
  const llmModeTag = llmConfig.tokenization === "word" ? "Word-LM" : "Char-LM";
  const llmModelLabel = `${llmModeTag} · ${llmConfig.contextSize}→${llmConfig.hiddenSize}→${textSnap.vocabSize}`;
  const llmModels = savedModels.filter((m) => m.type !== "MLP");

  const viewTitle = tab === "chat" ? "Chat Studio" : tab === "train" ? "Training Lab" : "Deploy Hub";

  return (
    <div className="h-dvh bg-[#000000] text-slate-100 flex overflow-hidden">

      {/* ═══════════════════════════════════════════════════════════════════
          ULTRA-SLIM SIDEBAR
      ══════════════════════════════════════════════════════════════════════ */}
      <aside className="hidden md:flex flex-col w-14 shrink-0 border-r border-white/[0.05] bg-[#000000] items-center pt-4 pb-3">

        {/* Logo mark */}
        <div className="size-9 mb-5 rounded-xl bg-[#0A84FF]/12 border border-[#0A84FF]/18 flex items-center justify-center shrink-0">
          <Brain className="size-[18px] text-[#0A84FF]" />
        </div>

        {/* Primary nav */}
        <div className="flex flex-col gap-0.5 w-full px-2">
          <SlimNavItem icon={MessageSquare} label="Chat Studio" active={tab === "chat"} onClick={() => setTab("chat")} />
          <SlimNavItem icon={SlidersHorizontal} label="Training Lab" active={tab === "train"} onClick={() => setTab("train")} />
          <SlimNavItem icon={Zap} label="Deploy Hub" active={tab === "deploy"} onClick={() => setTab("deploy")} />
        </div>

        <div className="flex-1" />

        {/* Bottom actions */}
        <div className="flex flex-col gap-1.5 w-full px-2">
          <button
            onClick={handlePlay}
            title={llmPlaying ? "Pause training" : "Start training"}
            className={`w-full h-10 rounded-xl flex items-center justify-center transition-all ${
              llmPlaying
                ? "bg-[#0A84FF]/12 text-[#0A84FF] border border-[#0A84FF]/22 hover:bg-[#0A84FF]/22"
                : "bg-[#0A84FF] text-white hover:bg-[#409CFF]"
            }`}
          >
            {llmPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
          </button>
          <button onClick={handleReset} title="Reset model" className="w-full h-9 rounded-xl flex items-center justify-center text-white/25 hover:text-white/65 hover:bg-white/[0.05] border border-white/[0.05] transition-all">
            <RefreshCcw className="size-3.5" />
          </button>
          <button onClick={handleOpenSave} title="Save model" className="w-full h-9 rounded-xl flex items-center justify-center text-white/25 hover:text-white/65 hover:bg-white/[0.05] border border-white/[0.05] transition-all">
            <Save className="size-3.5" />
          </button>
        </div>
      </aside>

      {/* ═══════════════════════════════════════════════════════════════════
          MAIN CONTENT AREA
      ══════════════════════════════════════════════════════════════════════ */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Top bar */}
        <header className="h-14 shrink-0 border-b border-white/[0.05] bg-[#000000]/90 backdrop-blur-xl flex items-center justify-between px-4 md:px-5 gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="md:hidden size-7 rounded-lg bg-[#0A84FF]/12 border border-[#0A84FF]/18 flex items-center justify-center shrink-0">
              <Brain className="size-3.5 text-[#0A84FF]" />
            </div>
            <div className="min-w-0">
              <div className="text-[13px] font-semibold text-white tracking-tight truncate">{viewTitle}</div>
              <div className="text-[10px] text-white/28 tabular-nums truncate font-mono">{llmModelLabel}</div>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {llmPlaying && (
              <div className="hidden md:flex items-center gap-3 mr-1">
                <div className="flex items-center gap-1.5">
                  <span className="relative flex size-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#0A84FF] opacity-70" />
                    <span className="relative inline-flex rounded-full size-2 bg-[#0A84FF]" />
                  </span>
                  <span className="text-[11px] font-medium text-[#0A84FF]">Training</span>
                </div>
                <div className="text-[11px] tabular-nums text-white/45">
                  Loss <span className="text-white/75">{textSnap.loss > 0 ? textSnap.loss.toFixed(4) : "—"}</span>
                </div>
                <div className="text-[11px] tabular-nums text-white/45">
                  Ep <span className="text-white/75">{textSnap.epoch > 0 ? textSnap.epoch : "—"}</span>
                </div>
                <div className="text-[11px] tabular-nums text-white/45">
                  Tokens <span className="text-white/75">{textSnap.trainedSamples > 0 ? textSnap.trainedSamples > 1000 ? `${(textSnap.trainedSamples / 1000).toFixed(1)}k` : String(textSnap.trainedSamples) : "—"}</span>
                </div>
              </div>
            )}
            {/* Mobile controls */}
            <div className="md:hidden flex items-center gap-1.5">
              <button
                onClick={handlePlay}
                className={`h-9 px-3 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                  llmPlaying ? "bg-[#0A84FF]/12 text-[#0A84FF] border border-[#0A84FF]/22" : "bg-[#0A84FF] text-white"
                }`}
              >
                {llmPlaying ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
                {llmPlaying ? "Pause" : "Train"}
              </button>
              <button onClick={handleReset} className="size-9 rounded-lg border border-white/[0.06] text-white/35 hover:text-white/65 hover:bg-white/[0.05] flex items-center justify-center transition-colors">
                <RefreshCcw className="size-3.5" />
              </button>
              <button onClick={handleOpenSave} className="size-9 rounded-lg border border-white/[0.06] text-white/35 hover:text-white/65 hover:bg-white/[0.05] flex items-center justify-center transition-colors">
                <Save className="size-3.5" />
              </button>
            </div>
          </div>
        </header>

        {/* ── View router ─────────────────────────────────────────────────── */}
        <div className="flex-1 overflow-hidden">

          {/* ────────────────────────────────────────────────────────────────
              TRAINING LAB
          ──────────────────────────────────────────────────────────────── */}
          {tab === "train" && (
            <div className="h-full overflow-hidden">

              {/* ─── Fleet Dashboard ───────────────────────────────────────── */}
              {trainStep === "fleet" && (
                <div className="h-full overflow-y-auto">
                  <div className="max-w-3xl mx-auto py-8 px-5">
                    <div className="mb-7">
                      <h2 className="text-[17px] font-semibold text-white/88 tracking-tight">Models</h2>
                      <p className="text-[12px] text-white/28 mt-0.5 leading-relaxed">
                        Select a model to continue training, or create a new base model.
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {llmModels.map((m) => (
                        <div
                          key={m.id}
                          className="group relative rounded-2xl border border-white/[0.06] bg-[#0d0d0d] hover:bg-[#111111] hover:border-white/[0.10] transition-all cursor-pointer overflow-hidden"
                          onClick={() => { handleLoadModel(m); setTrainStep("training"); setTrainTab("status"); }}
                        >
                          <div className="p-4">
                            <div className="flex items-start justify-between mb-3">
                              <div className="size-9 rounded-xl bg-[#0A84FF]/10 border border-[#0A84FF]/15 flex items-center justify-center">
                                <Brain className="size-4 text-[#0A84FF]/65" />
                              </div>
                              <span className="text-[9px] font-mono text-white/18 tracking-wider uppercase bg-white/[0.04] px-1.5 py-0.5 rounded-md">{m.type}</span>
                            </div>
                            <div className="text-[13px] font-semibold text-white/80 mb-1 truncate leading-snug">{m.name}</div>
                            <div className="text-[10px] font-mono text-white/28 tabular-nums">ep {m.epoch} · loss {m.loss.toFixed(3)}</div>
                            <div className="text-[10px] text-white/16 mt-0.5">{new Date(m.date).toLocaleDateString()}</div>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDeleteModel(m.id); }}
                            className="absolute top-3 right-3 size-6 rounded-lg flex items-center justify-center text-white/12 hover:text-red-400 hover:bg-red-400/10 opacity-0 group-hover:opacity-100 transition-all"
                          >
                            <Trash2 className="size-3" />
                          </button>
                        </div>
                      ))}
                      <button
                        onClick={() => setTrainStep("setup")}
                        className="rounded-2xl border-2 border-dashed border-white/[0.07] hover:border-[#0A84FF]/28 hover:bg-[#0A84FF]/[0.025] transition-all flex flex-col items-center justify-center gap-2.5 min-h-[140px] text-white/22 hover:text-[#0A84FF]/55"
                      >
                        <Plus className="size-6" />
                        <span className="text-[12px] font-medium">New Base Model</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* ─── Configuration Setup ──────────────────────────────────── */}
              {trainStep === "setup" && (
                <div className="h-full overflow-y-auto">
                  <div className="max-w-md mx-auto py-8 px-5 space-y-5">
                    <button
                      onClick={() => setTrainStep("fleet")}
                      className="flex items-center gap-1.5 text-white/28 hover:text-white/58 text-[12px] transition-colors"
                    >
                      <ArrowLeft className="size-4" />
                      Fleet
                    </button>
                    <div>
                      <h2 className="text-[17px] font-semibold text-white/88 tracking-tight">New Model Configuration</h2>
                      <p className="text-[12px] text-white/28 mt-0.5 leading-relaxed">
                        Set the base architecture for your new language model.
                      </p>
                    </div>

                    {/* Model name */}
                    <div className="space-y-1.5">
                      <label className="text-[11px] text-white/32 font-medium">Model Name</label>
                      <input
                        type="text"
                        value={newModelName}
                        onChange={(e) => setNewModelName(e.target.value)}
                        placeholder="My Language Model"
                        className="w-full h-11 rounded-xl bg-[#0d0d0d] border border-white/[0.06] px-4 text-[13px] text-white/78 placeholder:text-white/18 focus:outline-none focus:border-[#0A84FF]/38 transition-colors"
                      />
                    </div>

                    {/* Tokenization */}
                    <div className="space-y-1.5">
                      <label className="text-[11px] text-white/32 font-medium">Tokenization</label>
                      <div className="grid grid-cols-2 gap-1.5 bg-[#090909] rounded-xl border border-white/[0.05] p-1.5">
                        <button
                          onClick={() => setLLMConfig((c) => ({ ...c, tokenization: "char" }))}
                          className={`h-9 rounded-lg text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-all ${llmConfig.tokenization === "char" ? "bg-[#0A84FF]/14 text-[#0A84FF] border border-[#0A84FF]/18" : "text-white/28 hover:text-white/55"}`}
                        >
                          Char-Level
                        </button>
                        <button
                          onClick={() => setLLMConfig((c) => ({ ...c, tokenization: "word" }))}
                          className={`h-9 rounded-lg text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-all ${llmConfig.tokenization === "word" ? "bg-[#30D158]/10 text-[#30D158] border border-[#30D158]/18" : "text-white/28 hover:text-white/55"}`}
                        >
                          Word-Level
                        </button>
                      </div>
                    </div>

                    {/* Context slider */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center">
                        <label className="text-[11px] text-white/32 font-medium">Context Window</label>
                        <span className="text-[11px] tabular-nums text-white/58 font-mono">{llmConfig.contextSize}</span>
                      </div>
                      <Slider min={1} max={20} step={1} value={[llmConfig.contextSize]} onValueChange={([v]) => setLLMConfig((c) => ({ ...c, contextSize: v }))} className="py-2" />
                    </div>

                    {/* Hidden slider */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center">
                        <label className="text-[11px] text-white/32 font-medium">Hidden Neurons</label>
                        <span className="text-[11px] tabular-nums text-white/58 font-mono">{llmConfig.hiddenSize}</span>
                      </div>
                      <Slider min={4} max={512} step={4} value={[llmConfig.hiddenSize]} onValueChange={([v]) => setLLMConfig((c) => ({ ...c, hiddenSize: v }))} className="py-2" />
                    </div>

                    {/* Learning rate slider */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center">
                        <label className="text-[11px] text-white/32 font-medium">Learning Rate</label>
                        <span className="text-[11px] tabular-nums text-white/58 font-mono">{llmConfig.learningRate.toFixed(3)}</span>
                      </div>
                      <Slider min={0.005} max={0.5} step={0.005} value={[llmConfig.learningRate]} onValueChange={([v]) => setLLMConfig((c) => ({ ...c, learningRate: v }))} className="py-2" />
                    </div>

                    {/* Param count */}
                    <div className="rounded-xl bg-[#090909] border border-white/[0.05] px-4 py-3 flex items-center justify-between">
                      <span className="text-[11px] text-white/28">Est. parameters</span>
                      <span className={`text-[11px] tabular-nums font-mono ${estimatedLLMParams > MAX_PARAMS_LLM ? "text-red-400" : "text-white/55"}`}>
                        {estimatedLLMParams.toLocaleString()}
                      </span>
                    </div>

                    <button
                      onClick={saveNewModel}
                      disabled={estimatedLLMParams > MAX_PARAMS_LLM}
                      className="w-full h-12 rounded-xl bg-[#0A84FF] hover:bg-[#409CFF] text-white font-semibold text-[13px] flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Play className="size-4" />
                      Initialize &amp; Begin Training
                    </button>
                  </div>
                </div>
              )}

              {/* ─── Active Training Dashboard ──────────────────────────────── */}
              {trainStep === "training" && (
                <div className="h-full flex flex-col overflow-hidden">

                  {/* ── Mini-bar navigation + back breadcrumb ─────────────── */}
                  <div className="shrink-0 border-b border-white/[0.05] bg-[#050505] px-3 flex items-center gap-1 h-11">
                    <button
                      onClick={() => setTrainStep("fleet")}
                      className="flex items-center gap-1.5 h-7 px-2.5 rounded-lg text-white/25 hover:text-white/55 text-[11px] hover:bg-white/[0.04] transition-all mr-1 border border-transparent hover:border-white/[0.05]"
                    >
                      <ArrowLeft className="size-3" />
                      <span className="hidden sm:inline">Models</span>
                    </button>
                    <div className="w-px h-4 bg-white/[0.07] mr-1" />
                    <MiniBarTab
                      icon={Cpu}
                      label="Architecture & Hypers"
                      active={trainTab === "arch"}
                      onClick={() => setTrainTab("arch")}
                    />
                    <MiniBarTab
                      icon={Database}
                      label="Dataset Preparation"
                      active={trainTab === "dataset"}
                      onClick={() => setTrainTab("dataset")}
                    />
                    <MiniBarTab
                      icon={ScrollText}
                      label="Dataset Explorer"
                      active={trainTab === "explorer"}
                      onClick={() => setTrainTab("explorer")}
                    />
                    <MiniBarTab
                      icon={Activity}
                      label="Training Status"
                      active={trainTab === "status"}
                      onClick={() => setTrainTab("status")}
                    />
                  </div>

                  {/* ── Tab: Architecture & Hypers ─────────────────────────── */}
                  {trainTab === "arch" && (
                    <div className="flex-1 overflow-y-auto p-4">
                      <LLMArchitect
                        config={llmConfig}
                        onChange={setLLMConfig}
                        onApply={rebuildLLM}
                        paramCount={estimatedLLMParams}
                        vocabSize={Math.max(2, vocabSizeFor(llmConfig.corpus, llmConfig.tokenization))}
                        maxParams={MAX_PARAMS_LLM}
                        datasets={datasets}
                        onDatasetsChange={setDatasets}
                        section="arch"
                      />
                    </div>
                  )}

                  {/* ── Tab: Dataset Preparation (non-scrollable tools) ────── */}
                  {trainTab === "dataset" && (
                    <div className="flex-1 overflow-y-auto p-4">
                      <LLMArchitect
                        config={llmConfig}
                        onChange={setLLMConfig}
                        onApply={rebuildLLM}
                        paramCount={estimatedLLMParams}
                        vocabSize={Math.max(2, vocabSizeFor(llmConfig.corpus, llmConfig.tokenization))}
                        maxParams={MAX_PARAMS_LLM}
                        datasets={datasets}
                        onDatasetsChange={setDatasets}
                        section="dataset-tools"
                      />
                    </div>
                  )}

                  {/* ── Tab: Dataset Explorer (scrollable rows) ─────────────── */}
                  {trainTab === "explorer" && (
                    <div className="flex-1 overflow-y-auto">
                      <div className="p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <ScrollText className="size-4 text-white/35" />
                            <span className="text-[13px] font-semibold text-white/80">Dataset Explorer</span>
                          </div>
                          <span className="text-[11px] text-white/28 font-mono tabular-nums">
                            {datasets.filter((d) => d.active).length}/{datasets.length} active
                          </span>
                        </div>

                        {datasets.length === 0 ? (
                          <div className="flex flex-col items-center justify-center py-16 gap-3">
                            <div className="size-12 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center">
                              <Database className="size-5 text-white/15" />
                            </div>
                            <div className="text-[13px] text-white/28 text-center">No datasets yet.</div>
                            <div className="text-[11px] text-white/15 text-center max-w-[200px] leading-relaxed">
                              Add data via the Dataset Preparation tab.
                            </div>
                          </div>
                        ) : (
                          datasets.map((d) => {
                            const bytes = new Blob([d.text]).size;
                            const lines = d.text.split("\n").filter((l) => l.trim()).length;
                            return (
                              <div
                                key={d.id}
                                className={`rounded-2xl border transition-all ${
                                  d.active
                                    ? "border-white/[0.07] bg-[#0d0d0d]"
                                    : "border-white/[0.04] bg-[#080808] opacity-50"
                                }`}
                              >
                                {/* Row header */}
                                <div className="flex items-center gap-3 px-4 py-3 border-b border-white/[0.04]">
                                  <span className={`size-2 rounded-full shrink-0 ${d.active ? "bg-[#0A84FF]" : "bg-white/15"}`} />
                                  <span className="flex-1 text-[12px] font-semibold text-white/75 truncate">{d.name}</span>
                                  <span className="text-[10px] text-white/22 font-mono tabular-nums shrink-0">
                                    {lines} lines · {bytes < 1024 ? `${bytes}B` : `${(bytes / 1024).toFixed(1)}KB`}
                                  </span>
                                </div>
                                {/* Raw text preview */}
                                <pre className="px-4 py-3 text-[10px] font-mono text-white/30 whitespace-pre-wrap leading-relaxed break-words">
                                  {d.text.slice(0, 1200)}{d.text.length > 1200 ? "\n…" : ""}
                                </pre>
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>
                  )}

                  {/* ── Tab: Training Status ────────────────────────────────── */}
                  {trainTab === "status" && (
                    <div className="flex-1 overflow-y-auto">
                      {/* Backpropagation visualization */}
                      <div className="flex flex-col items-center justify-center py-10 px-6">
                        <div className="relative flex items-center justify-center mb-8">
                          <div className="absolute rounded-full border border-[#0A84FF]/22" style={{ width: 180, height: 180, animation: llmPlaying ? "backprop-ring 2.4s ease-out infinite 0s" : "none", opacity: llmPlaying ? 1 : 0.15 }} />
                          <div className="absolute rounded-full border border-[#0A84FF]/15" style={{ width: 180, height: 180, animation: llmPlaying ? "backprop-ring 2.4s ease-out infinite 0.8s" : "none", opacity: llmPlaying ? 1 : 0.10 }} />
                          <div className="absolute rounded-full border border-[#0A84FF]/10" style={{ width: 180, height: 180, animation: llmPlaying ? "backprop-ring 2.4s ease-out infinite 1.6s" : "none", opacity: llmPlaying ? 1 : 0.06 }} />
                          <div
                            className={`size-24 rounded-full flex items-center justify-center border ${
                              llmPlaying
                                ? "bg-[#0A84FF]/10 border-[#0A84FF]/20 shadow-[0_0_40px_rgba(10,132,255,0.15)]"
                                : "bg-white/[0.03] border-white/[0.06]"
                            }`}
                            style={{ animation: llmPlaying ? "backprop-orb 2s ease-in-out infinite" : "none" }}
                          >
                            <Brain className={`size-9 ${llmPlaying ? "text-[#0A84FF]/70" : "text-white/18"}`} />
                          </div>
                        </div>

                        <div className={`text-[11px] font-semibold tracking-[0.15em] uppercase mb-6 ${llmPlaying ? "text-[#0A84FF]" : "text-white/22"}`}>
                          {llmPlaying ? "Active Backpropagation" : "Paused"}
                        </div>

                        {/* Live telemetry — always visible, updates in real-time from textSnap */}
                        <div className="bg-[#080808] border border-white/[0.06] rounded-2xl px-6 py-4 flex items-center gap-0 w-full max-w-md">
                          <MetricCell label="Loss" value={textSnap.loss > 0 ? textSnap.loss.toFixed(4) : "—"} />
                          <div className="w-px h-10 bg-white/[0.06]" />
                          <MetricCell label="Epoch" value={textSnap.epoch > 0 ? String(textSnap.epoch) : "—"} />
                          <div className="w-px h-10 bg-white/[0.06]" />
                          <MetricCell
                            label="tok/s"
                            value={textSnap.tokensPerSecond > 0
                              ? textSnap.tokensPerSecond > 1000
                                ? `${(textSnap.tokensPerSecond / 1000).toFixed(1)}k`
                                : String(Math.round(textSnap.tokensPerSecond))
                              : "—"}
                          />
                          <div className="w-px h-10 bg-white/[0.06]" />
                          <MetricCell
                            label="Tokens"
                            value={textSnap.trainedSamples > 0
                              ? textSnap.trainedSamples > 1000
                                ? `${(textSnap.trainedSamples / 1000).toFixed(1)}k`
                                : String(textSnap.trainedSamples)
                              : "—"}
                          />
                        </div>

                        {textSnap.sample && (
                          <div className="mt-5 w-full max-w-md rounded-xl bg-[#080808] border border-white/[0.05] px-4 py-3">
                            <div className="text-[9px] uppercase tracking-[0.12em] text-white/22 mb-2 font-medium">Live Sample</div>
                            <code className="text-[10px] font-mono text-[#30D158]/55 leading-relaxed break-words">{textSnap.sample}</code>
                          </div>
                        )}
                      </div>

                      {/* Train speed */}
                      <div className="px-4 pb-5 max-w-md mx-auto">
                        <div className="rounded-2xl border border-white/[0.05] bg-[#080808] p-4 space-y-3">
                          <div className="flex items-center gap-2">
                            <Zap className="size-4 text-amber-400" />
                            <span className="text-[12px] font-semibold text-white/72">Train Speed</span>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-[11px] text-white/30">Epochs / sec</span>
                              <span className="text-[11px] tabular-nums text-white/60 font-mono">{llmEpochsPerSecond}</span>
                            </div>
                            <Slider min={1} max={60} step={1} value={[llmEpochsPerSecond]} onValueChange={handleLLMSpeed} className="py-2" />
                          </div>
                        </div>
                      </div>

                      {/* Stats & sharing */}
                      {llmHasModel && (
                        <div className="px-4 pb-6 space-y-4 max-w-md mx-auto">
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
                            liveSample={textSnap.sample}
                            tokenization={llmConfig.tokenization}
                          />
                          <SharingHub
                            mode="llm"
                            onDownload={handleOpenSave}
                            hasModel={llmHasModel}
                            onImport={handleImportModel}
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────
              CHAT STUDIO
          ──────────────────────────────────────────────────────────────── */}
          {tab === "chat" && (
            <div className="h-full overflow-hidden">
              <ChatView
                modelLabel={llmModelLabel}
                messages={messages}
                setMessages={setMessages}
                loading={chatLoading}
                setLoading={setChatLoading}
                generate={generateFromWorker}
                liveSample={textSnap.sample}
                epoch={textSnap.epoch}
                loss={textSnap.loss}
                isTraining={llmPlaying}
                modelOptions={llmModels.map((m) => ({ id: m.id, label: m.name }))}
                onSelectModel={(id) => {
                  const m = savedModels.find((s) => s.id === id);
                  if (m) handleLoadModel(m);
                }}
              />
            </div>
          )}

          {/* ────────────────────────────────────────────────────────────────
              DEPLOY HUB
          ──────────────────────────────────────────────────────────────── */}
          {tab === "deploy" && (
            <div className="h-full overflow-y-auto">
              <DeployHub
                modelLabel={llmModelLabel}
                hasModel={llmHasModel}
                epoch={textSnap.epoch}
                loss={textSnap.loss}
              />
            </div>
          )}

        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          MOBILE BOTTOM NAV
      ══════════════════════════════════════════════════════════════════════ */}
      <nav className="md:hidden fixed bottom-0 inset-x-0 z-20 border-t border-white/[0.05] bg-[#000000]/96 backdrop-blur-xl">
        <div className="grid grid-cols-4">
          <MobileNavButton icon={MessageSquare} label="Chat" active={tab === "chat"} onClick={() => setTab("chat")} />
          <MobileNavButton icon={SlidersHorizontal} label="Training" active={tab === "train"} onClick={() => setTab("train")} />
          <MobileNavButton icon={Zap} label="Deploy" active={tab === "deploy"} onClick={() => setTab("deploy")} />
          <button
            onClick={handlePlay}
            className={`flex flex-col items-center justify-center gap-1 py-2.5 min-h-[60px] transition-colors ${llmPlaying ? "text-[#0A84FF]" : "text-slate-400 hover:text-slate-200"}`}
          >
            {llmPlaying ? <Pause className="size-5" /> : <Play className="size-5" />}
            <span className="text-[10px] font-medium tracking-wide uppercase">{llmPlaying ? "Pause" : "Train"}</span>
          </button>
        </div>
      </nav>

      <SaveModal
        open={saveOpen}
        defaultName={defaultSaveName(llmModelLabel, textSnap.epoch, llmConfig.tokenization)}
        modeLabel={llmConfig.tokenization === "word" ? "Word-level LM" : "Char-level LM"}
        hasModel={llmHasModel}
        onClose={() => setSaveOpen(false)}
        onSaveToLibrary={handleSaveToLibrary}
        onExportFile={handleExportFile}
      />

      <Toaster />
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function SlimNavItem({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={label}
      className={`relative w-full h-10 rounded-xl flex items-center justify-center transition-all group ${
        active
          ? "bg-[#0A84FF]/10 text-[#0A84FF] border border-[#0A84FF]/15"
          : "text-white/28 hover:text-white/62 hover:bg-white/[0.04] border border-transparent"
      }`}
    >
      <Icon className="size-[18px] shrink-0" />
      <span className="absolute left-full ml-3 z-50 whitespace-nowrap bg-[#161616] border border-white/[0.08] text-white/70 text-[11px] font-medium px-2.5 py-1.5 rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity shadow-xl">
        {label}
      </span>
    </button>
  );
}

function MiniBarTab({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 h-8 px-3 rounded-lg text-[11px] font-medium transition-all ${
        active
          ? "bg-[#0A84FF]/12 text-[#0A84FF] border border-[#0A84FF]/18"
          : "text-white/30 hover:text-white/62 hover:bg-white/[0.04] border border-transparent"
      }`}
    >
      <Icon className="size-3.5 shrink-0" />
      <span className="hidden sm:inline">{label}</span>
    </button>
  );
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center flex-1">
      <div className="text-[9px] uppercase tracking-[0.12em] text-white/22 mb-1.5 font-medium">{label}</div>
      <div className="text-xl font-bold tabular-nums text-white">{value}</div>
    </div>
  );
}

function MobileNavButton({
  icon: Icon,
  label,
  active,
  onClick,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex flex-col items-center justify-center gap-1 py-2.5 min-h-[60px] transition-colors ${active ? "text-[#0A84FF]" : "text-slate-400 hover:text-slate-200"}`}
    >
      <Icon className={`size-5 ${active ? "drop-shadow-[0_0_8px_rgba(10,132,255,0.45)]" : ""}`} />
      <span className="text-[10px] font-medium tracking-wide uppercase">{label}</span>
    </button>
  );
}
