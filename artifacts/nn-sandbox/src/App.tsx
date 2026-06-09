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
  Library as LibraryIcon,
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
  type Dataset,
} from "@/components/llm/LLMArchitect";
import { ChatView, type ChatMessage } from "@/components/llm/ChatView";
import { LLMStats } from "@/components/llm/LLMStats";
import { SaveModal } from "@/components/SaveModal";
import { LibraryView } from "@/components/LibraryView";
import {
  saveModel,
  makeId,
  type SavedModel,
  type MLPWeights,
  type CharLMWeights,
} from "@/lib/storage";
import type { Activation, NetworkConfig, DataPoint } from "@/lib/nn";
import { tokenize, type Tokenization } from "@/lib/textnet";

type DatasetKind = "spiral" | "circle" | "xor";
type TabKey = "architect" | "brain" | "output" | "library";

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

// Mirrors `normalizeText` from text.worker.ts. Kept as a small inline helper
// here so the main thread doesn't have to import the worker module (which
// would also pull in its `self.onmessage` registration).
function normalizePromptForLLM(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// Function-word filter for the OOV shield in `generateFromWorker`. These are
// the connectors that the model's training prompts ("tell me about X",
// "what is X") are wrapped in — they almost always exist in the vocab and
// would mask a prompt whose actual content words are all unknown. Stripping
// them lets us decide "is the *topic* foreign?" rather than "are any of the
// surface tokens foreign?".
const OOV_STOP_WORDS = new Set([
  "tell",
  "me",
  "about",
  "what",
  "is",
  "a",
  "the",
]);

const OOV_FALLBACK_REPLY =
  "I don't know the answer to that. I haven't been trained on this topic yet.";

// Standard Levenshtein edit distance. Uses the two-row DP variant so memory
// stays at O(min(m, n)) instead of O(m * n) — matters when the autocorrect
// pass scans an unknown user word against every entry in a large vocab.
// Returns the minimum number of single-character insertions, deletions, or
// substitutions needed to transform `a` into `b`.
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
      curr[j] = Math.min(
        curr[j - 1] + 1, // insertion
        prev[j] + 1, // deletion
        prev[j - 1] + cost, // substitution
      );
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n];
}

function uniqueChars(s: string): number {
  return new Set(s).size;
}

function vocabSizeFor(corpus: string, mode: Tokenization): number {
  // The worker normalizes the corpus before building its vocab, so estimate
  // against the normalized form to keep the UI's param/vocab numbers honest.
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

function defaultSaveName(
  mode: AppMode,
  modeLabel: string,
  epoch: number,
  tokenization?: "char" | "word",
): string {
  const tag =
    mode === "mlp"
      ? "MLP"
      : tokenization === "word"
        ? "Word-LM"
        : "Char-LM";
  return `${tag} · ${modeLabel} · ep ${epoch}`;
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
      className={`bg-apple-card/40 rounded-2xl border border-apple-divider/10 backdrop-blur-md ${className}`}
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
  // Datasets are the single source of truth for the user's individual files.
  // The flattened `corpus` string in `llmConfig` is derived from datasets by
  // <LLMArchitect/> and pushed back up via onChange — never the other way
  // around. Lifted here (rather than inside the architect) so model loads can
  // selectively restore datasets when the saved JSON contains them, without
  // ever overwriting them from a flattened corpus blob.
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

  const [tab, setTab] = useState<TabKey>("brain");
  const [saveOpen, setSaveOpen] = useState(false);
  const [libraryRefresh, setLibraryRefresh] = useState(0);

  const layers = useMemo(() => [2, ...hidden, 1], [hidden]);
  const estimatedMLPParams = useMemo(
    () => estimateMLPParams(layers),
    [layers],
  );
  const estimatedLLMParams = useMemo(
    () =>
      estimateLLMParams(
        Math.max(2, vocabSizeFor(llmConfig.corpus, llmConfig.tokenization)),
        llmConfig.contextSize,
        llmConfig.hiddenSize,
      ),
    [
      llmConfig.corpus,
      llmConfig.contextSize,
      llmConfig.hiddenSize,
      llmConfig.tokenization,
    ],
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
        tokenization: llmConfig.tokenization,
        topK: llmConfig.topK,
      },
    });
    return () => worker.terminate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push live config tweaks (lr/temperature/topK) to the LLM worker without
  // resetting weights, so the user can tune sampling on the fly mid-chat.
  useEffect(() => {
    textWorkerRef.current?.postMessage({
      type: "config",
      partial: {
        learningRate: llmConfig.learningRate,
        temperature: llmConfig.temperature,
        topK: llmConfig.topK,
      },
    });
  }, [llmConfig.learningRate, llmConfig.temperature, llmConfig.topK]);

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
        tokenization: llmConfig.tokenization,
        topK: llmConfig.topK,
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

        let normalizedInput = normalizePromptForLLM(seed);

        // ── Autocorrect + OOV semantic shield (word mode only) ─────────────
        // Word-level tokenization is brittle: a single typo turns a valid
        // training word into an unknown token, which would otherwise either
        // fire the OOV shield or get fed to the model as gibberish. So we
        // do two things in sequence on the normalized prompt:
        //
        //  1. AUTOCORRECT each unknown content word against the live vocab
        //     using Levenshtein distance, snapping it to its nearest
        //     neighbour when the edit distance is small enough that the
        //     user clearly *meant* the vocab word:
        //        • words < 5 chars  → tolerate at most 1 edit
        //        • words ≥ 5 chars  → tolerate up to 2 edits
        //     Stopwords and already-known words are passed through.
        //
        //  2. Re-run the OOV shield against the *corrected* words. A typo
        //     we successfully fixed should now be in vocab and proceed to
        //     the model; only genuinely foreign topics still trip the shield.
        //
        // Mirrors the worker's own pipeline (`normalizeText` → `tokenize`)
        // exactly so a word judged "in vocab" here is the same word the
        // worker would have looked up.
        //
        // Char mode is intentionally excluded: its vocab is single
        // characters, so neither typo-snapping nor "unseen word" is a
        // meaningful concept.
        if (llmConfig.tokenization === "word") {
          const promptWords = normalizedInput.split(" ").filter(Boolean);
          const corpusVocab = new Set(
            tokenize(normalizePromptForLLM(llmConfig.corpus), "word"),
          );
          const corpusVocabArr = Array.from(corpusVocab);

          const correctedWords = promptWords.map((w) => {
            if (corpusVocab.has(w) || OOV_STOP_WORDS.has(w)) return w;
            const threshold = w.length < 5 ? 1 : 2;
            let best = w;
            let bestDist = Infinity;
            for (const v of corpusVocabArr) {
              // Length-difference lower bound: levenshtein(a, b) is at
              // least ||a| - |b||, so candidates that are too far apart
              // in length can be skipped without running the full DP —
              // typically prunes most of a large vocab for free.
              if (Math.abs(v.length - w.length) > threshold) continue;
              const d = levenshteinDistance(w, v);
              if (d < bestDist) {
                bestDist = d;
                best = v;
                if (d === 0) break;
              }
            }
            return bestDist <= threshold ? best : w;
          });

          const meaningful = correctedWords.filter(
            (w) => !OOV_STOP_WORDS.has(w),
          );
          if (
            meaningful.length > 0 &&
            meaningful.every((w) => !corpusVocab.has(w))
          ) {
            resolve(OOV_FALLBACK_REPLY);
            return;
          }

          // ── Intent router (best-overlap-wins) ────────────────────────
          // Tiny LMs only reliably reproduce prompts they were literally
          // trained on. A user typing the bare keyword "liverpool" never
          // appears verbatim in the corpus, even though the corpus has a
          // "User: tell me about liverpool" line whose answer is exactly
          // what the user wants. So before handing the seed to the worker
          // we try to map the user's keywords onto a canonical training
          // prompt:
          //
          //   1. Pull every `User: …` line out of the corpus, strip the
          //      speaker tag, and defensively chop off any inline `Bot: …`
          //      tail so a malformed single-line entry can't leak the
          //      assistant's reply into the keyword pool. The `\bBot:`
          //      word boundary keeps incidental words like "robot:" from
          //      matching.
          //   2. For each, normalize and drop stopwords to get the prompt's
          //      "core keywords" (the same shape as `meaningful`).
          //   3. Score each prompt by the size of its core-keyword set
          //      intersected with the user's meaningful keyword set.
          //   4. The highest-scoring prompt with a non-zero overlap wins
          //      and silently replaces the user's input. Ties resolve to
          //      whichever prompt appears first in the corpus, which is a
          //      reasonable proxy for "more canonical".
          //
          // Skip the router entirely when `meaningful` is empty: there's
          // nothing to overlap on. If nothing overlaps we fall back to the
          // autocorrected sentence — the model still gets a clean prompt,
          // just not a routed one.
          let routedPrompt: string | null = null;
          if (meaningful.length > 0) {
            const meaningfulSet = new Set(meaningful);
            const knownPrompts = llmConfig.corpus
              .split("\n")
              .filter((l) => /^\s*User:\s/i.test(l))
              .map((l) =>
                l
                  .replace(/^\s*User:\s*/i, "")
                  .replace(/\bBot:\s.*$/i, "")
                  .trim(),
              );
            let bestScore = 0;
            for (const kp of knownPrompts) {
              const kpNormalized = normalizePromptForLLM(kp);
              const kpCore = new Set(
                kpNormalized
                  .split(" ")
                  .filter((w) => w && !OOV_STOP_WORDS.has(w)),
              );
              let overlap = 0;
              for (const w of kpCore) {
                if (meaningfulSet.has(w)) overlap++;
              }
              if (overlap > bestScore) {
                bestScore = overlap;
                routedPrompt = kpNormalized;
              }
            }
          }

          // Use the routed canonical prompt when one was found; otherwise
          // fall back to the typo-fixed sentence so the model still sees
          // a clean version of what the user actually typed.
          normalizedInput = routedPrompt ?? correctedWords.join(" ");
        }

        const id = `g-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        pendingGenRef.current.set(id, resolve);

        // Build a seed that strictly mirrors the normalized corpus format so
        // the model is primed to output the bot's turn immediately:
        //   "[system prompt] user [message] bot "
        //
        // The trailing "bot " forces the model to predict the next token
        // *after* the speaker tag rather than wasting capacity re-generating
        // the tag itself (which caused hallucinations and empty replies).
        const systemPart = llmConfig.systemPrompt
          ? normalizePromptForLLM(llmConfig.systemPrompt) + " "
          : "";
        const formattedSeed = `${systemPart}user ${normalizedInput} bot `;

        worker.postMessage({
          type: "generate",
          id,
          seed: formattedSeed,
          length: 300,
          temperature: llmConfig.temperature,
        });
      }),
    [
      llmConfig.temperature,
      llmConfig.systemPrompt,
      llmConfig.corpus,
      llmConfig.tokenization,
    ],
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
          // Tag the saved model with the live corpus + temperature so loads
          // restore a fully-usable chat session.
          parsed.corpus = llmConfig.corpus;
          parsed.temperature = llmConfig.temperature;
          parsed.tokenization = llmConfig.tokenization;
          // Snapshot the user's individual datasets (without their UI-only
          // numeric ids) so the Dataset Manager can be restored on load
          // instead of collapsing into a single "Base Training" file.
          parsed.datasets = datasets.map(({ name, text, active }) => ({
            name,
            text,
            active,
          }));
          resolve(parsed as CharLMWeights & { corpus: string });
        } catch {
          resolve(null);
        }
      });
      worker.postMessage({ type: "exportModel", id });
    });
  }, [
    llmConfig.corpus,
    llmConfig.temperature,
    llmConfig.tokenization,
    datasets,
  ]);

  const downloadJSON = (filename: string, payload: unknown) => {
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const buildMLPPayload = (): MLPWeights | null => {
    if (!snap) return null;
    return {
      layers,
      activation,
      weights: snap.weights,
      biases: snap.biases,
      learningRate,
    };
  };

  const handleOpenSave = () => {
    const ready = mode === "mlp" ? !!snap : textSnap.epoch > 0;
    if (!ready) {
      toast({
        title: "Train first",
        description: "Run a few epochs before saving.",
      });
      return;
    }
    setSaveOpen(true);
  };

  const handleExportFile = async (name: string) => {
    if (mode === "mlp") {
      const w = buildMLPPayload();
      if (!w) return;
      downloadJSON(`${slug(name)}.json`, { kind: "mlp", name, ...w });
      toast({ title: "File exported", description: `${name}.json saved.` });
    } else {
      const w = await requestLLMExport();
      if (!w) {
        toast({ title: "Nothing to export", description: "Train first." });
        return;
      }
      downloadJSON(`${slug(name)}.json`, { kind: "char-lm", name, ...w });
      toast({ title: "File exported", description: `${name}.json saved.` });
    }
  };

  const handleSaveToLibrary = async (name: string) => {
    if (mode === "mlp") {
      const w = buildMLPPayload();
      if (!w || !snap) return;
      const model: SavedModel = {
        id: makeId(),
        name,
        type: "MLP",
        date: Date.now(),
        paramsCount: snap.paramCount,
        loss: snap.loss,
        epoch: snap.epoch,
        weights: w,
      };
      await saveModel(model);
    } else {
      const w = await requestLLMExport();
      if (!w) {
        toast({ title: "Nothing to save", description: "Train first." });
        return;
      }
      const model: SavedModel = {
        id: makeId(),
        name,
        type: "Char-LM",
        date: Date.now(),
        paramsCount: textSnap.paramCount,
        loss: textSnap.loss,
        epoch: textSnap.epoch,
        weights: w,
      };
      await saveModel(model);
    }
    setLibraryRefresh((v) => v + 1);
    toast({
      title: "Saved to Library",
      description: `${name} is in your model library.`,
    });
  };

  const handleLoadModel = (model: SavedModel) => {
    // Always pause both training loops first so the new weights land cleanly.
    if (playing) {
      workerRef.current?.postMessage({ type: "pause" });
      setPlaying(false);
    }
    if (llmPlaying) {
      textWorkerRef.current?.postMessage({ type: "pause" });
      setLLMPlaying(false);
    }

    if (model.type === "MLP") {
      const w = model.weights as MLPWeights;
      const nextHidden = w.layers.slice(1, -1);
      setHidden(nextHidden);
      setActivation(w.activation as Activation);
      setLearningRate(w.learningRate);
      const config: NetworkConfig = {
        inputSize: w.layers[0],
        hiddenLayers: nextHidden,
        outputSize: w.layers[w.layers.length - 1],
        activation: w.activation as Activation,
        learningRate: w.learningRate,
      };
      workerRef.current?.postMessage({
        type: "loadWeights",
        config,
        dataset,
        weights: w.weights,
        biases: w.biases,
      });
      setMode("mlp");
    } else {
      const w = model.weights as CharLMWeights;
      // Mirror the saved config into the LLM panel UI so the architect tab
      // matches what the worker is now serving.
      //
      // Squished-text bug fix: older saves pre-date the `tokenization` field
      // and would silently fall back to "char", which made restored word-
      // level models output strings like "goodbyehaveaniceday" with no
      // spaces. Sniff the vocab in that case — any token longer than one
      // character can only have come from word-level tokenization.
      const restoredTok: Tokenization =
        w.tokenization === "word" || w.tokenization === "char"
          ? w.tokenization
          : (w.vocab.some((t) => typeof t === "string" && t.length > 1)
              ? "word"
              : "char");
      setLLMConfig((prev) => ({
        ...prev,
        corpus:
          w.corpus ?? w.vocab.join(restoredTok === "word" ? " " : ""),
        contextSize: w.config.contextSize,
        hiddenSize: w.config.hiddenSize,
        learningRate: w.config.learningRate,
        temperature: typeof w.temperature === "number" ? w.temperature : 0.6,
        tokenization: restoredTok,
      }));
      // Restore datasets ONLY when the saved file explicitly carries them.
      // We never derive datasets from the flattened corpus string — doing so
      // would collapse the user's individual files into one giant blob (and
      // contaminate it with the <PAD>-wall document separators). Older saves
      // pre-date this field and simply leave the user's existing datasets in
      // place, which is the documented graceful fallback.
      if (Array.isArray(w.datasets) && w.datasets.length > 0) {
        const baseId = Date.now();
        setDatasets(
          w.datasets
            .filter(
              (d): d is { name: string; text: string; active: boolean } =>
                !!d &&
                typeof d === "object" &&
                typeof (d as { name?: unknown }).name === "string" &&
                typeof (d as { text?: unknown }).text === "string" &&
                typeof (d as { active?: unknown }).active === "boolean",
            )
            .map((d, i) => ({
              id: baseId + i,
              name: d.name,
              text: d.text,
              active: d.active,
            })),
        );
      }
      textWorkerRef.current?.postMessage({
        type: "loadWeights",
        payload: {
          ...w,
          epoch: model.epoch,
          loss: model.loss,
        },
      });
      setMessages([]);
      setMode("llm");
    }
    setTab("brain");
    toast({
      title: "Model loaded",
      description: `${model.name} restored to the ${model.type} workspace.`,
    });
  };

  const handleImportModel = (json: string) => {
    try {
      const parsed = JSON.parse(json) as Record<string, unknown>;
      if (!parsed || typeof parsed !== "object") {
        toast({ title: "Import failed", description: "Invalid file." });
        return;
      }
      const kind = parsed.kind as string | undefined;
      if (kind === "char-lm") {
        const model: SavedModel = {
          id: makeId(),
          name: (parsed.name as string | undefined) ?? "Imported Model",
          type: "Char-LM",
          date: Date.now(),
          paramsCount: 0,
          loss: (parsed.loss as number | undefined) ?? 0,
          epoch: (parsed.epoch as number | undefined) ?? 0,
          weights: parsed as unknown as CharLMWeights,
        };
        handleLoadModel(model);
      } else if (kind === "mlp") {
        const model: SavedModel = {
          id: makeId(),
          name: (parsed.name as string | undefined) ?? "Imported Model",
          type: "MLP",
          date: Date.now(),
          paramsCount: 0,
          loss: 0,
          epoch: 0,
          weights: parsed as unknown as MLPWeights,
        };
        handleLoadModel(model);
      } else {
        toast({
          title: "Import failed",
          description: "Unrecognized model format.",
        });
      }
    } catch {
      toast({
        title: "Import failed",
        description: "Could not parse the JSON file.",
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

  const llmModeTag = llmConfig.tokenization === "word" ? "Word-LM" : "Char-LM";
  const llmModelLabel = `${llmModeTag} · ${llmConfig.contextSize}→${llmConfig.hiddenSize}→${textSnap.vocabSize}`;

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

      <SharingHub mode="mlp" onDownload={handleOpenSave} hasModel={!!snap} onImport={handleImportModel} />
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
        vocabSize={Math.max(
          2,
          vocabSizeFor(llmConfig.corpus, llmConfig.tokenization),
        )}
        maxParams={MAX_PARAMS_LLM}
        datasets={datasets}
        onDatasetsChange={setDatasets}
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
        liveSample={textSnap.sample}
        tokenization={llmConfig.tokenization}
      />
      <SharingHub mode="llm" onDownload={handleOpenSave} hasModel={llmHasModel} onImport={handleImportModel} />
    </div>
  );

  const ArchitectContent = mode === "mlp" ? MLPArchitect : LLMArchitectView;
  const BrainContent = mode === "mlp" ? MLPBrain : LLMBrain;
  const OutputContent = mode === "mlp" ? MLPOutput : LLMOutput;

  const LibraryContent = (
    <LibraryView
      refreshKey={libraryRefresh}
      onLoad={handleLoadModel}
      onDeleted={() => setLibraryRefresh((v) => v + 1)}
    />
  );

  const isFleet = tab === "output" || tab === "library";

  return (
    <div className="h-dvh bg-[#0a0a0a] text-slate-100 flex overflow-hidden">

      {/* ═══════════════════════════════════════════════════════════════════
          LEFT SIDEBAR — desktop (md+)
      ══════════════════════════════════════════════════════════════════════ */}
      <aside className="hidden md:flex flex-col w-[220px] shrink-0 border-r border-white/[0.06] bg-[#0c0c0c]">

        {/* Logo */}
        <div className="h-14 px-4 flex items-center gap-3 border-b border-white/[0.06] shrink-0">
          <div className="size-7 rounded-lg bg-[#0A84FF]/15 border border-[#0A84FF]/20 flex items-center justify-center shrink-0">
            <Brain className="size-3.5 text-[#0A84FF]" />
          </div>
          <div>
            <div className="text-[13px] font-semibold tracking-tight text-white leading-none">
              AI Foundry
            </div>
            <div className="text-[10px] text-white/30 mt-0.5">browser-native ML</div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2.5 py-3 space-y-0.5 overflow-y-auto">
          <FoundryNavItem
            icon={Layers}
            label="Model Fleet"
            active={isFleet}
            onClick={() => setTab("output")}
          />
          <FoundryNavItem
            icon={SlidersHorizontal}
            label="Training Foundry"
            active={tab === "architect"}
            onClick={() => setTab("architect")}
          />
          <FoundryNavItem
            icon={MessageSquare}
            label="Chat Studio"
            active={tab === "brain"}
            onClick={() => setTab("brain")}
          />
        </nav>

        {/* Bottom controls */}
        <div className="px-2.5 pb-5 pt-3 border-t border-white/[0.06] space-y-2.5 shrink-0">
          <div className="px-0.5">
            <ModeToggle mode={mode} onChange={handleModeChange} />
          </div>
          <div className="flex gap-1.5">
            <button
              onClick={handleReset}
              title="Reset"
              className="flex-1 h-9 flex items-center justify-center rounded-lg border border-white/[0.07] text-white/40 hover:text-white/70 hover:bg-white/[0.05] transition-colors"
            >
              <RefreshCcw className="size-3.5" />
            </button>
            <button
              onClick={handleOpenSave}
              title="Save"
              className="flex-1 h-9 flex items-center justify-center rounded-lg border border-white/[0.07] text-white/40 hover:text-white/70 hover:bg-white/[0.05] transition-colors"
            >
              <Save className="size-3.5" />
            </button>
            <button
              onClick={handlePlay}
              title={isPlaying ? "Pause" : "Train"}
              className={`flex-1 h-9 flex items-center justify-center gap-1 rounded-lg text-[11px] font-semibold transition-colors ${
                isPlaying
                  ? "bg-[#0A84FF]/15 text-[#0A84FF] border border-[#0A84FF]/25 hover:bg-[#0A84FF]/25"
                  : "bg-[#0A84FF] text-white border border-transparent hover:bg-[#409CFF]"
              }`}
            >
              {isPlaying ? (
                <Pause className="size-3.5" />
              ) : (
                <Play className="size-3.5" />
              )}
              {isPlaying ? "Pause" : "Train"}
            </button>
          </div>
        </div>
      </aside>

      {/* ═══════════════════════════════════════════════════════════════════
          MAIN CONTENT AREA
      ══════════════════════════════════════════════════════════════════════ */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Top bar */}
        <header className="h-14 shrink-0 border-b border-white/[0.06] bg-[#0c0c0c]/80 backdrop-blur-xl flex items-center justify-between px-4 md:px-5 gap-3">
          <div className="flex items-center gap-3 min-w-0">
            {/* Mobile-only logo dot */}
            <div className="md:hidden size-7 rounded-lg bg-[#0A84FF]/15 border border-[#0A84FF]/20 flex items-center justify-center shrink-0">
              <Brain className="size-3.5 text-[#0A84FF]" />
            </div>
            <div className="min-w-0">
              <div className="text-[13px] font-semibold text-white tracking-tight truncate">
                {isFleet
                  ? "Model Fleet"
                  : tab === "architect"
                    ? "Training Foundry"
                    : "Chat Studio"}
              </div>
              <div className="text-[10px] text-white/35 tabular-nums truncate">
                {mode === "llm"
                  ? llmModelLabel
                  : `${layers.join("→")} · ${estimatedMLPParams}p`}
              </div>
            </div>
          </div>
          {/* Mobile-only controls */}
          <div className="md:hidden flex items-center gap-1.5 shrink-0">
            <ModeToggle mode={mode} onChange={handleModeChange} />
            <button
              onClick={handleReset}
              className="size-9 rounded-lg border border-white/[0.07] text-white/40 hover:text-white/70 hover:bg-white/[0.05] flex items-center justify-center transition-colors"
            >
              <RefreshCcw className="size-3.5" />
            </button>
            <button
              onClick={handleOpenSave}
              className="size-9 rounded-lg border border-white/[0.07] text-white/40 hover:text-white/70 hover:bg-white/[0.05] flex items-center justify-center transition-colors"
            >
              <Save className="size-3.5" />
            </button>
          </div>
        </header>

        {/* Scrollable view content */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-5xl mx-auto p-4 md:p-6 pb-24 md:pb-8 space-y-5">

            {/* ── MODEL FLEET ─────────────────────────────────────────────── */}
            {isFleet && (
              <div className="space-y-6">

                {/* Premium model cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">

                  {/* Active model card */}
                  <div className="rounded-2xl border border-white/[0.08] bg-[#141414] p-5 space-y-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-1.5">
                          <span
                            className={`size-1.5 rounded-full bg-[#30D158] ${isPlaying ? "animate-pulse" : ""}`}
                          />
                          <span className="text-[10px] font-medium uppercase tracking-[0.1em] text-[#30D158]">
                            {isPlaying ? "Training" : "Active"}
                          </span>
                        </div>
                        <div className="text-[15px] font-semibold text-white leading-tight">
                          {mode === "llm" ? "Tiny Text LM" : "Visual MLP"}
                        </div>
                        <div className="mt-1 font-mono text-[11px] text-white/40 truncate">
                          {mode === "llm"
                            ? llmModelLabel
                            : `${layers.join(" → ")} · ${activation}`}
                        </div>
                      </div>
                      <div className="size-10 rounded-xl bg-[#0A84FF]/10 border border-[#0A84FF]/15 flex items-center justify-center shrink-0">
                        <Brain className="size-5 text-[#0A84FF]" />
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <FleetMetric
                        label="Loss"
                        value={
                          mode === "llm"
                            ? textSnap.loss > 0 ? textSnap.loss.toFixed(3) : "—"
                            : snap?.loss ? snap.loss.toFixed(3) : "—"
                        }
                      />
                      <FleetMetric
                        label={mode === "llm" ? "Vocab" : "Params"}
                        value={
                          mode === "llm"
                            ? textSnap.vocabSize > 0 ? String(textSnap.vocabSize) : "—"
                            : String(estimatedMLPParams)
                        }
                      />
                      <FleetMetric
                        label="Epoch"
                        value={
                          mode === "llm"
                            ? textSnap.epoch > 0 ? String(textSnap.epoch) : "—"
                            : snap?.epoch ? String(snap.epoch) : "—"
                        }
                      />
                    </div>
                  </div>

                  {/* Coming soon — Vision Model */}
                  <div className="rounded-2xl border border-white/[0.04] bg-[#0e0e0e] p-5 space-y-4 select-none">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-1.5">
                          <span className="text-[10px] font-medium uppercase tracking-[0.1em] text-white/20">
                            Coming Soon
                          </span>
                        </div>
                        <div className="text-[15px] font-semibold text-white/30 leading-tight">
                          Vision Model
                        </div>
                        <div className="mt-1 text-[11px] text-white/20">
                          Image encoder · multi-modal
                        </div>
                      </div>
                      <div className="size-10 rounded-xl bg-white/[0.03] border border-white/[0.05] flex items-center justify-center shrink-0">
                        <Zap className="size-5 text-white/15" />
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-2">
                      <FleetMetric label="Loss" value="—" dim />
                      <FleetMetric label="Params" value="—" dim />
                      <FleetMetric label="Epoch" value="—" dim />
                    </div>
                  </div>
                </div>

                {/* Existing telemetry / sharing */}
                {OutputContent}

                {/* Saved model library */}
                {LibraryContent}
              </div>
            )}

            {/* ── TRAINING FOUNDRY ────────────────────────────────────────── */}
            {tab === "architect" && ArchitectContent}

            {/* ── CHAT STUDIO ─────────────────────────────────────────────── */}
            {tab === "brain" && BrainContent}

          </div>
        </main>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          MOBILE BOTTOM NAV
      ══════════════════════════════════════════════════════════════════════ */}
      <nav className="md:hidden fixed bottom-0 inset-x-0 z-20 border-t border-white/[0.06] bg-[#0c0c0c]/96 backdrop-blur-xl">
        <div className="grid grid-cols-4">
          <MobileNavButton
            icon={Layers}
            label="Fleet"
            active={isFleet}
            onClick={() => setTab("output")}
          />
          <MobileNavButton
            icon={SlidersHorizontal}
            label="Foundry"
            active={tab === "architect"}
            onClick={() => setTab("architect")}
          />
          <MobileNavButton
            icon={MessageSquare}
            label="Studio"
            active={tab === "brain"}
            onClick={() => setTab("brain")}
          />
          <button
            onClick={handlePlay}
            className={`flex flex-col items-center justify-center gap-1 py-2.5 min-h-[60px] transition-colors ${
              isPlaying
                ? "text-[#0A84FF]"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {isPlaying ? (
              <Pause className="size-5" />
            ) : (
              <Play className="size-5" />
            )}
            <span className="text-[10px] font-medium tracking-wide uppercase">
              {isPlaying ? "Pause" : "Train"}
            </span>
          </button>
        </div>
      </nav>

      <SaveModal
        open={saveOpen}
        defaultName={defaultSaveName(
          mode,
          mode === "mlp" ? dataset : llmModelLabel,
          mode === "mlp" ? snap?.epoch ?? 0 : textSnap.epoch,
          llmConfig.tokenization,
        )}
        modeLabel={
          mode === "mlp"
            ? "MLP Classifier"
            : llmConfig.tokenization === "word"
              ? "Word-level LM"
              : "Char-level LM"
        }
        hasModel={mode === "mlp" ? !!snap : textSnap.epoch > 0}
        onClose={() => setSaveOpen(false)}
        onSaveToLibrary={handleSaveToLibrary}
        onExportFile={handleExportFile}
      />

      <Toaster />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-apple-divider/10 bg-apple-card/30 px-2 py-3 text-center">
      <div className="text-[10px] uppercase tracking-wider text-apple-mid">
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

function FoundryNavItem({
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
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13px] font-medium transition-all ${
        active
          ? "bg-[#0A84FF]/12 text-[#0A84FF] border border-[#0A84FF]/15"
          : "text-white/45 hover:text-white/80 hover:bg-white/[0.05] border border-transparent"
      }`}
    >
      <Icon
        className={`size-4 shrink-0 ${active ? "text-[#0A84FF]" : "text-white/35"}`}
      />
      {label}
    </button>
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
      className={`flex flex-col items-center justify-center gap-1 py-2.5 min-h-[60px] transition-colors ${
        active ? "text-[#0A84FF]" : "text-slate-400 hover:text-slate-200"
      }`}
    >
      <Icon
        className={`size-5 ${active ? "drop-shadow-[0_0_8px_rgba(10,132,255,0.5)]" : ""}`}
      />
      <span className="text-[10px] font-medium tracking-wide uppercase">
        {label}
      </span>
    </button>
  );
}

function FleetMetric({
  label,
  value,
  dim = false,
}: {
  label: string;
  value: string;
  dim?: boolean;
}) {
  return (
    <div
      className={`rounded-xl border px-3 py-2.5 ${
        dim
          ? "border-white/[0.04] bg-white/[0.02]"
          : "border-white/[0.07] bg-white/[0.03]"
      }`}
    >
      <div
        className={`text-[9px] uppercase tracking-[0.1em] font-medium mb-1 ${
          dim ? "text-white/20" : "text-white/35"
        }`}
      >
        {label}
      </div>
      <div
        className={`text-sm font-semibold tabular-nums ${
          dim ? "text-white/20" : "text-white"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
