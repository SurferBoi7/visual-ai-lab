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
} from "lucide-react";
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

// Function-word filter for the OOV shield in `generateFromWorker`.
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

// Standard Levenshtein edit distance (two-row DP variant, O(min(m,n)) memory).
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
        curr[j - 1] + 1,
        prev[j] + 1,
        prev[j - 1] + cost,
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
  modeLabel: string,
  epoch: number,
  tokenization?: "char" | "word",
): string {
  const tag = tokenization === "word" ? "Word-LM" : "Char-LM";
  return `${tag} · ${modeLabel} · ep ${epoch}`;
}

export default function App() {
  const { toast } = useToast();
  const textWorkerRef = useRef<Worker | null>(null);
  const pendingGenRef = useRef<
    Map<string, (text: string) => void>
  >(new Map());

  // ---- LLM state ----
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

  const [tab, setTab] = useState<TabKey>("train");
  const [trainStep, setTrainStep] = useState<TrainStep>("fleet");
  const [newModelName, setNewModelName] = useState("");
  const [saveOpen, setSaveOpen] = useState(false);
  const [libraryRefresh, setLibraryRefresh] = useState(0);
  const [savedModels, setSavedModels] = useState<SavedModel[]>([]);

  // Load / refresh model library whenever libraryRefresh ticks
  useEffect(() => {
    getModels()
      .then(setSavedModels)
      .catch(() => {});
  }, [libraryRefresh]);

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
    if (llmPlaying) {
      textWorkerRef.current?.postMessage({
        type: "play",
        epochsPerSecond: v,
      });
    }
  };

  const handlePlay = () => {
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
  };

  const handleReset = () => {
    rebuildLLM();
    setMessages([]);
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
        //     using Levenshtein distance ≤ 1. This silently fixes "helo" →
        //     "hello" before the OOV check runs, so the shield doesn't fire
        //     on a mere typo.
        //
        //  2. OOV SHIELD: after correction, strip stop-words and check
        //     whether every remaining "content" word is still absent from
        //     the corpus vocab. If so, return the fallback reply immediately
        //     so the model doesn't hallucinate an answer it was never trained
        //     on.
        if (llmConfig.tokenization === "word") {
          const corpusVocab = new Set(
            tokenize(normalizePromptForLLM(llmConfig.corpus), "word"),
          );
          const inputWords = normalizedInput.split(" ").filter(Boolean);

          const AUTOCORRECT_MAX_DIST = 1;
          const correctedWords = inputWords.map((w) => {
            if (corpusVocab.has(w)) return w;
            let best = w;
            let bestDist = Infinity;
            for (const v of corpusVocab) {
              const d = levenshteinDistance(w, v);
              if (d < bestDist) {
                bestDist = d;
                best = v;
              }
            }
            return bestDist <= AUTOCORRECT_MAX_DIST ? best : w;
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

  const handleOpenSave = () => {
    if (textSnap.epoch === 0) {
      toast({
        title: "Train first",
        description: "Run a few epochs before saving.",
      });
      return;
    }
    setSaveOpen(true);
  };

  const handleExportFile = async (name: string) => {
    const w = await requestLLMExport();
    if (!w) {
      toast({ title: "Nothing to export", description: "Train first." });
      return;
    }
    downloadJSON(`${slug(name)}.json`, { kind: "char-lm", name, ...w });
    toast({ title: "File exported", description: `${name}.json saved.` });
  };

  const handleSaveToLibrary = async (name: string) => {
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
    setLibraryRefresh((v) => v + 1);
    toast({
      title: "Saved to Library",
      description: `${name} is in your model library.`,
    });
  };

  const handleLoadModel = (model: SavedModel) => {
    if (llmPlaying) {
      textWorkerRef.current?.postMessage({ type: "pause" });
      setLLMPlaying(false);
    }
    const w = model.weights as CharLMWeights;
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
    toast({
      title: "Model loaded",
      description: `${model.name} is now active.`,
    });
  };

  const handleDeleteModel = async (id: string) => {
    await deleteModel(id);
    setLibraryRefresh((v) => v + 1);
    toast({ title: "Deleted", description: "Model removed from library." });
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
      } else {
        toast({
          title: "Import failed",
          description: "Only Char-LM models are supported.",
        });
      }
    } catch {
      toast({
        title: "Import failed",
        description: "Could not parse the JSON file.",
      });
    }
  };

  // ---- Derived state ----
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

  const llmHasModel = textSnap.epoch > 0;
  const llmModeTag =
    llmConfig.tokenization === "word" ? "Word-LM" : "Char-LM";
  const llmModelLabel = `${llmModeTag} · ${llmConfig.contextSize}→${llmConfig.hiddenSize}→${textSnap.vocabSize}`;

  // LLM Architect + Train Speed card (used in Training Lab right panel)
  const TrainingConfig = (
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
      <div className="rounded-2xl border border-white/[0.06] bg-[#141414] p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Zap className="size-4 text-amber-400" />
          <span className="text-sm font-semibold text-slate-100">
            Train Speed
          </span>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-white/40">Epochs / sec</span>
            <span className="text-xs tabular-nums text-white/70">
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
      </div>
    </div>
  );

  const viewTitle =
    tab === "chat"
      ? "Chat Sandbox"
      : tab === "train"
        ? "Training Lab"
        : "Deploy Hub";

  const llmModels = savedModels.filter((m) => m.type !== "MLP");

  return (
    <div className="h-dvh bg-[#080808] text-slate-100 flex overflow-hidden">

      {/* ═══════════════════════════════════════════════════════════════════
          ULTRA-SLIM SIDEBAR — desktop (md+)
      ══════════════════════════════════════════════════════════════════════ */}
      <aside className="hidden md:flex flex-col w-16 shrink-0 border-r border-white/[0.05] bg-[#0a0a0a] items-center pt-4 pb-3">

        {/* Logo mark */}
        <div className="size-9 mb-5 rounded-xl bg-[#0A84FF]/15 border border-[#0A84FF]/20 flex items-center justify-center shrink-0">
          <Brain className="size-[18px] text-[#0A84FF]" />
        </div>

        {/* Primary nav */}
        <div className="flex flex-col gap-0.5 w-full px-2">
          <SlimNavItem
            icon={MessageSquare}
            label="Chat Sandbox"
            active={tab === "chat"}
            onClick={() => setTab("chat")}
          />
          <SlimNavItem
            icon={SlidersHorizontal}
            label="Training Lab"
            active={tab === "train"}
            onClick={() => setTab("train")}
          />
          <SlimNavItem
            icon={Zap}
            label="Deploy Hub"
            active={tab === "deploy"}
            onClick={() => setTab("deploy")}
          />
        </div>

        <div className="flex-1" />

        {/* Bottom actions */}
        <div className="flex flex-col gap-1.5 w-full px-2">
          <button
            onClick={handlePlay}
            title={llmPlaying ? "Pause training" : "Start training"}
            className={`w-full h-10 rounded-xl flex items-center justify-center transition-colors ${
              llmPlaying
                ? "bg-[#0A84FF]/15 text-[#0A84FF] border border-[#0A84FF]/25 hover:bg-[#0A84FF]/25"
                : "bg-[#0A84FF] text-white hover:bg-[#409CFF]"
            }`}
          >
            {llmPlaying ? (
              <Pause className="size-4" />
            ) : (
              <Play className="size-4" />
            )}
          </button>
          <button
            onClick={handleReset}
            title="Reset model"
            className="w-full h-9 rounded-xl flex items-center justify-center text-white/30 hover:text-white/70 hover:bg-white/[0.05] border border-white/[0.05] transition-colors"
          >
            <RefreshCcw className="size-3.5" />
          </button>
          <button
            onClick={handleOpenSave}
            title="Save model"
            className="w-full h-9 rounded-xl flex items-center justify-center text-white/30 hover:text-white/70 hover:bg-white/[0.05] border border-white/[0.05] transition-colors"
          >
            <Save className="size-3.5" />
          </button>
        </div>
      </aside>

      {/* ═══════════════════════════════════════════════════════════════════
          MAIN CONTENT AREA
      ══════════════════════════════════════════════════════════════════════ */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Top bar */}
        <header className="h-14 shrink-0 border-b border-white/[0.05] bg-[#0a0a0a]/80 backdrop-blur-xl flex items-center justify-between px-4 md:px-5 gap-3">
          <div className="flex items-center gap-3 min-w-0">
            {/* Mobile logo */}
            <div className="md:hidden size-7 rounded-lg bg-[#0A84FF]/15 border border-[#0A84FF]/20 flex items-center justify-center shrink-0">
              <Brain className="size-3.5 text-[#0A84FF]" />
            </div>
            <div className="min-w-0">
              <div className="text-[13px] font-semibold text-white tracking-tight truncate">
                {viewTitle}
              </div>
              <div className="text-[10px] text-white/30 tabular-nums truncate font-mono">
                {llmModelLabel}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {/* Desktop training telemetry pill */}
            {llmPlaying && (
              <div className="hidden md:flex items-center gap-3 mr-1">
                <div className="flex items-center gap-1.5">
                  <span className="relative flex size-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#0A84FF] opacity-75" />
                    <span className="relative inline-flex rounded-full size-2 bg-[#0A84FF]" />
                  </span>
                  <span className="text-[11px] font-medium text-[#0A84FF]">
                    Training
                  </span>
                </div>
                <div className="text-[11px] tabular-nums text-white/50">
                  Loss{" "}
                  <span className="text-white/80">
                    {textSnap.loss > 0 ? textSnap.loss.toFixed(4) : "—"}
                  </span>
                </div>
                <div className="text-[11px] tabular-nums text-white/50">
                  Ep{" "}
                  <span className="text-white/80">
                    {textSnap.epoch > 0 ? textSnap.epoch : "—"}
                  </span>
                </div>
              </div>
            )}
            {/* Mobile controls */}
            <div className="md:hidden flex items-center gap-1.5">
              <button
                onClick={handlePlay}
                className={`h-9 px-3 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors ${
                  llmPlaying
                    ? "bg-[#0A84FF]/15 text-[#0A84FF] border border-[#0A84FF]/25"
                    : "bg-[#0A84FF] text-white"
                }`}
              >
                {llmPlaying ? (
                  <Pause className="size-3.5" />
                ) : (
                  <Play className="size-3.5" />
                )}
                {llmPlaying ? "Pause" : "Train"}
              </button>
              <button
                onClick={handleReset}
                className="size-9 rounded-lg border border-white/[0.06] text-white/40 hover:text-white/70 hover:bg-white/[0.05] flex items-center justify-center transition-colors"
              >
                <RefreshCcw className="size-3.5" />
              </button>
              <button
                onClick={handleOpenSave}
                className="size-9 rounded-lg border border-white/[0.06] text-white/40 hover:text-white/70 hover:bg-white/[0.05] flex items-center justify-center transition-colors"
              >
                <Save className="size-3.5" />
              </button>
            </div>
          </div>
        </header>

        {/* ── View router ──────────────────────────────────────────────────── */}
        <div className="flex-1 overflow-hidden">

          {/* ──────────────────────────────────────────────────────────────────
              TRAINING LAB — Step-based workflow
          ────────────────────────────────────────────────────────────────── */}
          {tab === "train" && (
            <div className="h-full overflow-hidden">

              {/* ─── Fleet Dashboard ──────────────────────────────────────── */}
              {trainStep === "fleet" && (
                <div className="h-full overflow-y-auto">
                  <div className="max-w-3xl mx-auto py-8 px-5">
                    <div className="mb-7">
                      <h2 className="text-[17px] font-semibold text-white/88 tracking-tight">
                        Model Fleet
                      </h2>
                      <p className="text-[12px] text-white/30 mt-0.5 leading-relaxed">
                        Select a model to continue training, or create a new base model.
                      </p>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {llmModels.map((m) => (
                        <div
                          key={m.id}
                          className="group relative rounded-2xl border border-white/[0.06] bg-[#141414] hover:bg-[#181818] hover:border-white/[0.1] transition-all cursor-pointer overflow-hidden"
                          onClick={() => { handleLoadModel(m); setTrainStep("training"); }}
                        >
                          <div className="p-4">
                            <div className="flex items-start justify-between mb-3">
                              <div className="size-9 rounded-xl bg-[#0A84FF]/10 border border-[#0A84FF]/15 flex items-center justify-center">
                                <Brain className="size-4 text-[#0A84FF]/70" />
                              </div>
                              <span className="text-[9px] font-mono text-white/20 tracking-wider uppercase bg-white/[0.04] px-1.5 py-0.5 rounded-md">
                                {m.type}
                              </span>
                            </div>
                            <div className="text-[13px] font-semibold text-white/82 mb-1 truncate leading-snug">
                              {m.name}
                            </div>
                            <div className="text-[10px] font-mono text-white/30 tabular-nums">
                              ep {m.epoch} · loss {m.loss.toFixed(3)}
                            </div>
                            <div className="text-[10px] text-white/18 mt-0.5">
                              {new Date(m.date).toLocaleDateString()}
                            </div>
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDeleteModel(m.id); }}
                            className="absolute top-3 right-3 size-6 rounded-lg flex items-center justify-center text-white/15 hover:text-red-400 hover:bg-red-400/10 opacity-0 group-hover:opacity-100 transition-all"
                          >
                            <Trash2 className="size-3" />
                          </button>
                        </div>
                      ))}
                      <button
                        onClick={() => setTrainStep("setup")}
                        className="rounded-2xl border-2 border-dashed border-white/[0.07] hover:border-[#0A84FF]/30 hover:bg-[#0A84FF]/[0.03] transition-all flex flex-col items-center justify-center gap-2.5 min-h-[140px] text-white/25 hover:text-[#0A84FF]/60"
                      >
                        <Plus className="size-6" />
                        <span className="text-[12px] font-medium">New Base Model</span>
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* ─── Configuration Setup ─────────────────────────────────── */}
              {trainStep === "setup" && (
                <div className="h-full overflow-y-auto">
                  <div className="max-w-md mx-auto py-8 px-5 space-y-6">
                    <button
                      onClick={() => setTrainStep("fleet")}
                      className="flex items-center gap-1.5 text-white/30 hover:text-white/60 text-[12px] transition-colors"
                    >
                      <ArrowLeft className="size-4" />
                      Fleet
                    </button>
                    <div>
                      <h2 className="text-[17px] font-semibold text-white/88 tracking-tight">
                        New Model Configuration
                      </h2>
                      <p className="text-[12px] text-white/30 mt-0.5 leading-relaxed">
                        Set the base architecture for your new language model.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <label className="text-[11px] text-white/35 font-medium">Model Name</label>
                      <input
                        type="text"
                        value={newModelName}
                        onChange={(e) => setNewModelName(e.target.value)}
                        placeholder="My Language Model"
                        className="w-full h-11 rounded-xl bg-[#121212] border border-white/[0.06] px-4 text-[13px] text-white/80 placeholder:text-white/20 focus:outline-none focus:border-[#0A84FF]/40 transition-colors"
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-[11px] text-white/35 font-medium">Tokenization</label>
                      <div className="grid grid-cols-2 gap-1.5 bg-[#0a0a0a] rounded-xl border border-white/[0.05] p-1.5">
                        <button
                          onClick={() => setLLMConfig((c) => ({ ...c, tokenization: "char" }))}
                          className={`h-9 rounded-lg text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-all ${llmConfig.tokenization === "char" ? "bg-[#0A84FF]/15 text-[#0A84FF] border border-[#0A84FF]/20" : "text-white/30 hover:text-white/55"}`}
                        >
                          Char-Level
                        </button>
                        <button
                          onClick={() => setLLMConfig((c) => ({ ...c, tokenization: "word" }))}
                          className={`h-9 rounded-lg text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-all ${llmConfig.tokenization === "word" ? "bg-[#30D158]/12 text-[#30D158] border border-[#30D158]/20" : "text-white/30 hover:text-white/55"}`}
                        >
                          Word-Level
                        </button>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <label className="text-[11px] text-white/35 font-medium">Context Window</label>
                        <span className="text-[11px] tabular-nums text-white/60 font-mono">{llmConfig.contextSize}</span>
                      </div>
                      <Slider min={1} max={20} step={1} value={[llmConfig.contextSize]} onValueChange={([v]) => setLLMConfig((c) => ({ ...c, contextSize: v }))} className="py-2" />
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <label className="text-[11px] text-white/35 font-medium">Hidden Neurons</label>
                        <span className="text-[11px] tabular-nums text-white/60 font-mono">{llmConfig.hiddenSize}</span>
                      </div>
                      <Slider min={4} max={512} step={4} value={[llmConfig.hiddenSize]} onValueChange={([v]) => setLLMConfig((c) => ({ ...c, hiddenSize: v }))} className="py-2" />
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <label className="text-[11px] text-white/35 font-medium">Learning Rate</label>
                        <span className="text-[11px] tabular-nums text-white/60 font-mono">{llmConfig.learningRate.toFixed(3)}</span>
                      </div>
                      <Slider min={0.005} max={0.5} step={0.005} value={[llmConfig.learningRate]} onValueChange={([v]) => setLLMConfig((c) => ({ ...c, learningRate: v }))} className="py-2" />
                    </div>
                    <div className="rounded-xl bg-[#0a0a0a] border border-white/[0.05] px-4 py-3 flex items-center justify-between">
                      <span className="text-[11px] text-white/30">Est. parameters</span>
                      <span className={`text-[11px] tabular-nums font-mono ${estimatedLLMParams > MAX_PARAMS_LLM ? "text-red-400" : "text-white/60"}`}>
                        {estimatedLLMParams.toLocaleString()}
                      </span>
                    </div>
                    <button
                      onClick={() => { rebuildLLM(); setTrainStep("training"); setMessages([]); }}
                      disabled={estimatedLLMParams > MAX_PARAMS_LLM}
                      className="w-full h-12 rounded-xl bg-[#0A84FF] hover:bg-[#409CFF] text-white font-semibold text-[13px] flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <Play className="size-4" />
                      Initialize &amp; Begin Training
                    </button>
                  </div>
                </div>
              )}

              {/* ─── Active Training Dashboard ───────────────────────────── */}
              {trainStep === "training" && (
                <div className="h-full flex overflow-hidden">
                  {/* Left panel — model list */}
                  <div className="hidden sm:flex flex-col w-56 shrink-0 border-r border-white/[0.05] bg-[#090909] overflow-hidden">
                    <div className="px-2.5 pt-3 pb-2 border-b border-white/[0.05] shrink-0 space-y-1.5">
                      <button
                        onClick={() => setTrainStep("fleet")}
                        className="w-full h-8 rounded-xl text-white/30 hover:text-white/60 text-[11px] flex items-center gap-2 px-3 hover:bg-white/[0.03] transition-colors"
                      >
                        <ArrowLeft className="size-3.5" />
                        Model Fleet
                      </button>
                      <button
                        onClick={() => setTrainStep("setup")}
                        className="w-full h-9 rounded-xl border border-white/[0.07] hover:border-white/[0.13] bg-white/[0.02] hover:bg-white/[0.05] text-white/50 hover:text-white/80 text-xs font-medium flex items-center justify-center gap-2 transition-all"
                      >
                        <Plus className="size-3.5" />
                        New Base Model
                      </button>
                    </div>
                    <div className="flex-1 overflow-y-auto py-1.5">
                      {llmModels.length === 0 ? (
                        <div className="text-center py-10 px-4 text-white/20 text-[11px] leading-relaxed">
                          No saved models yet.<br />Train and save to build<br />your library.
                        </div>
                      ) : (
                        llmModels.map((m) => (
                          <div key={m.id} className="group relative mx-1.5 mb-px">
                            <button
                              onClick={() => handleLoadModel(m)}
                              className="w-full text-left px-3 py-2.5 rounded-xl hover:bg-white/[0.05] transition-colors"
                            >
                              <div className="text-[12px] font-medium text-white/75 truncate pr-7">
                                {m.name}
                              </div>
                              <div className="text-[10px] text-white/25 mt-0.5 tabular-nums font-mono">
                                ep {m.epoch} · {m.loss.toFixed(3)}
                              </div>
                            </button>
                            <button
                              onClick={(e) => { e.stopPropagation(); handleDeleteModel(m.id); }}
                              className="absolute right-2 top-1/2 -translate-y-1/2 size-6 rounded-lg flex items-center justify-center text-white/15 hover:text-red-400 hover:bg-red-400/10 opacity-0 group-hover:opacity-100 transition-all"
                            >
                              <Trash2 className="size-3" />
                            </button>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  {/* Right panel — config + pulsing training overlay */}
                  <div className="flex-1 relative overflow-y-auto">

                    {/* Training-in-progress overlay — pulsing neural animation */}
                    {llmPlaying && (
                      <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-8 pointer-events-none">
                        <div className="relative flex items-center justify-center">
                          <div
                            className="absolute rounded-full border border-[#0A84FF]/25"
                            style={{ width: 160, height: 160, animation: "backprop-ring 2.4s ease-out infinite 0s" }}
                          />
                          <div
                            className="absolute rounded-full border border-[#0A84FF]/18"
                            style={{ width: 160, height: 160, animation: "backprop-ring 2.4s ease-out infinite 0.8s" }}
                          />
                          <div
                            className="absolute rounded-full border border-[#0A84FF]/12"
                            style={{ width: 160, height: 160, animation: "backprop-ring 2.4s ease-out infinite 1.6s" }}
                          />
                          <div
                            className="size-20 rounded-full bg-[#0A84FF]/8 border border-[#0A84FF]/18 flex items-center justify-center"
                            style={{ animation: "backprop-orb 2s ease-in-out infinite" }}
                          >
                            <Brain className="size-8 text-[#0A84FF]/60" />
                          </div>
                        </div>
                        <div className="bg-[#0c0c0c]/90 backdrop-blur-xl rounded-2xl border border-white/[0.07] px-8 py-5 flex items-center gap-6">
                          <div className="text-center">
                            <div className="text-[9px] uppercase tracking-[0.12em] text-white/25 mb-1.5 font-medium">Loss</div>
                            <div className="text-2xl font-bold tabular-nums text-white">
                              {textSnap.loss > 0 ? textSnap.loss.toFixed(4) : "—"}
                            </div>
                          </div>
                          <div className="w-px h-10 bg-white/[0.07]" />
                          <div className="text-center">
                            <div className="text-[9px] uppercase tracking-[0.12em] text-white/25 mb-1.5 font-medium">Epoch</div>
                            <div className="text-2xl font-bold tabular-nums text-white">
                              {textSnap.epoch > 0 ? textSnap.epoch : "—"}
                            </div>
                          </div>
                          <div className="w-px h-10 bg-white/[0.07]" />
                          <div className="text-center">
                            <div className="text-[9px] uppercase tracking-[0.12em] text-white/25 mb-1.5 font-medium">tok/s</div>
                            <div className="text-2xl font-bold tabular-nums text-white">
                              {textSnap.tokensPerSecond > 0
                                ? textSnap.tokensPerSecond > 1000
                                  ? `${(textSnap.tokensPerSecond / 1000).toFixed(1)}k`
                                  : String(Math.round(textSnap.tokensPerSecond))
                                : "—"}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Config + stats — dimmed while training */}
                    <div
                      className={`p-4 md:p-5 pb-16 space-y-5 transition-opacity duration-300 ${
                        llmPlaying ? "opacity-20 pointer-events-none select-none" : "opacity-100"
                      }`}
                    >
                      {llmHasModel && (
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
                      )}
                      {TrainingConfig}
                      <SharingHub
                        mode="llm"
                        onDownload={handleOpenSave}
                        hasModel={llmHasModel}
                        onImport={handleImportModel}
                      />
                    </div>
                  </div>
                </div>
              )}

            </div>
          )}

          {/* ──────────────────────────────────────────────────────────────────
              CHAT SANDBOX
          ────────────────────────────────────────────────────────────────── */}
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

          {/* ──────────────────────────────────────────────────────────────────
              DEPLOY HUB
          ────────────────────────────────────────────────────────────────── */}
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
      <nav className="md:hidden fixed bottom-0 inset-x-0 z-20 border-t border-white/[0.05] bg-[#0a0a0a]/96 backdrop-blur-xl">
        <div className="grid grid-cols-4">
          <MobileNavButton
            icon={MessageSquare}
            label="Chat"
            active={tab === "chat"}
            onClick={() => setTab("chat")}
          />
          <MobileNavButton
            icon={SlidersHorizontal}
            label="Training"
            active={tab === "train"}
            onClick={() => setTab("train")}
          />
          <MobileNavButton
            icon={Zap}
            label="Deploy"
            active={tab === "deploy"}
            onClick={() => setTab("deploy")}
          />
          <button
            onClick={handlePlay}
            className={`flex flex-col items-center justify-center gap-1 py-2.5 min-h-[60px] transition-colors ${
              llmPlaying
                ? "text-[#0A84FF]"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {llmPlaying ? (
              <Pause className="size-5" />
            ) : (
              <Play className="size-5" />
            )}
            <span className="text-[10px] font-medium tracking-wide uppercase">
              {llmPlaying ? "Pause" : "Train"}
            </span>
          </button>
        </div>
      </nav>

      <SaveModal
        open={saveOpen}
        defaultName={defaultSaveName(
          llmModelLabel,
          textSnap.epoch,
          llmConfig.tokenization,
        )}
        modeLabel={
          llmConfig.tokenization === "word" ? "Word-level LM" : "Char-level LM"
        }
        hasModel={llmHasModel}
        onClose={() => setSaveOpen(false)}
        onSaveToLibrary={handleSaveToLibrary}
        onExportFile={handleExportFile}
      />

      <Toaster />
    </div>
  );
}

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
          ? "bg-[#0A84FF]/12 text-[#0A84FF] border border-[#0A84FF]/15"
          : "text-white/30 hover:text-white/65 hover:bg-white/[0.05] border border-transparent"
      }`}
    >
      <Icon className="size-[18px] shrink-0" />
      {/* Hover tooltip */}
      <span className="absolute left-full ml-3 z-50 whitespace-nowrap bg-[#1c1c1c] border border-white/[0.08] text-white/75 text-[11px] font-medium px-2.5 py-1.5 rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity shadow-xl">
        {label}
      </span>
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
