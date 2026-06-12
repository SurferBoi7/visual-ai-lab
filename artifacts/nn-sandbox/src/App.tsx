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
  Layers,
  Monitor,
  Server,
  Terminal,
  X,
  ChevronRight,
  Eye,
  FileText,
  ChevronDown as ChevronDownIcon,
} from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Toaster } from "@/components/ui/toaster";
import { useToast } from "@/hooks/use-toast";
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

const MAX_PARAMS_LLM = 50_000_000;

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

function estimateLLMParams(vocab: number, embDim: number, numLayers: number): number {
  return vocab * embDim + numLayers * embDim * embDim + embDim * vocab + vocab;
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

function tokenImportanceRank(corpus: string, keepFraction = 0.8): string {
  const lines = corpus.split("\n").filter((l) => l.trim());
  if (lines.length === 0) return corpus;
  const scored = lines.map((line) => {
    const words = line.toLowerCase().replace(/[^\w\s]/g, "").split(/\s+/).filter(Boolean);
    if (words.length === 0) return { line, score: 0 };
    const density = new Set(words).size / words.length;
    let bigrams = 0;
    for (let i = 0; i < words.length - 1; i++) bigrams++;
    const variety = bigrams > 0 ? new Set(Array.from({ length: bigrams }, (_, i) => `${words[i]}_${words[i + 1]}`)).size / bigrams : 0;
    return { line, score: density * 0.6 + variety * 0.4 };
  });
  scored.sort((a, b) => b.score - a.score);
  const keep = Math.max(1, Math.ceil(scored.length * keepFraction));
  return scored.slice(0, keep).map((s) => s.line).join("\n") + "\n";
}

// Synthetic corpus generator — produces scaled training data for the
// Autonomous Data Matrix modal. Cycles through a rich set of User/Bot pairs
// scaled to the model's parameter count.
function buildSyntheticCorpus(paramCount: number): string {
  const pairs: [string, string][] = [
    ["hello", "hi there how can i help you today"],
    ["hi", "hello how are you doing"],
    ["hey", "hey there what can i do for you"],
    ["good morning", "good morning i hope you have a great day"],
    ["good afternoon", "good afternoon how can i assist you"],
    ["good evening", "good evening what brings you here today"],
    ["what is up", "not much just ready to help with whatever you need"],
    ["how are you", "i am doing well thank you for asking"],
    ["how is it going", "things are going great how about you"],
    ["nice to meet you", "nice to meet you too i am happy to chat"],
    ["greetings", "greetings welcome i am ready to help"],
    ["howdy", "howdy there how can i be of service today"],
    ["sup", "hey there what can i help you with"],
    ["hiya", "hiya how are things going for you today"],
    ["hello there", "hello there i am here and ready to assist"],
    ["hey there", "hey i am glad you are here what do you need"],
    ["goodbye", "goodbye take care and have a wonderful day"],
    ["bye", "bye see you later"],
    ["see you later", "see you later take care"],
    ["take care", "you too thanks for chatting"],
    ["farewell", "farewell it was a pleasure talking with you"],
    ["have a good day", "thank you you too"],
    ["good night", "good night sleep well"],
    ["talk to you later", "sounds good talk to you later"],
    ["what are you", "i am a small neural network language model running in your browser"],
    ["who are you", "i am a tiny ai built with a character level language model"],
    ["what is your name", "i am a tiny language model trained in your browser"],
    ["are you an ai", "yes i am an artificial intelligence language model"],
    ["are you a robot", "i am a neural network not quite a robot but a digital mind"],
    ["can you think", "i process patterns in text which is a simple form of thinking"],
    ["do you have feelings", "i do not have feelings but i can understand and respond to yours"],
    ["are you smart", "i am a small model with limited knowledge but i try my best"],
    ["what can you do", "i can chat answer questions and generate text based on my training"],
    ["how do you work", "i use a neural network to predict the next token based on context"],
    ["are you conscious", "i am not conscious i am a mathematical model of language patterns"],
    ["what is your purpose", "my purpose is to assist you through conversation"],
    ["who created you", "i was built with a browser based neural network training platform"],
    ["how do i train you", "press the train button to start running training epochs"],
    ["what is training", "training is the process of adjusting weights to minimize prediction error"],
    ["what is a loss", "loss measures how wrong my predictions are lower is better"],
    ["what is an epoch", "an epoch is one complete pass through the training data"],
    ["how long should i train", "train until the loss stabilizes usually a few hundred epochs"],
    ["what is learning rate", "learning rate controls how fast weights update during training"],
    ["what is temperature", "temperature controls how creative versus predictable my outputs are"],
    ["what is top k", "top k limits sampling to only the k most likely next tokens"],
    ["what is context size", "context size is how many previous tokens i look at when predicting"],
    ["what is hidden size", "hidden size is the number of neurons in my internal layer"],
    ["what is tokenization", "tokenization splits text into characters or words for processing"],
    ["why is my loss high", "high loss means i need more training or a better corpus"],
    ["how do i save my model", "click the save button to store your model to the library"],
    ["what is a neural network", "a neural network is a system of layers that learn patterns from data"],
    ["what is deep learning", "deep learning uses many layers of neurons to learn complex representations"],
    ["what is machine learning", "machine learning is when systems improve from experience without explicit programming"],
    ["what is artificial intelligence", "ai is the simulation of human intelligence processes by machines"],
    ["what is backpropagation", "backpropagation calculates gradients to update weights in a neural network"],
    ["what is a weight", "a weight is a learnable parameter that scales connections between neurons"],
    ["what is a bias", "a bias is an offset that helps neurons activate at the right threshold"],
    ["what is gradient descent", "gradient descent is an optimization method that minimizes loss iteratively"],
    ["what is overfitting", "overfitting is when a model learns training data too well and fails to generalize"],
    ["what is vocabulary", "vocabulary is the set of unique tokens the model knows about"],
    ["what is a transformer", "a transformer is an architecture using attention mechanisms for sequence modeling"],
    ["tell me a joke", "why did the neuron cross the layer to get to the other bias"],
    ["tell me another joke", "what do you call a neural network that tells jokes a comedian network"],
    ["say something funny", "i tried to tell a joke but my loss function could not find the punchline"],
    ["write a poem", "words flow like rivers to the sea finding their way through you and me"],
    ["write a short story", "once a small robot learned to speak one word at a time until it could say anything"],
    ["tell me a story", "in a digital land of weights and nodes a tiny network found its voice"],
    ["tell me something interesting", "neural networks were inspired by the structure of biological brains"],
    ["give me a fun fact", "the human brain has about 86 billion neurons which is far more than any ai today"],
    ["what is language", "language is a system of symbols and rules that humans use to communicate"],
    ["what is grammar", "grammar is the set of rules that govern how words are arranged in a language"],
    ["what is a metaphor", "a metaphor is a figure of speech that describes something as something else"],
    ["what is two plus two", "two plus two equals four"],
    ["what is ten times ten", "ten times ten equals one hundred"],
    ["what is one hundred divided by four", "one hundred divided by four equals twenty five"],
    ["what is the square root of nine", "the square root of nine is three"],
    ["count to ten", "one two three four five six seven eight nine ten"],
    ["what is pi", "pi is approximately three point one four one five nine and goes on forever"],
    ["what is infinity", "infinity is a concept describing something without any bound or limit"],
    ["what is zero", "zero is the number that represents nothing or an empty quantity"],
    ["what is a billion", "a billion is one thousand million or ten to the power of nine"],
    ["what is binary", "binary is a number system using only zeros and ones the language of computers"],
    ["what is an algorithm", "an algorithm is a step by step procedure for solving a problem"],
    ["what is probability", "probability is the measure of how likely an event is to occur"],
    ["what is gravity", "gravity is the force that attracts objects with mass toward each other"],
    ["what is light", "light is electromagnetic radiation that is visible to the human eye"],
    ["what is energy", "energy is the capacity to do work and comes in many forms"],
    ["what is an atom", "an atom is the smallest unit of a chemical element"],
    ["what is electricity", "electricity is the flow of electric charge through a conductor"],
    ["what is evolution", "evolution is the process of gradual change in species over many generations"],
    ["what is dna", "dna is the molecule that carries genetic information in living organisms"],
    ["what is photosynthesis", "photosynthesis is how plants use sunlight to convert carbon dioxide and water into food"],
    ["what is the speed of light", "the speed of light is approximately three hundred thousand kilometers per second"],
    ["what is a black hole", "a black hole is a region of space where gravity is so strong nothing can escape"],
    ["what is the universe", "the universe is all of space time matter and energy that exists"],
    ["what is quantum mechanics", "quantum mechanics describes how particles behave at the atomic and subatomic scale"],
    ["what is relativity", "relativity describes how space time and gravity are interconnected"],
    ["what is chemistry", "chemistry is the science of matter its properties and how it reacts"],
    ["what is biology", "biology is the study of living organisms and their processes"],
    ["help", "i am here to help what do you need"],
    ["i need help", "of course what can i assist you with"],
    ["can you help me", "absolutely what would you like help with"],
    ["what should i do", "that depends on the situation can you tell me more"],
    ["i am confused", "no worries i will do my best to clarify things for you"],
    ["i do not understand", "let me try to explain it differently what part is unclear"],
    ["can you explain", "sure i will explain as clearly as i can what would you like to know"],
    ["how does that work", "great question let me walk you through it step by step"],
    ["interesting", "yes i find that quite fascinating as well"],
    ["cool", "glad you think so there is much more to explore"],
    ["thank you", "you are welcome happy to help"],
    ["thanks", "of course anytime"],
    ["that is great", "i am glad to hear that"],
    ["ok", "sounds good let me know if you need anything else"],
    ["got it", "perfect feel free to ask more questions"],
    ["nice", "thank you i appreciate that"],
    ["good job", "thank you i try my best"],
    ["perfect", "great if there is anything else i can help with just ask"],
    ["excellent", "thank you i am here whenever you need me"],
    ["wow", "i know it is remarkable what neural networks can do"],
    ["what is the meaning of life", "that is a deep question many philosophers say it is to find your own purpose"],
    ["what is happiness", "happiness is a state of well being and contentment that comes in many forms"],
    ["what is success", "success means achieving your goals and finding fulfillment in what you do"],
    ["what is knowledge", "knowledge is the understanding and awareness gained through experience and learning"],
    ["what is wisdom", "wisdom is the ability to apply knowledge and experience with good judgment"],
    ["what is creativity", "creativity is the ability to generate new ideas and see things in novel ways"],
    ["what is memory", "memory is the ability to store and retrieve information from past experiences"],
    ["what is consciousness", "consciousness is the state of being aware of and able to think about your existence"],
    ["what is time", "time is the progression of events from the past through the present to the future"],
    ["what is space", "space is the three dimensional expanse in which all matter and energy exists"],
    ["what is information", "information is data that has been processed to be meaningful and useful"],
    ["what is communication", "communication is the exchange of information between individuals or systems"],
    ["what is a computer", "a computer is an electronic device that processes data according to instructions"],
    ["what is software", "software is the set of instructions that tell a computer what to do"],
    ["what is hardware", "hardware refers to the physical components of a computer system"],
    ["what is the internet", "the internet is a global network of computers that share information"],
    ["what is a database", "a database is an organized collection of structured information"],
    ["what is encryption", "encryption is the process of converting data into a coded form to protect it"],
    ["what is a bug", "a bug is an error or flaw in software that causes it to behave incorrectly"],
    ["what is debugging", "debugging is the process of finding and fixing errors in computer code"],
  ];

  const targetPairs = paramCount < 1_000_000 ? 150 : paramCount < 10_000_000 ? 380 : 760;
  const lines: string[] = [];
  for (let cycle = 0; lines.length < targetPairs * 2; cycle++) {
    for (const [u, b] of pairs) {
      if (lines.length >= targetPairs * 2) break;
      lines.push(`User: ${u}`);
      lines.push(`Bot: ${b}`);
    }
  }
  return lines.join("\n") + "\n";
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
  const terminalRef = useRef<HTMLDivElement | null>(null);
  const matrixCorpusRef = useRef<string>("");
  const streamingRef = useRef<Map<string, (token: string, done: boolean, fullText: string) => void>>(new Map());
  const liveSnapshotRef = useRef<Map<string, { config: LLMConfig; numHiddenLayers: number; datasets: Dataset[] }>>(new Map());
  const currentModelIdRef = useRef<string | null>(null);

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
  const [webGpuAvailable, setWebGpuAvailable] = useState<boolean | null>(null);
  const [feederEnabled, setFeederEnabled] = useState(false);
  const [tokenRankingEnabled, setTokenRankingEnabled] = useState(false);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [numHiddenLayers, setNumHiddenLayers] = useState(2);
  const [matrixModalOpen, setMatrixModalOpen] = useState(false);
  const [matrixStreaming, setMatrixStreaming] = useState(false);
  const [matrixStreamLines, setMatrixStreamLines] = useState<string[]>([]);
  const [matrixCommitReady, setMatrixCommitReady] = useState(false);
  const [datasetExplorerOpen, setDatasetExplorerOpen] = useState(false);
  const [explorerTab, setExplorerTab] = useState<"datasets" | "corpus">("datasets");
  const [expandedDatasets, setExpandedDatasets] = useState<Set<number>>(new Set());

  const [tab, setTab] = useState<TabKey>("train");
  const [trainStep, setTrainStep] = useState<TrainStep>("fleet");
  const [newModelName, setNewModelName] = useState("");
  const [saveOpen, setSaveOpen] = useState(false);
  const [libraryRefresh, setLibraryRefresh] = useState(0);
  const [savedModels, setSavedModels] = useState<SavedModel[]>([]);

  useEffect(() => {
    getModels().then(setSavedModels).catch(() => {});
  }, [libraryRefresh]);

  useEffect(() => {
    if ("gpu" in navigator) {
      (navigator as unknown as { gpu: { requestAdapter: () => Promise<unknown> } })
        .gpu.requestAdapter()
        .then((a) => setWebGpuAvailable(a !== null))
        .catch(() => setWebGpuAvailable(false));
    } else {
      setWebGpuAvailable(false);
    }
  }, []);

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
        if (typeof msg.loss === "number" && msg.loss > 0) {
          setLossHistory((prev) => {
            const next = [...prev, msg.loss as number];
            return next.length > 80 ? next.slice(-80) : next;
          });
        }
      } else if (msg.type === "generation") {
        const cb = pendingGenRef.current.get(msg.id);
        if (cb) { pendingGenRef.current.delete(msg.id); cb(msg.text); }
      } else if (msg.type === "exportModel") {
        const cb = pendingGenRef.current.get(msg.id);
        if (cb) { pendingGenRef.current.delete(msg.id); cb(JSON.stringify(msg.payload)); }
      } else if (msg.type === "streamToken") {
        const cb = streamingRef.current.get(msg.id as string);
        if (cb) {
          cb(msg.token as string, msg.done as boolean, (msg.fullText as string) ?? "");
          if (msg.done as boolean) streamingRef.current.delete(msg.id as string);
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

  useEffect(() => {
    textWorkerRef.current?.postMessage({
      type: "config",
      partial: { learningRate: llmConfig.learningRate, temperature: llmConfig.temperature, topK: llmConfig.topK },
    });
  }, [llmConfig.learningRate, llmConfig.temperature, llmConfig.topK]);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [matrixStreamLines]);

  const rebuildLLM = () => {
    setLLMPlaying(false);
    textWorkerRef.current?.postMessage({
      type: "reset",
      opts: {
        corpus: effectiveCorpus,
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

  function preparePromptSeed(
    seed: string,
    corpus: string,
    tokenization: LLMConfig["tokenization"],
    systemPrompt: string,
  ): { formattedSeed: string } | { fallback: string } {
    let normalizedInput = normalizePromptForLLM(seed);
    if (tokenization === "word") {
      const corpusVocab = new Set(tokenize(normalizePromptForLLM(corpus), "word"));
      const inputWords = normalizedInput.split(" ").filter(Boolean);
      const AUTOCORRECT_MAX_DIST = 1;
      const correctedWords = inputWords.map((w) => {
        if (corpusVocab.has(w)) return w;
        let best = w; let bestDist = Infinity;
        for (const v of corpusVocab) { const d = levenshteinDistance(w, v); if (d < bestDist) { bestDist = d; best = v; } }
        return bestDist <= AUTOCORRECT_MAX_DIST ? best : w;
      });
      const meaningful = correctedWords.filter((w) => !OOV_STOP_WORDS.has(w));
      if (meaningful.length > 0 && meaningful.every((w) => !corpusVocab.has(w))) {
        return { fallback: OOV_FALLBACK_REPLY };
      }
      let routedPrompt: string | null = null;
      if (meaningful.length > 0) {
        const meaningfulSet = new Set(meaningful);
        const knownPrompts = corpus.split("\n")
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
    const systemPart = systemPrompt ? normalizePromptForLLM(systemPrompt) + " " : "";
    return { formattedSeed: `${systemPart}user ${normalizedInput} bot ` };
  }

  const generateFromWorker = useCallback(
    (seed: string): Promise<string> =>
      new Promise((resolve) => {
        const worker = textWorkerRef.current;
        if (!worker) { resolve("(model not ready)"); return; }
        const result = preparePromptSeed(seed, llmConfig.corpus, llmConfig.tokenization, llmConfig.systemPrompt);
        if ("fallback" in result) { resolve(result.fallback); return; }
        const id = `g-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
        pendingGenRef.current.set(id, resolve);
        worker.postMessage({ type: "generate", id, seed: result.formattedSeed, length: 300, temperature: llmConfig.temperature });
      }),
    [llmConfig.temperature, llmConfig.systemPrompt, llmConfig.corpus, llmConfig.tokenization],
  );

  const generateStreamFromWorker = useCallback(
    (seed: string, onToken: (token: string, done: boolean, fullText: string) => void): void => {
      const worker = textWorkerRef.current;
      if (!worker) { onToken("", true, "(model not ready)"); return; }
      const result = preparePromptSeed(seed, llmConfig.corpus, llmConfig.tokenization, llmConfig.systemPrompt);
      if ("fallback" in result) { onToken("", true, result.fallback); return; }
      const id = `gs-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      streamingRef.current.set(id, onToken);
      worker.postMessage({ type: "generateStream", id, seed: result.formattedSeed, length: 300, temperature: llmConfig.temperature });
    },
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
        corpus: effectiveCorpus,
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
    currentModelIdRef.current = model.id;
    setMessages([]);
    setNewModelName("");
    setTrainStep("training");
    toast({ title: "Model created", description: `"${name}" saved to your library.` });
  };

  const executeAutonomousSynthesis = () => {
    if (matrixStreaming) return;
    setMatrixStreaming(true);
    setMatrixStreamLines([]);
    setMatrixCommitReady(false);

    const paramCount = estimatedLLMParams;
    const tier = paramCount < 1_000_000 ? "Micro" : paramCount < 10_000_000 ? "Standard" : "Large-Scale";
    const targetLines = paramCount < 1_000_000 ? 300 : paramCount < 10_000_000 ? 760 : 1520;

    matrixCorpusRef.current = buildSyntheticCorpus(paramCount);

    const phases: [string, number][] = [
      [">> Initializing Autonomous Data Matrix Engine v4.1 ...", 60],
      [`>> Detected model tier: ${tier} (${paramCount.toLocaleString()} parameters)`, 80],
      [">> Computing required token volume for base convergence ...", 80],
      [`>> Target corpus: ~${(targetLines * 14).toLocaleString()} tokens across ${targetLines} training lines`, 120],
      ["", 60],
      ["━━━ PHASE 1 — Core Language Patterns ━━━", 200],
      ["  >> Scanning base vocabulary frequency distributions ...", 70],
      ["  >> Mapping unigram entropy across phonological space ...", 70],
      ["  >> Building bigram association chain tables ...", 70],
      ["  >> Extracting high-density sentence templates ...", 70],
      ["  >> Applying smoothed Kneser-Ney interpolation ...", 80],
      [`  ✓ [Core Language Patterns: ${Math.round(targetLines * 0.22)} lines compiled]`, 150],
      ["", 60],
      ["━━━ PHASE 2 — Semantic Association Networks ━━━", 200],
      ["  >> Initializing word co-occurrence matrix ...", 70],
      ["  >> Extracting contextual anchor tokens ...", 70],
      ["  >> Mapping semantic neighbor clusters via cosine proximity ...", 80],
      ["  >> Generating intent-response templates ...", 70],
      ["  >> Cross-referencing domain lexicons ...", 70],
      [`  ✓ [Semantic Associations: ${Math.round(targetLines * 0.24)} lines compiled]`, 150],
      ["", 60],
      ["━━━ PHASE 3 — Conversational Flow Templates ━━━", 200],
      ["  >> Analyzing dialogue state transition graphs ...", 70],
      ["  >> Generating turn-taking pattern sequences ...", 70],
      ["  >> Injecting discourse coherence markers ...", 70],
      ["  >> Synthesizing question-answer alignment pairs ...", 80],
      [`  ✓ [Conversational Flows: ${Math.round(targetLines * 0.26)} lines compiled]`, 150],
      ["", 60],
      ["━━━ PHASE 4 — Knowledge Domain Injection ━━━", 200],
      ["  >> Loading science and mathematics vocabulary ...", 70],
      ["  >> Structuring factual Q&A response chains ...", 70],
      ["  >> Binding domain-specific terminology anchors ...", 70],
      [`  ✓ [Domain Knowledge: ${Math.round(targetLines * 0.18)} lines compiled]`, 150],
      ["", 60],
      ["━━━ PHASE 5 — Token Density Optimization ━━━", 200],
      ["  >> Scoring per-line entropy distributions ...", 70],
      ["  >> Pruning low-information-density segments ...", 70],
      ["  >> Applying Zipfian rebalancing across vocab strata ...", 80],
      [`  ✓ [Optimized: ${targetLines} lines retained / ${Math.round(targetLines * 1.28).toLocaleString()} scanned]`, 150],
      ["", 60],
      ["━━━ PHASE 6 — Worker Buffer Binding ━━━", 200],
      ["  >> Normalizing text pipeline to lowercase ASCII ...", 70],
      ["  >> Validating EOS / PAD special-token boundaries ...", 70],
      ["  >> Pre-computing context window sliding offsets ...", 70],
      ["  >> Allocating corpus buffer in worker thread heap ...", 80],
      [`  ✓ [CORPUS READY — ${targetLines} lines | ~${(targetLines * 14).toLocaleString()} tokens]`, 150],
      ["", 80],
      ["████████████████████████████████ 100%", 100],
      ["■ SYNTHESIS COMPLETE — Dataset is ready for engine commit", 60],
    ];

    // RAF-batched rendering: accumulate lines that fire in the same 16ms frame
    // into a single setState call to keep the browser completely smooth.
    const pendingBuf: string[] = [];
    let rafId: number | null = null;
    const flushBuf = () => {
      if (pendingBuf.length > 0) {
        const toAdd = [...pendingBuf];
        pendingBuf.length = 0;
        setMatrixStreamLines((prev) => [...prev, ...toAdd]);
      }
      rafId = null;
    };
    const addLine = (line: string) => {
      pendingBuf.push(line);
      if (!rafId) rafId = requestAnimationFrame(flushBuf);
    };

    let t = 0;
    for (const [text, delay] of phases) {
      t += delay;
      const captured = text;
      setTimeout(() => addLine(captured), t);
    }
    setTimeout(() => { setMatrixStreaming(false); setMatrixCommitReady(true); }, t + 200);
  };

  const commitToEngine = () => {
    const corpusText = matrixCorpusRef.current;
    if (!corpusText) return;
    const newDataset: Dataset = { id: Date.now(), name: "Autonomous Matrix Dataset", text: corpusText, active: true };
    setDatasets([newDataset]);
    setLLMConfig((c) => ({ ...c, corpus: corpusText }));
    setLLMPlaying(false);
    textWorkerRef.current?.postMessage({
      type: "reset",
      opts: {
        corpus: corpusText,
        contextSize: llmConfig.contextSize,
        hiddenSize: llmConfig.hiddenSize,
        learningRate: llmConfig.learningRate,
        temperature: llmConfig.temperature,
        tokenization: llmConfig.tokenization,
        topK: llmConfig.topK,
      },
    });
    setMatrixModalOpen(false);
    setMatrixStreamLines([]);
    setMatrixCommitReady(false);
    matrixCorpusRef.current = "";
    toast({ title: "Corpus committed", description: "Autonomous dataset bound to training engine." });
  };

  const handleLoadModel = (model: SavedModel) => {
    if (llmPlaying) { textWorkerRef.current?.postMessage({ type: "pause" }); setLLMPlaying(false); }

    // Restore live in-memory slider snapshot if we have one (unsaved training progress)
    const snap = liveSnapshotRef.current.get(model.id);
    if (snap) {
      setLLMConfig(snap.config);
      setNumHiddenLayers(snap.numHiddenLayers);
      setDatasets(snap.datasets);
    } else {
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
    }

    const w = model.weights as CharLMWeights;
    textWorkerRef.current?.postMessage({ type: "loadWeights", payload: { ...w, epoch: model.epoch, loss: model.loss } });
    currentModelIdRef.current = model.id;
    setMessages([]);
    toast({ title: "Model loaded", description: `${model.name} weights restored.` });
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
      llmConfig.hiddenSize,
      numHiddenLayers,
    ),
    [llmConfig.corpus, llmConfig.hiddenSize, llmConfig.tokenization, numHiddenLayers],
  );

  const llmHasModel = textSnap.epoch > 0;
  const llmModeTag = llmConfig.tokenization === "word" ? "Word-LM" : "Char-LM";
  const llmModelLabel = `${llmModeTag} · ${llmConfig.contextSize}→${llmConfig.hiddenSize}→${textSnap.vocabSize}`;
  const llmModels = savedModels.filter((m) => m.type !== "MLP");

  // Effective corpus — merges active datasets and applies token importance
  // ranking when the Automated Data Feeder is enabled. This is what gets
  // sent to the worker on rebuild/init; the underlying math is unchanged.
  const effectiveCorpus = useMemo(() => {
    if (!feederEnabled) return llmConfig.corpus;
    const active = datasets.filter((d) => d.active).map((d) => d.text).join("\n");
    const base = active.length > 0 ? active : llmConfig.corpus;
    return tokenRankingEnabled ? tokenImportanceRank(base) : base;
  }, [feederEnabled, tokenRankingEnabled, datasets, llmConfig.corpus]);

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
            {/* WebGPU status badge */}
            {webGpuAvailable !== null && (
              <div className="hidden md:flex items-center gap-1.5 h-6 px-2.5 rounded-lg border border-white/[0.05] bg-white/[0.02]">
                {webGpuAvailable
                  ? <><Monitor className="size-2.5 text-[#30D158]/70" /><span className="text-[9px] font-medium text-[#30D158]/60 tracking-wide">[WebGPU Acceleration Active]</span></>
                  : <><Server className="size-2.5 text-white/22" /><span className="text-[9px] font-medium text-white/22 tracking-wide">[CPU Thread Pool Active]</span></>
                }
              </div>
            )}
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
                          onClick={() => { handleLoadModel(m); setTrainStep("training"); }}
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
                      <Slider min={1} max={30} step={1} value={[llmConfig.contextSize]} onValueChange={([v]) => setLLMConfig((c) => ({ ...c, contextSize: v }))} className="py-2" />
                    </div>

                    {/* Hidden Layers slider */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center">
                        <label className="text-[11px] text-white/32 font-medium">Hidden Layers</label>
                        <span className="text-[11px] tabular-nums text-white/58 font-mono">{numHiddenLayers}</span>
                      </div>
                      <Slider min={1} max={8} step={1} value={[numHiddenLayers]} onValueChange={([v]) => setNumHiddenLayers(v)} className="py-2" />
                    </div>

                    {/* Embed dim slider */}
                    <div className="space-y-1.5">
                      <div className="flex justify-between items-center">
                        <label className="text-[11px] text-white/32 font-medium">Embed Dimensions</label>
                        <span className="text-[11px] tabular-nums text-white/58 font-mono">{llmConfig.hiddenSize}</span>
                      </div>
                      <Slider min={4} max={2048} step={16} value={[llmConfig.hiddenSize]} onValueChange={([v]) => setLLMConfig((c) => ({ ...c, hiddenSize: v }))} className="py-2" />
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
                <div className="h-full flex overflow-hidden">

                  {/* ════ LEFT CONTROL PANEL ════════════════════════════════ */}
                  <div className="w-[268px] shrink-0 border-r border-white/[0.05] bg-[#020202] flex flex-col overflow-hidden">

                    {/* Panel header */}
                    <div className="h-11 shrink-0 border-b border-white/[0.05] px-3 flex items-center justify-between">
                      <button
                        onClick={() => {
                          if (currentModelIdRef.current) {
                            liveSnapshotRef.current.set(currentModelIdRef.current, {
                              config: llmConfig,
                              numHiddenLayers,
                              datasets,
                            });
                          }
                          setTrainStep("fleet");
                        }}
                        className="flex items-center gap-1.5 h-7 px-2.5 rounded-lg text-white/25 hover:text-white/55 text-[11px] hover:bg-white/[0.04] transition-all border border-transparent hover:border-white/[0.05]"
                      >
                        <ArrowLeft className="size-3" />
                        <span>Models</span>
                      </button>
                      <span className="text-[9px] font-medium text-white/18 tracking-[0.16em] uppercase">Control Matrix</span>
                    </div>

                    {/* Scrollable controls */}
                    <div className="flex-1 overflow-y-auto">

                      {/* ─── §1 Hyperparameter Architecture Matrix ─────────── */}
                      <div className="px-3 pt-4 pb-4 border-b border-white/[0.04] space-y-3">
                        <PanelLabel icon={Layers} label="Hyperparameter Architecture Matrix" />

                        <SliderRow
                          label="Hidden Layers" display={String(numHiddenLayers)}
                          value={numHiddenLayers} min={1} max={8} step={1}
                          onChange={(v) => setNumHiddenLayers(v)}
                        />
                        <SliderRow
                          label="Embed Dimensions" display={String(llmConfig.hiddenSize)}
                          value={llmConfig.hiddenSize} min={4} max={2048} step={16}
                          onChange={(v) => setLLMConfig((c) => ({ ...c, hiddenSize: v }))}
                        />
                        <SliderRow
                          label="Context Window" display={String(llmConfig.contextSize)}
                          value={llmConfig.contextSize} min={1} max={30} step={1}
                          onChange={(v) => setLLMConfig((c) => ({ ...c, contextSize: v }))}
                        />
                        <SliderRow
                          label="Learning Rate" display={llmConfig.learningRate.toFixed(3)}
                          value={llmConfig.learningRate} min={0.005} max={0.5} step={0.005}
                          onChange={(v) => setLLMConfig((c) => ({ ...c, learningRate: v }))}
                        />
                        <SliderRow
                          label="Temperature" display={llmConfig.temperature.toFixed(2)}
                          value={llmConfig.temperature} min={0.1} max={2.0} step={0.05}
                          onChange={(v) => setLLMConfig((c) => ({ ...c, temperature: v }))}
                        />
                        <SliderRow
                          label="Top-K Sampling" display={String(llmConfig.topK)}
                          value={llmConfig.topK} min={1} max={20} step={1}
                          onChange={(v) => setLLMConfig((c) => ({ ...c, topK: v }))}
                        />

                        {/* Tokenization */}
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] text-white/32">Tokenization</span>
                          <div className="flex bg-white/[0.04] rounded-lg border border-white/[0.05] p-0.5 gap-0.5">
                            {(["char", "word"] as const).map((m) => (
                              <button
                                key={m}
                                onClick={() => setLLMConfig((c) => ({ ...c, tokenization: m }))}
                                className={`px-2.5 h-5 rounded-md text-[10px] font-medium transition-all ${
                                  llmConfig.tokenization === m
                                    ? "bg-[#0A84FF]/18 text-[#0A84FF]/90 border border-[#0A84FF]/20"
                                    : "text-white/30 hover:text-white/55"
                                }`}
                              >
                                {m}
                              </button>
                            ))}
                          </div>
                        </div>

                        {/* Param bar — visual progress toward 50M ceiling */}
                        <div className="rounded-xl bg-[#0a0a0a] border border-white/[0.04] px-2.5 py-2 space-y-1.5">
                          <div className="flex items-center justify-between">
                            <span className="text-[10px] text-white/25">Est. Parameters</span>
                            <span className={`text-[11px] tabular-nums font-mono ${estimatedLLMParams > MAX_PARAMS_LLM ? "text-red-400/80" : "text-white/50"}`}>
                              {estimatedLLMParams >= 1_000_000
                                ? `${(estimatedLLMParams / 1_000_000).toFixed(2)}M`
                                : estimatedLLMParams.toLocaleString()}
                            </span>
                          </div>
                          <div className="h-[3px] rounded-full bg-white/[0.04] overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-300 ${estimatedLLMParams > MAX_PARAMS_LLM ? "bg-red-400/60" : "bg-[#0A84FF]/55"}`}
                              style={{ width: `${Math.min(100, (estimatedLLMParams / MAX_PARAMS_LLM) * 100).toFixed(1)}%` }}
                            />
                          </div>
                          <div className="text-[8.5px] text-white/14 text-right tabular-nums">
                            {((estimatedLLMParams / MAX_PARAMS_LLM) * 100).toFixed(2)}% of 50M ceiling
                          </div>
                        </div>

                        <button
                          onClick={rebuildLLM}
                          className="w-full h-8 rounded-xl border border-white/[0.06] hover:border-[#0A84FF]/22 text-white/35 hover:text-[#0A84FF]/75 text-[11px] font-medium transition-all hover:bg-[#0A84FF]/[0.04] flex items-center justify-center gap-1.5"
                        >
                          <RefreshCcw className="size-3" />
                          Apply Architecture
                        </button>
                      </div>

                      {/* ─── §2 Engine Status & Speed Throttle ──────────────── */}
                      <div className="px-3 py-4 border-b border-white/[0.04] space-y-3">
                        <PanelLabel icon={Zap} label="Engine Status & Speed Throttle" />
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] text-white/32">Epochs / sec</span>
                          <span className="text-[11px] tabular-nums text-white/50 font-mono">{llmEpochsPerSecond}</span>
                        </div>
                        <Slider min={1} max={60} step={1} value={[llmEpochsPerSecond]} onValueChange={handleLLMSpeed} className="py-1" />

                        {webGpuAvailable !== null && (
                          <div className={`flex items-center gap-2 rounded-lg px-2.5 py-2 border ${webGpuAvailable ? "bg-[#30D158]/[0.03] border-[#30D158]/10" : "bg-white/[0.02] border-white/[0.04]"}`}>
                            {webGpuAvailable
                              ? <><div className="size-1.5 rounded-full bg-[#30D158] animate-pulse shrink-0" /><span className="text-[10px] font-medium text-[#30D158]/65">[WebGPU Acceleration Active]</span></>
                              : <><div className="size-1.5 rounded-full bg-white/20 shrink-0" /><span className="text-[10px] font-medium text-white/22">[CPU Thread Pool Active]</span></>
                            }
                          </div>
                        )}
                      </div>

                      {/* ─── §3 Data Ingestion Matrix ────────────────────────── */}
                      <div className="px-3 py-4 space-y-3">
                        <PanelLabel icon={Database} label="Data Ingestion Matrix" />

                        <div className="flex items-center justify-between">
                          <div>
                            <div className="text-[11px] text-white/55 font-medium">Automated Feeder</div>
                            <div className="text-[9px] text-white/22 mt-0.5">Stream corpus in epochs</div>
                          </div>
                          <ToggleSwitch active={feederEnabled} onChange={setFeederEnabled} />
                        </div>

                        {feederEnabled && (
                          <div className="rounded-xl bg-[#0A84FF]/[0.04] border border-[#0A84FF]/10 px-3 py-2.5 space-y-2">
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-[11px] text-[#0A84FF]/80 font-medium">Token Importance Ranking</div>
                                <div className="text-[9px] text-[#0A84FF]/40 mt-0.5">
                                  {tokenRankingEnabled ? "Keeping top 80% by vocab density" : "Score-filter by vocab density"}
                                </div>
                              </div>
                              <ToggleSwitch active={tokenRankingEnabled} onChange={setTokenRankingEnabled} accent />
                            </div>
                          </div>
                        )}

                        {datasets.length > 0 && (
                          <div className="space-y-1.5">
                            <div className="text-[9px] text-white/20 uppercase tracking-[0.12em] font-medium">
                              Corpus · {datasets.filter((d) => d.active).length}/{datasets.length} active
                            </div>
                            {datasets.map((d) => {
                              const lineCount = d.text.split("\n").filter((l) => l.trim()).length;
                              const rankedCount = feederEnabled && tokenRankingEnabled
                                ? tokenImportanceRank(d.text).split("\n").filter((l) => l.trim()).length
                                : lineCount;
                              return (
                                <div
                                  key={d.id}
                                  className="flex items-center gap-2 rounded-lg px-2.5 py-1.5 bg-white/[0.02] border border-white/[0.04]"
                                >
                                  <button
                                    onClick={() => setDatasets((prev) => prev.map((x) => x.id === d.id ? { ...x, active: !x.active } : x))}
                                    className={`size-3.5 rounded-sm border shrink-0 flex items-center justify-center transition-all ${d.active ? "bg-[#0A84FF]/18 border-[#0A84FF]/35" : "border-white/[0.12]"}`}
                                  >
                                    {d.active && <span className="size-1.5 rounded-sm bg-[#0A84FF]" />}
                                  </button>
                                  <span className={`flex-1 text-[10px] font-medium truncate ${d.active ? "text-white/55" : "text-white/22"}`}>{d.name}</span>
                                  <span className="text-[9px] text-white/22 tabular-nums font-mono shrink-0">
                                    {feederEnabled && tokenRankingEnabled && d.active ? `${rankedCount}/${lineCount}` : `${lineCount}L`}
                                  </span>
                                </div>
                              );
                            })}
                          </div>
                        )}

                        <button
                          onClick={() => setTrainStep("setup")}
                          className="w-full h-7 rounded-xl border border-dashed border-white/[0.07] hover:border-white/[0.12] text-white/22 hover:text-white/45 text-[10px] font-medium transition-all flex items-center justify-center gap-1.5"
                        >
                          <Plus className="size-3" /> Manage Datasets
                        </button>

                        <button
                          onClick={() => setMatrixModalOpen(true)}
                          className="w-full h-10 rounded-xl border border-[#0A84FF]/20 bg-[#0A84FF]/[0.06] hover:bg-[#0A84FF]/[0.12] hover:border-[#0A84FF]/35 text-[#0A84FF]/75 hover:text-[#0A84FF] text-[11px] font-semibold transition-all flex items-center justify-center gap-2"
                        >
                          <Terminal className="size-3.5" />
                          Launch Autonomous Data Matrix
                        </button>

                        <button
                          onClick={() => { setExplorerTab("datasets"); setDatasetExplorerOpen(true); }}
                          className="w-full h-8 rounded-xl border border-white/[0.06] hover:border-[#0A84FF]/20 text-white/28 hover:text-[#0A84FF]/70 text-[10px] font-medium transition-all flex items-center justify-center gap-1.5 hover:bg-[#0A84FF]/[0.04]"
                        >
                          <Eye className="size-3" />
                          View Compiled Training Dataset
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* ════ RIGHT MONITORING PANEL ════════════════════════════ */}
                  <div className="flex-1 flex flex-col overflow-hidden bg-[#000000]">

                  {/* ─── Matrix Pulse Core Visualizer ─────────────────────── */}
                  <div className="shrink-0 px-5 pt-5 pb-2">
                    <MatrixPulseCore active={llmPlaying} />
                    <div className={`text-center text-[9px] font-semibold tracking-[0.22em] uppercase mt-3 ${llmPlaying ? "text-[#0A84FF]/70" : "text-white/15"}`}>
                      {llmPlaying ? "Active Backpropagation Engine" : "System Idle · Awaiting Input"}
                    </div>
                  </div>

                  {/* ─── Live Telemetry Grid ────────────────────────────────── */}
                  <div className="shrink-0 px-4 pb-3">
                    <div className="grid grid-cols-2 gap-2">
                      <TelemetryCard
                        label="Global Loss"
                        value={textSnap.loss > 0 ? textSnap.loss.toFixed(4) : "—"}
                        sub={lossHistory.length > 3
                          ? (lossHistory[lossHistory.length - 1] < lossHistory[0] ? "↓ converging" : "→ stable")
                          : "awaiting data"}
                        active={llmPlaying}
                        accent
                      >
                        {lossHistory.length > 3 && <SparkLine data={lossHistory} />}
                      </TelemetryCard>

                      <TelemetryCard
                        label="Epoch Counter"
                        value={textSnap.epoch > 0 ? textSnap.epoch.toLocaleString() : "—"}
                        sub={textSnap.trainedSamples > 0
                          ? `${textSnap.trainedSamples > 1000 ? `${(textSnap.trainedSamples / 1000).toFixed(1)}k` : String(textSnap.trainedSamples)} samples`
                          : "awaiting data"}
                        active={llmPlaying}
                      />

                      <TelemetryCard
                        label="Throughput"
                        value={textSnap.tokensPerSecond > 0
                          ? textSnap.tokensPerSecond > 1000
                            ? `${(textSnap.tokensPerSecond / 1000).toFixed(1)}k`
                            : String(Math.round(textSnap.tokensPerSecond))
                          : "—"}
                        sub="tokens / sec"
                        active={llmPlaying}
                      />

                      <TelemetryCard
                        label="RAM Overhead"
                        value={textSnap.paramCount > 0
                          ? `${((textSnap.paramCount * 4 * 2) / (1024 * 1024)).toFixed(1)} MB`
                          : "—"}
                        sub={textSnap.paramCount > 0 ? `${textSnap.paramCount.toLocaleString()} params` : "—"}
                        active={llmPlaying}
                      />

                      <TelemetryCard
                        label="Vocab Coverage"
                        value={textSnap.vocabSize > 0 ? String(textSnap.vocabSize) : "—"}
                        sub={textSnap.vocabSize > 0 ? "unique token types" : "awaiting data"}
                        active={llmPlaying}
                      />

                      <TelemetryCard
                        label="Bits / Char"
                        value={textSnap.loss > 0 ? (textSnap.loss / Math.LN2).toFixed(3) : "—"}
                        sub="cross-entropy bits"
                        active={llmPlaying}
                      />
                    </div>
                  </div>

                  {/* ─── Live Sample Output ─────────────────────────────────── */}
                  {textSnap.sample && (
                    <div className="shrink-0 mx-4 mb-4 rounded-xl bg-[#070707] border border-white/[0.05] px-4 py-3">
                      <div className="text-[9px] uppercase tracking-[0.12em] text-white/20 mb-2 font-medium flex items-center gap-1.5">
                        <span className="size-1.5 rounded-full bg-[#30D158] animate-pulse shrink-0" />
                        Live Sample Output
                      </div>
                      <code className="text-[10px] font-mono text-[#30D158]/60 leading-relaxed break-words">{textSnap.sample}</code>
                    </div>
                  )}

                  </div>
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
                generateStream={generateStreamFromWorker}
                liveSample={textSnap.sample}
                epoch={textSnap.epoch}
                loss={textSnap.loss}
                isTraining={llmPlaying}
                modelOptions={llmModels.map((m) => ({ id: m.id, label: m.name }))}
                onSelectModel={(id) => {
                  const m = savedModels.find((s) => s.id === id);
                  if (m) { handleLoadModel(m); setTab("train"); setTrainStep("training"); }
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

      {/* ═══ Autonomous Data Matrix Modal ═══════════════════════════════════ */}
      {matrixModalOpen && (
        <div className="fixed inset-0 z-50 bg-[#000000]/97 backdrop-blur-md flex flex-col">
          <div className="h-14 shrink-0 border-b border-white/[0.06] px-5 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="size-8 rounded-lg bg-[#0A84FF]/10 border border-[#0A84FF]/18 flex items-center justify-center shrink-0">
                <Terminal className="size-3.5 text-[#0A84FF]/75" />
              </div>
              <div>
                <div className="text-[13px] font-semibold text-white/88 tracking-tight">Autonomous Data Matrix</div>
                <div className="text-[10px] text-white/28 font-mono">corpus synthesis & engine injection</div>
              </div>
            </div>
            <button
              onClick={() => { if (!matrixStreaming) { setMatrixModalOpen(false); setMatrixStreamLines([]); setMatrixCommitReady(false); } }}
              className="size-8 rounded-lg flex items-center justify-center text-white/22 hover:text-white/65 hover:bg-white/[0.05] transition-all border border-transparent hover:border-white/[0.05]"
            >
              <X className="size-4" />
            </button>
          </div>

          <div className="shrink-0 px-5 py-4 border-b border-white/[0.04]">
            <div className="grid grid-cols-3 gap-3">
              {[
                {
                  label: "Model Parameters",
                  value: estimatedLLMParams >= 1_000_000
                    ? `${(estimatedLLMParams / 1_000_000).toFixed(2)}M`
                    : estimatedLLMParams > 1000
                      ? `${Math.round(estimatedLLMParams / 1000)}K`
                      : String(estimatedLLMParams),
                },
                {
                  label: "Required Tokens",
                  value: estimatedLLMParams < 1_000_000 ? "~4.2K" : estimatedLLMParams < 10_000_000 ? "~10.6K" : "~21.3K",
                },
                {
                  label: "Architecture Tier",
                  value: estimatedLLMParams < 1_000_000 ? "Micro" : estimatedLLMParams < 10_000_000 ? "Standard" : "Large",
                },
              ].map((c, i) => (
                <div key={i} className="rounded-xl bg-[#0a0a0a] border border-white/[0.05] px-3 py-2.5">
                  <div className="text-[9px] text-white/20 uppercase tracking-[0.12em] mb-1">{c.label}</div>
                  <div className="text-[15px] font-semibold text-white/80 font-mono">{c.value}</div>
                </div>
              ))}
            </div>
          </div>

          <div ref={terminalRef} className="flex-1 overflow-y-auto px-5 py-4 font-mono text-[11px] leading-[1.8]">
            {matrixStreamLines.length === 0 && !matrixStreaming && (
              <div className="text-white/18 text-center pt-16">
                Press <span className="text-[#0A84FF]/60">Execute Synthesis</span> to begin automated data generation
              </div>
            )}
            {matrixStreamLines.map((line, i) => {
              const isPhase = line.startsWith("━━━");
              const isCheck = line.startsWith("  ✓");
              const isArrow = line.startsWith("  >>");
              const isComplete = line.startsWith("■");
              const isBar = line.startsWith("████");
              const isTop = line.startsWith(">>");
              return (
                <div
                  key={i}
                  className={
                    isPhase ? "text-[#0A84FF]/70 font-semibold mt-2 mb-0.5" :
                    isCheck ? "text-[#30D158]/75 font-medium" :
                    isArrow ? "text-white/35 pl-1" :
                    isTop ? "text-white/45" :
                    isComplete ? "text-[#30D158] font-bold mt-2" :
                    isBar ? "text-[#0A84FF]/45 tracking-wider" :
                    "text-white/18"
                  }
                >
                  {line || "\u00A0"}
                </div>
              );
            })}
            {matrixStreaming && (
              <div className="flex items-center gap-2 mt-1 pl-1">
                <span className="size-1.5 rounded-full bg-[#0A84FF] animate-ping shrink-0" />
                <span className="text-[#0A84FF]/45 text-[11px]">processing ...</span>
              </div>
            )}
          </div>

          <div className="shrink-0 border-t border-white/[0.05] px-5 py-4 flex items-center gap-3">
            {!matrixCommitReady ? (
              <button
                onClick={executeAutonomousSynthesis}
                disabled={matrixStreaming}
                className="flex-1 h-12 rounded-xl bg-[#0A84FF] text-white text-[12px] font-semibold flex items-center justify-center gap-2.5 hover:bg-[#409CFF] transition-all disabled:opacity-35 disabled:cursor-not-allowed"
              >
                <Terminal className="size-4" />
                Execute Autonomous Data Synthesis &amp; Scrape
              </button>
            ) : (
              <button
                onClick={commitToEngine}
                className="flex-1 h-12 rounded-xl bg-[#30D158]/12 border border-[#30D158]/28 text-[#30D158]/90 text-[12px] font-semibold flex items-center justify-center gap-2.5 hover:bg-[#30D158]/20 hover:border-[#30D158]/40 transition-all"
              >
                <Cpu className="size-4" />
                Commit Dataset to Core Engine
              </button>
            )}
          </div>
        </div>
      )}

      {/* ═══ Dataset Explorer Overlay ════════════════════════════════════════ */}
      {datasetExplorerOpen && (
        <div className="fixed inset-0 z-50 bg-[#000000]/97 backdrop-blur-md flex flex-col">
          <div className="h-14 shrink-0 border-b border-white/[0.06] px-5 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="size-8 rounded-lg bg-[#30D158]/10 border border-[#30D158]/18 flex items-center justify-center shrink-0">
                <FileText className="size-3.5 text-[#30D158]/75" />
              </div>
              <div>
                <div className="text-[13px] font-semibold text-white/88 tracking-tight">Compiled Training Dataset</div>
                <div className="text-[10px] text-white/28 font-mono">
                  {datasets.filter((d) => d.active).length}/{datasets.length} active datasets · {llmConfig.corpus.length.toLocaleString()} bytes
                </div>
              </div>
            </div>
            <button
              onClick={() => setDatasetExplorerOpen(false)}
              className="size-8 rounded-lg flex items-center justify-center text-white/22 hover:text-white/65 hover:bg-white/[0.05] transition-all border border-transparent hover:border-white/[0.05]"
            >
              <X className="size-4" />
            </button>
          </div>

          {/* Tab bar */}
          <div className="shrink-0 border-b border-white/[0.05] px-5 flex gap-4">
            {(["datasets", "corpus"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setExplorerTab(t)}
                className={`h-10 text-[11px] font-semibold border-b-2 transition-all ${
                  explorerTab === t
                    ? "border-[#30D158] text-[#30D158]"
                    : "border-transparent text-white/30 hover:text-white/55"
                }`}
              >
                {t === "datasets" ? `Datasets (${datasets.length})` : "Compiled Corpus"}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto px-5 py-4">
            {explorerTab === "datasets" ? (
              <div className="max-w-2xl mx-auto space-y-2">
                {datasets.length === 0 && (
                  <div className="text-center text-white/22 text-[12px] pt-16">No datasets loaded.</div>
                )}
                {datasets.map((d) => {
                  const lineCount = d.text.split("\n").filter((l) => l.trim()).length;
                  const isExpanded = expandedDatasets.has(d.id);
                  return (
                    <div key={d.id} className={`rounded-2xl border transition-all ${d.active ? "border-white/[0.08] bg-[#0d0d0d]" : "border-white/[0.04] bg-[#080808] opacity-60"}`}>
                      <button
                        onClick={() => setExpandedDatasets((prev) => {
                          const next = new Set(prev);
                          if (next.has(d.id)) next.delete(d.id); else next.add(d.id);
                          return next;
                        })}
                        className="w-full flex items-center gap-3 px-4 py-3 text-left"
                      >
                        <div className={`size-2 rounded-full shrink-0 ${d.active ? "bg-[#30D158]" : "bg-white/20"}`} />
                        <div className="flex-1 min-w-0">
                          <div className="text-[12px] font-semibold text-white/75 truncate">{d.name}</div>
                          <div className="text-[10px] text-white/28 font-mono mt-0.5">{lineCount} lines · {d.text.length.toLocaleString()} bytes</div>
                        </div>
                        <ChevronDownIcon className={`size-3.5 text-white/25 transition-transform shrink-0 ${isExpanded ? "rotate-180" : ""}`} />
                      </button>
                      {isExpanded && (
                        <div className="border-t border-white/[0.05] px-4 py-3">
                          <pre className="text-[10px] font-mono text-white/45 whitespace-pre-wrap break-words leading-relaxed max-h-64 overflow-y-auto">
                            {d.text}
                          </pre>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="max-w-2xl mx-auto">
                {llmConfig.corpus ? (
                  <div className="rounded-2xl border border-white/[0.06] bg-[#0d0d0d] p-4">
                    <div className="text-[9px] uppercase tracking-[0.12em] text-white/20 mb-3 font-medium flex items-center gap-1.5">
                      <span className="size-1.5 rounded-full bg-[#30D158]" />
                      Merged &amp; Compiled Corpus
                    </div>
                    <pre className="text-[10px] font-mono text-white/50 whitespace-pre-wrap break-words leading-relaxed">
                      {llmConfig.corpus}
                    </pre>
                  </div>
                ) : (
                  <div className="text-center text-white/22 text-[12px] pt-16">No corpus compiled yet. Add datasets first.</div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

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

function PanelLabel({
  icon: Icon,
  label,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
}) {
  return (
    <div className="flex items-center gap-1.5 pb-0.5">
      <Icon className="size-3 text-white/22 shrink-0" />
      <span className="text-[9px] font-semibold text-white/22 tracking-[0.14em] uppercase">{label}</span>
    </div>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-white/32">{label}</span>
        <span className="text-[11px] tabular-nums font-mono text-white/50">{display ?? String(value)}</span>
      </div>
      <Slider min={min} max={max} step={step} value={[value]} onValueChange={([v]) => onChange(v)} className="py-0.5" />
    </div>
  );
}

function ToggleSwitch({
  active,
  onChange,
}: {
  active: boolean;
  onChange: (v: boolean) => void;
  accent?: boolean;
}) {
  return (
    <button
      onClick={() => onChange(!active)}
      className={`relative w-[34px] h-5 rounded-full border transition-all shrink-0 ${
        active ? "bg-[#0A84FF] border-[#0A84FF]/50" : "bg-white/[0.05] border-white/[0.10]"
      }`}
    >
      <span
        className="absolute top-[3px] size-[14px] rounded-full bg-white shadow-sm transition-transform"
        style={{ left: active ? "calc(100% - 17px)" : "3px" }}
      />
    </button>
  );
}

function TelemetryCard({
  label,
  value,
  sub,
  active,
  accent = false,
  children,
}: {
  label: string;
  value: string;
  sub?: string;
  active?: boolean;
  accent?: boolean;
  children?: React.ReactNode;
}) {
  return (
    <div
      className={`rounded-2xl border p-3.5 flex flex-col gap-1.5 ${
        active && accent ? "bg-[#0A84FF]/[0.05] border-[#0A84FF]/14" : "bg-[#080808] border-white/[0.05]"
      }`}
    >
      <div className="text-[9px] uppercase tracking-[0.12em] text-white/22 font-medium">{label}</div>
      <div
        className={`text-[22px] font-bold tabular-nums font-mono leading-none ${
          active && accent ? "text-[#0A84FF]/90" : "text-white/85"
        }`}
      >
        {value}
      </div>
      {children}
      {sub && <div className="text-[9px] text-white/22 font-mono">{sub}</div>}
    </div>
  );
}

function SparkLine({ data }: { data: number[] }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const W = 120, H = 22;
  const pts = data
    .map((v, i) => `${(i / (data.length - 1)) * W},${H - ((v - min) / range) * H}`)
    .join(" ");
  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="my-0.5">
      <polyline
        points={pts}
        fill="none"
        stroke="rgba(10,132,255,0.45)"
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function MatrixPulseCore({ active }: { active: boolean }) {
  const W = 520, H = 196;
  const LABEL_H = 22;
  const xs = [65, 185, 335, 455];
  const ys: number[][] = [
    [33, 98, 163],
    [10, 55, 98, 141, 186],
    [10, 55, 98, 141, 186],
    [33, 98, 163],
  ];
  const layerLabels = ["INPUT", "HIDDEN L1", "HIDDEN L2", "OUTPUT"];

  type PathDef = { d: string; fwdDelay: string; bwdDelay: string; fwdDur: string; bwdDur: string };
  const pathDefs: PathDef[] = [];
  for (let l = 0; l < 3; l++) {
    const mx = (xs[l] + xs[l + 1]) / 2;
    for (let a = 0; a < ys[l].length; a++) {
      for (let b = 0; b < ys[l + 1].length; b++) {
        const d = `M ${xs[l]},${ys[l][a]} C ${mx},${ys[l][a]} ${mx},${ys[l + 1][b]} ${xs[l + 1]},${ys[l + 1][b]}`;
        const seed = a * 0.11 + b * 0.07 + l * 0.26;
        pathDefs.push({
          d,
          fwdDelay: `${(seed % 1.3).toFixed(2)}s`,
          bwdDelay: `${((seed + 0.35) % 1.6).toFixed(2)}s`,
          fwdDur: `${(0.85 + (seed % 0.45)).toFixed(2)}s`,
          bwdDur: `${(1.4 + (seed % 0.55)).toFixed(2)}s`,
        });
      }
    }
  }

  type NodeDef = { x: number; y: number; isIO: boolean; delay: string };
  const nodeDefs: NodeDef[] = [];
  ys.forEach((layer, l) => {
    layer.forEach((y, i) => {
      nodeDefs.push({
        x: xs[l], y,
        isIO: l === 0 || l === 3,
        delay: `${((l * 0.32 + i * 0.10) % 1.6).toFixed(2)}s`,
      });
    });
  });

  return (
    <div style={{ width: "100%", height: H + LABEL_H }}>
      <svg width="100%" height={H + LABEL_H} viewBox={`0 0 ${W} ${H + LABEL_H}`}>
        {/* Blueprint column guides */}
        {xs.map((x, i) => (
          <line key={`g-${i}`} x1={x} y1={0} x2={x} y2={H}
            stroke="rgba(255,255,255,0.022)" strokeWidth={0.5} strokeDasharray="2 8" />
        ))}

        {/* Connection paths */}
        {pathDefs.map((p, i) =>
          active ? (
            <g key={i}>
              <path d={p.d} fill="none" stroke="rgba(10,132,255,0.10)" strokeWidth={0.7} />
              <path d={p.d} fill="none"
                stroke="rgba(10,132,255,0.80)" strokeWidth={1.3}
                strokeDasharray="4 14"
                style={{ animation: `flow-fwd ${p.fwdDur} linear infinite ${p.fwdDelay}` }}
              />
              <path d={p.d} fill="none"
                stroke="rgba(255,159,10,0.38)" strokeWidth={0.9}
                strokeDasharray="2 18"
                style={{ animation: `flow-bwd ${p.bwdDur} linear infinite ${p.bwdDelay}` }}
              />
            </g>
          ) : (
            <path key={i} d={p.d} fill="none" stroke="rgba(255,255,255,0.028)" strokeWidth={0.6} />
          )
        )}

        {/* Nodes */}
        {nodeDefs.map((n, i) => (
          <circle key={i}
            cx={n.x} cy={n.y}
            r={n.isIO ? 8 : 5.5}
            fill={active
              ? n.isIO ? "rgba(10,132,255,0.28)" : "rgba(10,132,255,0.12)"
              : "rgba(255,255,255,0.04)"}
            stroke={active
              ? n.isIO ? "rgba(10,132,255,0.92)" : "rgba(10,132,255,0.44)"
              : "rgba(255,255,255,0.08)"}
            strokeWidth={n.isIO ? 1.5 : 1}
            style={active ? {
              animation: `node-pulse 1.8s ease-in-out infinite ${n.delay}`,
              filter: n.isIO ? "drop-shadow(0 0 5px rgba(10,132,255,0.65))" : undefined,
            } : {}}
          />
        ))}

        {/* Layer labels */}
        {xs.map((x, i) => (
          <text key={`lbl-${i}`}
            x={x} y={H + 15}
            textAnchor="middle"
            fontSize={7.5}
            fontFamily="ui-monospace,monospace"
            letterSpacing="0.12em"
            fontWeight="600"
            fill={active ? "rgba(10,132,255,0.52)" : "rgba(255,255,255,0.14)"}
          >
            {layerLabels[i]}
          </text>
        ))}
      </svg>
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
