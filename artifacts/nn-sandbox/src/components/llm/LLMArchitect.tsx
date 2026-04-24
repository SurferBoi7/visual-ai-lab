import { useRef, useState } from "react";
import {
  Cpu,
  Sparkles,
  BookOpen,
  Upload,
  FileText,
  Lightbulb,
  Type,
  WholeWord,
  Filter,
  MessageSquare,
  GraduationCap,
  Plus,
  Globe,
  Loader2,
  FileCode2,
  Hammer,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { Slider } from "@/components/ui/slider";

export type Tokenization = "char" | "word";

export interface LLMConfig {
  corpus: string;
  contextSize: number;
  hiddenSize: number;
  learningRate: number;
  temperature: number;
  tokenization: Tokenization;
  topK: number;
  systemPrompt: string;
}

interface Props {
  config: LLMConfig;
  onChange: (next: LLMConfig) => void;
  onApply: () => void;
  paramCount: number;
  vocabSize: number;
  maxParams: number;
}

// 1MB hard cap on the corpus to keep the worker from OOM-crashing the tab.
const MAX_CORPUS_BYTES = 1_000_000;

// Alternating prompts for the Bulk Text Formatter conversation simulation.
const BULK_USER_PROMPTS = [
  "tell me a story",
  "tell me more",
  "keep going",
  "what happens next",
  "continue",
];

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

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export function LLMArchitect({
  config,
  onChange,
  onApply,
  paramCount,
  vocabSize,
  maxParams,
}: Props) {
  const overBudget = paramCount > maxParams;
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const bulkFileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadInfo, setUploadInfo] = useState<{
    name: string;
    bytes: number;
    truncated: boolean;
  } | null>(null);
  const [readError, setReadError] = useState<string | null>(null);
  const [factQ, setFactQ] = useState("");
  const [factA, setFactA] = useState("");

  // --- Wikipedia Knowledge Fetcher state ---
  const [wikiTopic, setWikiTopic] = useState("");
  const [wikiStatus, setWikiStatus] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");
  const [wikiMsg, setWikiMsg] = useState("");

  // --- Bulk Text Formatter state ---
  const [bulkFormatInfo, setBulkFormatInfo] = useState<{
    name: string;
    paragraphs: number;
  } | null>(null);
  const [bulkError, setBulkError] = useState<string | null>(null);

  const appendToCorpus = (block: string) => {
    const sep =
      config.corpus.endsWith("\n") || config.corpus.length === 0 ? "" : "\n";
    const next = (config.corpus + sep + block + "\n").slice(
      0,
      MAX_CORPUS_BYTES,
    );
    onChange({ ...config, corpus: next });
  };

  // Generate 4-5 hardcoded phrasings of the user's question and append them to
  // the corpus as `User: <q>\nBot: <a>` pairs.
  const addFactVariations = () => {
    const q = factQ.trim().replace(/[?.!]+$/g, "");
    const a = factA.trim();
    if (!q || !a) return;
    const variations = [
      `${q}?`,
      `What is ${q}?`,
      `Tell me ${q}.`,
      `Can you tell me ${q}?`,
      `Do you know ${q}?`,
    ];
    const block = variations.map((v) => `User: ${v}\nBot: ${a}`).join("\n");
    appendToCorpus(block);
    setFactQ("");
    setFactA("");
  };

  // Wikipedia Knowledge Fetcher
  const fetchWikipedia = async () => {
    const topic = wikiTopic.trim();
    if (!topic) return;
    setWikiStatus("loading");
    setWikiMsg("");
    try {
      const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(topic)}`;
      const res = await fetch(url);
      if (res.status === 404) {
        setWikiStatus("error");
        setWikiMsg(`No Wikipedia page found for "${topic}".`);
        return;
      }
      if (!res.ok) {
        setWikiStatus("error");
        setWikiMsg(`Wikipedia returned an error (${res.status}). Try again.`);
        return;
      }
      const data = await res.json();
      const extract: string = data.extract ?? "";
      if (!extract) {
        setWikiStatus("error");
        setWikiMsg("Wikipedia returned an empty summary for this topic.");
        return;
      }
      const block = `User: tell me about ${topic}\nBot: ${extract}`;
      const corpusBytes = new Blob([config.corpus]).size;
      const blockBytes = new Blob([block]).size;
      if (corpusBytes + blockBytes > MAX_CORPUS_BYTES) {
        setWikiStatus("error");
        setWikiMsg(
          "Corpus is full (1 MB limit). Remove some text before adding more.",
        );
        return;
      }
      appendToCorpus(block);
      setWikiStatus("success");
      setWikiMsg(`Added Wikipedia summary for "${topic}".`);
      setWikiTopic("");
    } catch {
      setWikiStatus("error");
      setWikiMsg("Network error — check your connection and try again.");
    }
  };

  // Bulk Text Formatter
  const handleBulkFile = (files: FileList | null) => {
    setBulkError(null);
    setBulkFormatInfo(null);
    const f = files?.[0];
    if (!f) return;
    if (!/\.txt$/i.test(f.name) && !f.type.startsWith("text/")) {
      setBulkError("Only .txt files are supported.");
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => setBulkError("Could not read that file.");
    reader.onload = () => {
      const raw = String(reader.result ?? "");
      // Split by double newline first, fall back to single newline.
      const rawParagraphs = raw.split(/\n\n+/).flatMap((chunk) => {
        const trimmed = chunk.trim();
        if (!trimmed) return [];
        // If the chunk itself has multiple lines, keep as-is (it's a paragraph).
        return [trimmed];
      });
      const paragraphs = rawParagraphs.filter((p) => p.length > 0);
      if (paragraphs.length === 0) {
        setBulkError("No usable paragraphs found in that file.");
        return;
      }
      const block = paragraphs
        .map(
          (p, i) =>
            `User: ${BULK_USER_PROMPTS[i % BULK_USER_PROMPTS.length]}\nBot: ${p}`,
        )
        .join("\n");
      const corpusBytes = new Blob([config.corpus]).size;
      const blockBytes = new Blob([block]).size;
      const combined = config.corpus + (config.corpus.endsWith("\n") || config.corpus.length === 0 ? "" : "\n") + block + "\n";
      const truncated = combined.length > MAX_CORPUS_BYTES;
      onChange({ ...config, corpus: combined.slice(0, MAX_CORPUS_BYTES) });
      setBulkFormatInfo({
        name: f.name,
        paragraphs: paragraphs.length,
      });
      if (truncated) {
        setBulkError(
          `Corpus hit the 1 MB limit — some paragraphs were truncated.`,
        );
      }
      // Reset the file input so the same file can be re-selected.
      if (bulkFileInputRef.current) bulkFileInputRef.current.value = "";
      void corpusBytes; void blockBytes;
    };
    reader.readAsText(f);
  };

  const isWord = config.tokenization === "word";
  const tokenLabel = isWord ? "tokens" : "chars";
  const corpusBytes = new Blob([config.corpus]).size;

  const handleFiles = (files: FileList | null) => {
    setReadError(null);
    const f = files?.[0];
    if (!f) return;
    if (!/\.txt$/i.test(f.name) && !f.type.startsWith("text/")) {
      setReadError("Only .txt files are supported.");
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => setReadError("Could not read that file.");
    reader.onload = () => {
      const raw = String(reader.result ?? "");
      const truncated = raw.length > MAX_CORPUS_BYTES;
      const text = truncated ? raw.slice(0, MAX_CORPUS_BYTES) : raw;
      setUploadInfo({ name: f.name, bytes: f.size, truncated });
      onChange({ ...config, corpus: text });
    };
    reader.readAsText(f);
  };

  return (
    <div className="space-y-4">
      <Card className="p-5 space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu className="size-4 text-sky-400" />
            <span className="text-sm font-semibold text-slate-100">
              Tiny-LM Architecture
            </span>
          </div>
          <span
            className={`text-[11px] tabular-nums ${
              overBudget ? "text-red-400" : "text-slate-400"
            }`}
          >
            {paramCount.toLocaleString()} / {maxParams.toLocaleString()} params
          </span>
        </div>

        {/* Tokenization toggle */}
        <div className="space-y-2">
          <span className="text-xs text-slate-400">Tokenization</span>
          <div className="grid grid-cols-2 gap-1 rounded-xl bg-slate-950/60 border border-slate-700 p-1">
            <button
              onClick={() => onChange({ ...config, tokenization: "char" })}
              className={`min-h-[40px] rounded-lg text-xs font-semibold inline-flex items-center justify-center gap-1.5 transition ${
                !isWord
                  ? "bg-sky-500/20 text-sky-200 shadow-inner"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Type className="size-3.5" />
              Char-Level
            </button>
            <button
              onClick={() => onChange({ ...config, tokenization: "word" })}
              className={`min-h-[40px] rounded-lg text-xs font-semibold inline-flex items-center justify-center gap-1.5 transition ${
                isWord
                  ? "bg-emerald-500/20 text-emerald-200 shadow-inner"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <WholeWord className="size-3.5" />
              Word-Level
            </button>
          </div>
          <p className="text-[10px] text-slate-500">
            {isWord
              ? "Vocabulary is built from whole words and punctuation. Better for sentences, larger vocab."
              : "Vocabulary is built from individual characters. Tiny vocab, learns spelling."}
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Context Window</span>
            <span className="text-xs tabular-nums text-slate-200">
              {config.contextSize} {tokenLabel}
            </span>
          </div>
          <Slider
            min={1}
            max={20}
            step={1}
            value={[config.contextSize]}
            onValueChange={([v]) => onChange({ ...config, contextSize: v })}
            className="py-2"
          />
          <p className="text-[10px] text-slate-500">
            How many {tokenLabel} the model sees before predicting the next one.
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Hidden Neurons</span>
            <span className="text-xs tabular-nums text-slate-200">
              {config.hiddenSize}
            </span>
          </div>
          <Slider
            min={4}
            max={512}
            step={4}
            value={[config.hiddenSize]}
            onValueChange={([v]) => onChange({ ...config, hiddenSize: v })}
            className="py-2"
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Learning Rate</span>
            <span className="text-xs tabular-nums text-slate-200">
              {config.learningRate.toFixed(3)}
            </span>
          </div>
          <Slider
            min={0.005}
            max={0.5}
            step={0.005}
            value={[config.learningRate]}
            onValueChange={([v]) => onChange({ ...config, learningRate: v })}
            className="py-2"
          />
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">
              Sampling Temperature
            </span>
            <span className="text-xs tabular-nums text-slate-200">
              {config.temperature.toFixed(2)}
            </span>
          </div>
          <Slider
            min={0.1}
            max={1.5}
            step={0.05}
            value={[config.temperature]}
            onValueChange={([v]) => onChange({ ...config, temperature: v })}
            className="py-2"
          />
          <p className="text-[10px] text-slate-500">
            Lower = focused / repetitive. Higher = creative / chaotic.
          </p>
        </div>

        {/* Top-K sampling */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-xs text-slate-400 inline-flex items-center gap-1.5">
              <Filter className="size-3 text-fuchsia-300" />
              Top-K Sampling
            </span>
            <span className="text-xs tabular-nums text-slate-200">
              {config.topK}
            </span>
          </div>
          <Slider
            min={1}
            max={40}
            step={1}
            value={[config.topK]}
            onValueChange={([v]) => onChange({ ...config, topK: v })}
            className="py-2"
          />
          <p className="text-[10px] text-slate-500">
            Only the {config.topK} most-likely next tokens are considered when
            sampling. Lower = safer & more on-topic.
          </p>
        </div>

        {/* System prompt */}
        <div className="space-y-2">
          <span className="text-xs text-slate-400 inline-flex items-center gap-1.5">
            <MessageSquare className="size-3 text-sky-300" />
            System Prompt
          </span>
          <input
            type="text"
            value={config.systemPrompt}
            onChange={(e) =>
              onChange({ ...config, systemPrompt: e.target.value })
            }
            placeholder="Bot: I am a helpful AI."
            className="w-full min-h-[40px] rounded-xl border border-slate-700 bg-slate-900/60 px-3 py-2 text-xs font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40"
          />
          <p className="text-[10px] text-slate-500">
            Invisibly prepended to every chat prompt to lock the bot's persona.
          </p>
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-3 flex flex-wrap gap-1.5">
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-300 border border-sky-500/30">
            vocab: {vocabSize.toLocaleString()}
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-fuchsia-500/10 text-fuchsia-300 border border-fuchsia-500/30">
            {isWord ? "word-level" : "char-level"}
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
            softmax · cross-entropy
          </span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/30">
            pure TS · in-worker
          </span>
        </div>
      </Card>

      <Card className="p-5 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <BookOpen className="size-4 text-amber-400" />
            <span className="text-sm font-semibold text-slate-100">
              Training Corpus
            </span>
          </div>
          <span className="text-[10px] text-slate-500 tabular-nums">
            {formatBytes(corpusBytes)} / {formatBytes(MAX_CORPUS_BYTES)}
          </span>
        </div>

        {/* Pro tip */}
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 flex gap-2.5">
          <Lightbulb className="size-4 text-amber-300 shrink-0 mt-0.5" />
          <div className="text-[11px] text-amber-100/90 leading-relaxed">
            <span className="font-semibold text-amber-200">Pro tip:</span> To
            train a Q&A bot, upload a text file formatted as
            <code className="mx-1 px-1.5 py-0.5 rounded bg-slate-900/80 text-amber-200 font-mono text-[10px]">
              User: question Bot: answer
            </code>
            with one pair per line.
          </div>
        </div>

        {/* Drag and drop */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragOver(false);
            handleFiles(e.dataTransfer.files);
          }}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
          className={`rounded-xl border-2 border-dashed p-4 text-center cursor-pointer transition select-none min-h-[110px] flex flex-col items-center justify-center gap-1.5 ${
            dragOver
              ? "border-sky-400 bg-sky-500/10"
              : "border-slate-700 hover:border-slate-500 hover:bg-slate-900/60"
          }`}
        >
          <Upload className="size-5 text-sky-300" />
          <div className="text-xs font-semibold text-slate-200">
            Drop a <code className="text-sky-300">.txt</code> file here
          </div>
          <div className="text-[10px] text-slate-500">
            or tap to browse — max {formatBytes(MAX_CORPUS_BYTES)}, larger files
            will be truncated.
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,text/plain"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
        </div>

        {uploadInfo && (
          <div className="flex items-start gap-2 rounded-xl border border-slate-700 bg-slate-900/60 px-3 py-2">
            <FileText className="size-3.5 text-emerald-300 mt-0.5 shrink-0" />
            <div className="min-w-0 flex-1">
              <div className="text-[11px] text-slate-200 truncate">
                {uploadInfo.name}
              </div>
              <div className="text-[10px] text-slate-500">
                {formatBytes(uploadInfo.bytes)}
                {uploadInfo.truncated && (
                  <span className="text-amber-300">
                    {" "}
                    · truncated to {formatBytes(MAX_CORPUS_BYTES)}
                  </span>
                )}
              </div>
            </div>
          </div>
        )}

        {readError && (
          <div className="text-[11px] text-rose-300 bg-rose-500/10 border border-rose-500/30 rounded-lg px-2.5 py-1.5">
            {readError}
          </div>
        )}

        {/* Fact Teacher */}
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-3 space-y-2.5">
          <div className="flex items-center gap-2">
            <GraduationCap className="size-4 text-emerald-300" />
            <span className="text-xs font-semibold text-emerald-100">
              Fact Teacher
            </span>
          </div>
          <p className="text-[10px] text-emerald-100/70 leading-relaxed">
            Teach the bot one fact in 5 different phrasings — pairs are
            appended to the corpus as <code className="font-mono">User:/Bot:</code> lines.
          </p>
          <input
            type="text"
            value={factQ}
            onChange={(e) => setFactQ(e.target.value)}
            placeholder="Question (e.g. the capital of France)"
            className="w-full min-h-[36px] rounded-lg border border-slate-700 bg-slate-950/60 px-2.5 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
          />
          <input
            type="text"
            value={factA}
            onChange={(e) => setFactA(e.target.value)}
            placeholder="Answer (e.g. Paris)"
            className="w-full min-h-[36px] rounded-lg border border-slate-700 bg-slate-950/60 px-2.5 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/40"
          />
          <button
            type="button"
            onClick={addFactVariations}
            disabled={!factQ.trim() || !factA.trim()}
            className="w-full min-h-[38px] rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-400/40 text-emerald-100 text-xs font-semibold inline-flex items-center justify-center gap-1.5 transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="size-3.5" />
            Add Fact Variations
          </button>
        </div>

        {/* ── Data Forge ── */}
        <div className="rounded-xl border border-violet-500/30 bg-violet-500/5 p-3 space-y-4">
          <div className="flex items-center gap-2">
            <Hammer className="size-4 text-violet-300" />
            <span className="text-xs font-semibold text-violet-100">
              Data Forge
            </span>
          </div>
          <p className="text-[10px] text-violet-100/70 leading-relaxed">
            Automated corpus-generation tools. Results are appended to the
            corpus below and immediately ready to train on.
          </p>

          {/* Wikipedia Knowledge Fetcher */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5">
              <Globe className="size-3.5 text-sky-300" />
              <span className="text-[11px] font-semibold text-slate-200">
                Wikipedia Knowledge Fetcher
              </span>
            </div>
            <p className="text-[10px] text-slate-500">
              Fetches a Wikipedia summary and formats it as a{" "}
              <code className="font-mono">User:/Bot:</code> Q&A pair.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={wikiTopic}
                onChange={(e) => {
                  setWikiTopic(e.target.value);
                  setWikiStatus("idle");
                  setWikiMsg("");
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") fetchWikipedia();
                }}
                placeholder="Topic (e.g. Quantum Mechanics)"
                className="flex-1 min-h-[36px] rounded-lg border border-slate-700 bg-slate-950/60 px-2.5 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40"
              />
              <button
                type="button"
                onClick={fetchWikipedia}
                disabled={!wikiTopic.trim() || wikiStatus === "loading"}
                className="min-h-[36px] px-3 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 border border-sky-400/40 text-sky-100 text-xs font-semibold inline-flex items-center justify-center gap-1.5 transition disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
              >
                {wikiStatus === "loading" ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <Globe className="size-3.5" />
                )}
                {wikiStatus === "loading" ? "Fetching…" : "Fetch & Learn"}
              </button>
            </div>

            {wikiStatus === "success" && (
              <div className="flex items-start gap-1.5 text-[10px] text-emerald-300">
                <CheckCircle2 className="size-3.5 shrink-0 mt-0.5" />
                {wikiMsg}
              </div>
            )}
            {wikiStatus === "error" && (
              <div className="flex items-start gap-1.5 text-[10px] text-rose-300">
                <AlertCircle className="size-3.5 shrink-0 mt-0.5" />
                {wikiMsg}
              </div>
            )}
          </div>

          {/* Divider */}
          <div className="border-t border-violet-500/20" />

          {/* Bulk Text Formatter */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5">
              <FileCode2 className="size-3.5 text-fuchsia-300" />
              <span className="text-[11px] font-semibold text-slate-200">
                Bulk Text Formatter
              </span>
            </div>
            <p className="text-[10px] text-slate-500">
              Upload a raw <code className="font-mono">.txt</code> file.
              Paragraphs are split and wrapped in alternating{" "}
              <code className="font-mono">User:/Bot:</code> turns automatically.
            </p>
            <button
              type="button"
              onClick={() => bulkFileInputRef.current?.click()}
              className="w-full min-h-[38px] rounded-lg bg-fuchsia-500/20 hover:bg-fuchsia-500/30 border border-fuchsia-400/40 text-fuchsia-100 text-xs font-semibold inline-flex items-center justify-center gap-1.5 transition"
            >
              <FileCode2 className="size-3.5" />
              Upload Raw Text File
            </button>
            <input
              ref={bulkFileInputRef}
              type="file"
              accept=".txt,text/plain"
              className="hidden"
              onChange={(e) => handleBulkFile(e.target.files)}
            />

            {bulkFormatInfo && (
              <div className="flex items-start gap-1.5 text-[10px] text-emerald-300">
                <CheckCircle2 className="size-3.5 shrink-0 mt-0.5" />
                <span>
                  <span className="font-semibold">{bulkFormatInfo.name}</span>{" "}
                  — {bulkFormatInfo.paragraphs} paragraph
                  {bulkFormatInfo.paragraphs !== 1 ? "s" : ""} formatted and
                  appended.
                </span>
              </div>
            )}
            {bulkError && (
              <div className="flex items-start gap-1.5 text-[10px] text-rose-300">
                <AlertCircle className="size-3.5 shrink-0 mt-0.5" />
                {bulkError}
              </div>
            )}
          </div>
        </div>

        {/* Manual fallback */}
        <div className="space-y-1.5">
          <label className="text-[10px] uppercase tracking-wider text-slate-500">
            Paste / edit text
          </label>
          <textarea
            value={config.corpus}
            onChange={(e) => {
              const v = e.target.value;
              const truncated =
                v.length > MAX_CORPUS_BYTES ? v.slice(0, MAX_CORPUS_BYTES) : v;
              onChange({ ...config, corpus: truncated });
            }}
            rows={6}
            className="w-full min-h-[140px] rounded-xl border border-slate-700 bg-slate-900/60 p-3 text-sm font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 resize-none"
            placeholder={
              isWord
                ? "User: hello\nBot: hi how can i help"
                : "hello world. hello friend…"
            }
          />
        </div>

        <button
          onClick={onApply}
          disabled={overBudget || config.corpus.trim().length < 8}
          className="w-full min-h-[44px] rounded-xl bg-gradient-to-br from-sky-500 to-emerald-500 text-slate-900 font-semibold text-sm flex items-center justify-center gap-1.5 shadow-lg shadow-sky-500/20 hover:brightness-110 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Sparkles className="size-4" />
          Rebuild Model
        </button>
      </Card>
    </div>
  );
}
