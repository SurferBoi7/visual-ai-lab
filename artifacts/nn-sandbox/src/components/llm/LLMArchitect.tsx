import { useEffect, useRef, useState } from "react";
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
  Trash2,
  Database,
  Eye,
  EyeOff,
} from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { PAD_TOKEN, EOS_TOKEN } from "@/lib/textnet";

export type Tokenization = "char" | "word";

// Wall of <PAD> tokens injected between datasets so the model treats each one
// as an independent document. Every <PAD> here is recognised by the tokenizer
// as the single special padding id (not split character-by-character), so a
// run of 20 of them gives the rolling context window enough room to fully
// flush the previous topic before the next dataset starts. Newlines on either
// side keep the separator visually obvious in the corpus textarea too.
const DATASET_SEPARATOR = `\n${Array(20).fill(PAD_TOKEN).join(" ")}\n`;

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

// A single user-managed dataset row in the Dataset Manager. The `id` is a
// UI-only handle (used as React key and for update/remove targeting); the
// persisted shape in storage.ts intentionally omits it.
export interface Dataset {
  id: number;
  name: string;
  text: string;
  active: boolean;
}

interface Props {
  config: LLMConfig;
  onChange: (next: LLMConfig) => void;
  onApply: () => void;
  paramCount: number;
  vocabSize: number;
  maxParams: number;
  // Controlled dataset state. `datasets` is the SINGLE SOURCE OF TRUTH for the
  // user's individual files; the flattened `config.corpus` string is derived
  // FROM datasets (never the other way around) and pushed to the parent via
  // `onChange`. Lifted to App.tsx so model loads can selectively restore the
  // datasets array without ever overwriting it from a flattened corpus blob.
  datasets: Dataset[];
  onDatasetsChange: (next: Dataset[]) => void;
}

export const MAX_CORPUS_BYTES = 1_000_000;

// Truncate a Wikipedia extract to the first N sentences.
function truncateToSentences(text: string, n: number): string {
  const sentences = text.split(".").filter((s) => s.trim().length > 0);
  return sentences.slice(0, n).join(".").trim() + (sentences.length > 0 ? "." : "");
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

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

let nextId = 2;
function genId() {
  return nextId++;
}

export function LLMArchitect({
  config,
  onChange,
  onApply,
  paramCount,
  vocabSize,
  maxParams,
  datasets,
  onDatasetsChange,
}: Props) {
  const overBudget = paramCount > maxParams;

  // Ref always reflects the latest config so effects don't capture stale values.
  const configRef = useRef(config);
  useEffect(() => {
    configRef.current = config;
  });
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  });

  // Track the last combined corpus we pushed outward so we can avoid bouncing
  // an unchanged value back into onChange every render. Note: this ref only
  // serves as a write-side dedupe — datasets remain the source of truth and
  // are NEVER reconstructed from `config.corpus`.
  const lastCombinedRef = useRef<string | null>(null);

  // Whenever datasets change, aggregate and push to parent. Two structural
  // markers get physically injected into the corpus here:
  //
  // 1. <EOS> — appended to the END of every active dataset BEFORE the PAD
  //    wall, so the model learns "when the factual paragraph finishes, emit
  //    <EOS>". The inference loop in text.worker.ts watches for this token
  //    and breaks immediately, killing infinite generation loops.
  //
  // 2. <PAD> wall — placed BETWEEN every active dataset and as a PREFIX to
  //    the very first one, so each dataset reads as an independent document
  //    instead of one giant continuous string. This stops cross-document
  //    contamination — e.g. "Bot:" from dataset A flowing straight into the
  //    first sentence of dataset B.
  useEffect(() => {
    const active = datasets
      .filter((d) => d.active)
      .map((d) => `${d.text}\n${EOS_TOKEN}`);
    const combined = (
      active.length === 0
        ? ""
        : DATASET_SEPARATOR + active.join(DATASET_SEPARATOR)
    ).slice(0, MAX_CORPUS_BYTES);
    if (combined !== lastCombinedRef.current) {
      lastCombinedRef.current = combined;
      onChangeRef.current({ ...configRef.current, corpus: combined });
    }
  }, [datasets]);

  const addDataset = (name: string, text: string) => {
    const trimmed = text.slice(0, MAX_CORPUS_BYTES);
    onDatasetsChange([
      ...datasets,
      { id: genId(), name, text: trimmed, active: true },
    ]);
  };

  const updateDataset = (id: number, patch: Partial<Dataset>) => {
    onDatasetsChange(
      datasets.map((d) => (d.id === id ? { ...d, ...patch } : d)),
    );
  };

  const removeDataset = (id: number) => {
    onDatasetsChange(datasets.filter((d) => d.id !== id));
  };

  // ── Formatted .txt upload (replaces base dataset) ─────────────────────────
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadInfo, setUploadInfo] = useState<{
    name: string;
    bytes: number;
    truncated: boolean;
  } | null>(null);
  const [readError, setReadError] = useState<string | null>(null);

  const handleFormattedFile = (files: FileList | null) => {
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
      addDataset(`Uploaded: ${f.name}`, text);
    };
    reader.readAsText(f);
  };

  // ── Fact Teacher ──────────────────────────────────────────────────────────
  const [factQ, setFactQ] = useState("");
  const [factA, setFactA] = useState("");

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
    addDataset(`Fact: ${q}`, block);
    setFactQ("");
    setFactA("");
  };

  // ── Wikipedia Knowledge Fetcher ───────────────────────────────────────────
  const [wikiTopic, setWikiTopic] = useState("");
  const [wikiStatus, setWikiStatus] = useState<
    "idle" | "loading" | "success" | "error"
  >("idle");
  const [wikiMsg, setWikiMsg] = useState("");

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
      const rawExtract: string = data.extract ?? "";
      if (!rawExtract) {
        setWikiStatus("error");
        setWikiMsg("Wikipedia returned an empty summary for this topic.");
        return;
      }
      // Smart truncation: keep only the first 3 sentences to prevent
      // the model from looping over a very long context window.
      const extract = truncateToSentences(rawExtract, 3);
      const text = `User: tell me about ${topic}\nBot: ${extract}`;
      addDataset(`Wiki: ${topic}`, text);
      setWikiStatus("success");
      setWikiMsg(`Added Wikipedia summary for "${topic}" (3 sentences).`);
      setWikiTopic("");
    } catch {
      setWikiStatus("error");
      setWikiMsg("Network error — check your connection and try again.");
    }
  };

  // ── Bulk Text Formatter ───────────────────────────────────────────────────
  const bulkFileInputRef = useRef<HTMLInputElement | null>(null);
  const [bulkFormatInfo, setBulkFormatInfo] = useState<{
    name: string;
    paragraphs: number;
  } | null>(null);
  const [bulkError, setBulkError] = useState<string | null>(null);

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
    reader.onload = (e) => {
      // Hard-enforce the conversational transformation. Raw prose must NEVER
      // hit the dataset untouched — every paragraph becomes a User/Bot turn
      // so the resulting corpus matches the same Q&A shape the model is
      // trained to chat in.
      const rawText = (e.target?.result as string) ?? "";
      // `\n\s*\n` matches a blank line that may contain trailing whitespace
      // (Windows `\r\n\r\n`, soft-wrapped editors, etc.) — strictly broader
      // than `\n\n+` so real-world `.txt` files split into paragraphs cleanly.
      const paragraphs = rawText
        .split(/\n\s*\n/)
        .filter((p) => p.trim().length > 0);
      if (paragraphs.length === 0) {
        setBulkError("No usable paragraphs found in that file.");
        return;
      }
      const formattedText = paragraphs
        .map((p) => `User: tell me more\nBot: ${p.trim()}`)
        .join("\n\n");
      addDataset(`Import: ${f.name}`, formattedText);
      setBulkFormatInfo({ name: f.name, paragraphs: paragraphs.length });
      if (bulkFileInputRef.current) bulkFileInputRef.current.value = "";
    };
    reader.readAsText(f);
  };

  // ── Derived values ────────────────────────────────────────────────────────
  const isWord = config.tokenization === "word";
  const tokenLabel = isWord ? "tokens" : "chars";
  const corpusBytes = new Blob([config.corpus]).size;
  const totalDatasetBytes = datasets
    .filter((d) => d.active)
    .reduce((s, d) => s + new Blob([d.text]).size, 0);

  return (
    <div className="space-y-4">
      {/* ── Architecture Card ─────────────────────────────────────────────── */}
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
            <span className="text-xs text-slate-400">Sampling Temperature</span>
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
            Only the {config.topK} most-likely next tokens are considered.
            Lower = safer & more on-topic.
          </p>
        </div>

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

      {/* ── Training Corpus Card ──────────────────────────────────────────── */}
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

        {/* Formatted .txt drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFormattedFile(e.dataTransfer.files); }}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInputRef.current?.click(); } }}
          className={`rounded-xl border-2 border-dashed p-4 text-center cursor-pointer transition select-none min-h-[100px] flex flex-col items-center justify-center gap-1.5 ${
            dragOver
              ? "border-sky-400 bg-sky-500/10"
              : "border-slate-700 hover:border-slate-500 hover:bg-slate-900/60"
          }`}
        >
          <Upload className="size-5 text-sky-300" />
          <div className="text-xs font-semibold text-slate-200">
            Drop a formatted <code className="text-sky-300">.txt</code> corpus here
          </div>
          <div className="text-[10px] text-slate-500">
            Added as a new dataset · max {formatBytes(MAX_CORPUS_BYTES)}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,text/plain"
            className="hidden"
            onChange={(e) => handleFormattedFile(e.target.files)}
          />
        </div>

        {uploadInfo && (
          <div className="flex items-start gap-2 rounded-xl border border-slate-700 bg-slate-900/60 px-3 py-2">
            <FileText className="size-3.5 text-emerald-300 mt-0.5 shrink-0" />
            <div className="min-w-0 flex-1">
              <div className="text-[11px] text-slate-200 truncate">{uploadInfo.name}</div>
              <div className="text-[10px] text-slate-500">
                {formatBytes(uploadInfo.bytes)}
                {uploadInfo.truncated && (
                  <span className="text-amber-300"> · truncated to {formatBytes(MAX_CORPUS_BYTES)}</span>
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
            <span className="text-xs font-semibold text-emerald-100">Fact Teacher</span>
          </div>
          <p className="text-[10px] text-emerald-100/70 leading-relaxed">
            Teach the bot one fact in 5 phrasings — added as a new dataset.
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
            onKeyDown={(e) => { if (e.key === "Enter") addFactVariations(); }}
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

        {/* Data Forge */}
        <div className="rounded-xl border border-violet-500/30 bg-violet-500/5 p-3 space-y-4">
          <div className="flex items-center gap-2">
            <Hammer className="size-4 text-violet-300" />
            <span className="text-xs font-semibold text-violet-100">Data Forge</span>
          </div>
          <p className="text-[10px] text-violet-100/70 leading-relaxed">
            Each tool creates a new dataset entry in the library below.
          </p>

          {/* Wikipedia */}
          <div className="space-y-2">
            <div className="flex items-center gap-1.5">
              <Globe className="size-3.5 text-sky-300" />
              <span className="text-[11px] font-semibold text-slate-200">
                Wikipedia Knowledge Fetcher
              </span>
              <span className="ml-auto text-[9px] px-1.5 py-0.5 rounded-full bg-sky-500/15 text-sky-300 border border-sky-500/25">
                3-sentence summary
              </span>
            </div>
            <p className="text-[10px] text-slate-500">
              Fetches a concise Wikipedia summary and creates a{" "}
              <code className="font-mono">User:/Bot:</code> dataset entry.
            </p>
            <div className="flex gap-2">
              <input
                type="text"
                value={wikiTopic}
                onChange={(e) => { setWikiTopic(e.target.value); setWikiStatus("idle"); setWikiMsg(""); }}
                onKeyDown={(e) => { if (e.key === "Enter") fetchWikipedia(); }}
                placeholder="Topic (e.g. Quantum Mechanics)"
                className="flex-1 min-h-[36px] rounded-lg border border-slate-700 bg-slate-950/60 px-2.5 py-1.5 text-xs font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40"
              />
              <button
                type="button"
                onClick={fetchWikipedia}
                disabled={!wikiTopic.trim() || wikiStatus === "loading"}
                className="min-h-[36px] px-3 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 border border-sky-400/40 text-sky-100 text-xs font-semibold inline-flex items-center justify-center gap-1.5 transition disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
              >
                {wikiStatus === "loading" ? <Loader2 className="size-3.5 animate-spin" /> : <Globe className="size-3.5" />}
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
              Paragraphs are wrapped in alternating{" "}
              <code className="font-mono">User:/Bot:</code> turns and saved as a new dataset.
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
                  {bulkFormatInfo.paragraphs !== 1 ? "s" : ""} formatted and added as a dataset.
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

        {/* ── Dataset Manager ───────────────────────────────────────────────── */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Database className="size-3.5 text-sky-300" />
              <span className="text-xs font-semibold text-slate-200">
                Corpus Library
              </span>
              <span className="text-[10px] text-slate-500 tabular-nums">
                ({datasets.filter((d) => d.active).length}/{datasets.length} active · {formatBytes(totalDatasetBytes)})
              </span>
            </div>
            <button
              type="button"
              onClick={() => addDataset("New Dataset", "")}
              className="min-h-[28px] px-2.5 rounded-lg bg-slate-700/60 hover:bg-slate-700 border border-slate-600 text-slate-300 text-[11px] font-semibold inline-flex items-center gap-1 transition"
            >
              <Plus className="size-3" />
              Add
            </button>
          </div>

          <div className="space-y-2">
            {datasets.map((ds) => (
              <div
                key={ds.id}
                className={`rounded-xl border transition ${
                  ds.active
                    ? "border-slate-600 bg-slate-900/60"
                    : "border-slate-700/50 bg-slate-900/30 opacity-60"
                }`}
              >
                {/* Dataset header */}
                <div className="flex items-center gap-2 px-3 py-2 border-b border-slate-700/50">
                  <div
                    className={`size-2 rounded-full shrink-0 ${
                      ds.active ? "bg-emerald-400" : "bg-slate-600"
                    }`}
                  />
                  <input
                    type="text"
                    value={ds.name}
                    onChange={(e) => updateDataset(ds.id, { name: e.target.value })}
                    className="flex-1 min-w-0 bg-transparent text-[11px] font-semibold text-slate-200 placeholder:text-slate-500 focus:outline-none"
                    placeholder="Dataset name"
                  />
                  <button
                    type="button"
                    onClick={() => updateDataset(ds.id, { active: !ds.active })}
                    title={ds.active ? "Deactivate" : "Activate"}
                    className={`shrink-0 size-6 rounded-md flex items-center justify-center transition ${
                      ds.active
                        ? "text-emerald-300 hover:bg-emerald-500/20"
                        : "text-slate-500 hover:bg-slate-700"
                    }`}
                  >
                    {ds.active ? (
                      <Eye className="size-3.5" />
                    ) : (
                      <EyeOff className="size-3.5" />
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => removeDataset(ds.id)}
                    title="Delete dataset"
                    className="shrink-0 size-6 rounded-md flex items-center justify-center text-slate-500 hover:text-rose-400 hover:bg-rose-500/10 transition"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </div>
                {/* Dataset text */}
                <textarea
                  value={ds.text}
                  onChange={(e) => {
                    const v = e.target.value.slice(0, MAX_CORPUS_BYTES);
                    updateDataset(ds.id, { text: v });
                  }}
                  rows={4}
                  placeholder="Paste or type corpus text here…"
                  className="w-full rounded-b-xl bg-transparent px-3 py-2 text-[11px] font-mono text-slate-300 placeholder:text-slate-600 focus:outline-none resize-none"
                />
              </div>
            ))}
          </div>
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
