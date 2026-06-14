import { useEffect, useRef, useState } from "react";
import {
  Cpu,
  Sparkles,
  BookOpen,
  Upload,
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

const DATASET_SEPARATOR = `\n${Array(50).fill(PAD_TOKEN).join(" ")}\n`;

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
  datasets: Dataset[];
  onDatasetsChange: (next: Dataset[]) => void;
  /** When provided, only the specified section is rendered. */
  section?: "arch" | "dataset" | "dataset-tools";
}

export const MAX_CORPUS_BYTES = 1_000_000;

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
    <div className={`bg-[#101010] rounded-2xl border border-white/[0.06] ${className}`}>
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
  section,
}: Props) {
  const overBudget = paramCount > maxParams;

  const configRef = useRef(config);
  useEffect(() => { configRef.current = config; });
  const onChangeRef = useRef(onChange);
  useEffect(() => { onChangeRef.current = onChange; });
  const lastCombinedRef = useRef<string | null>(null);

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
    onDatasetsChange([...datasets, { id: genId(), name, text: trimmed, active: true }]);
  };

  const updateDataset = (id: number, patch: Partial<Dataset>) => {
    onDatasetsChange(datasets.map((d) => (d.id === id ? { ...d, ...patch } : d)));
  };

  const removeDataset = (id: number) => {
    onDatasetsChange(datasets.filter((d) => d.id !== id));
  };

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const bulkFileInputRef = useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadInfo, setUploadInfo] = useState<{ name: string; paragraphs: number } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const processTextUpload = (file: File) => {
    setUploadError(null);
    setUploadInfo(null);
    if (!/\.txt$/i.test(file.name) && !file.type.startsWith("text/")) {
      setUploadError("Only .txt files are supported.");
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => setUploadError("Could not read that file.");
    reader.onload = (e) => {
      const rawText = (e.target?.result as string) ?? "";
      const paragraphs = rawText.split(/\n\s*\n/).filter((p) => p.trim().length > 0);
      if (paragraphs.length === 0) {
        setUploadError("No usable paragraphs found in that file.");
        return;
      }
      const formattedText = paragraphs
        .map((p) => {
          const trimmed = p.trim();
          return trimmed.startsWith("User:") ? trimmed : `User: tell me more\nBot: ${trimmed}`;
        })
        .join("\n\n");
      addDataset(`Import: ${file.name}`, formattedText);
      setUploadInfo({ name: file.name, paragraphs: paragraphs.length });
    };
    reader.readAsText(file);
  };

  const handleFileInputChange = (input: HTMLInputElement | null, files: FileList | null) => {
    const f = files?.[0];
    if (f) processTextUpload(f);
    if (input) input.value = "";
  };

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

  const [wikiTopic, setWikiTopic] = useState("");
  const [wikiStatus, setWikiStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
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

  const isWord = config.tokenization === "word";
  const tokenLabel = isWord ? "tokens" : "chars";
  const corpusBytes = new Blob([config.corpus]).size;

  const showArch = !section || section === "arch";
  const showDataset = !section || section === "dataset" || section === "dataset-tools";
  const showDatasetList = !section || section === "dataset";

  return (
    <div className="space-y-4">

      {/* ── Architecture Card ──────────────────────────────────────────────── */}
      {showArch && (
        <Card className="p-5 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="size-4 text-[#0A84FF]" />
              <span className="text-sm font-semibold text-white/88">Architecture</span>
            </div>
            <span className={`text-[11px] tabular-nums font-mono ${overBudget ? "text-red-400" : "text-white/30"}`}>
              {paramCount.toLocaleString()} / {maxParams.toLocaleString()}
            </span>
          </div>

          {/* Tokenization */}
          <div className="space-y-2">
            <span className="text-[11px] text-white/35 font-medium">Tokenization</span>
            <div className="grid grid-cols-2 gap-1 rounded-xl bg-[#0a0a0a] border border-white/[0.05] p-1">
              <button
                onClick={() => onChange({ ...config, tokenization: "char" })}
                className={`h-9 rounded-lg text-[11px] font-semibold inline-flex items-center justify-center gap-1.5 transition-all ${
                  !isWord ? "bg-[#0A84FF]/15 text-[#0A84FF] border border-[#0A84FF]/20" : "text-white/30 hover:text-white/60"
                }`}
              >
                <Type className="size-3.5" />
                Char-Level
              </button>
              <button
                onClick={() => onChange({ ...config, tokenization: "word" })}
                className={`h-9 rounded-lg text-[11px] font-semibold inline-flex items-center justify-center gap-1.5 transition-all ${
                  isWord ? "bg-[#30D158]/10 text-[#30D158] border border-[#30D158]/20" : "text-white/30 hover:text-white/60"
                }`}
              >
                <WholeWord className="size-3.5" />
                Word-Level
              </button>
            </div>
            <p className="text-[10px] text-white/22 leading-relaxed">
              {isWord
                ? "Vocabulary from whole words. Better for sentences, larger vocab."
                : "Vocabulary from individual characters. Tiny vocab, learns spelling."}
            </p>
          </div>

          {/* Context Window */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-[11px] text-white/35 font-medium">Context Window</span>
              <span className="text-[11px] tabular-nums text-white/65 font-mono">{config.contextSize} {tokenLabel}</span>
            </div>
            <Slider min={1} max={512} step={1} value={[config.contextSize]} onValueChange={([v]) => onChange({ ...config, contextSize: v })} className="py-2" />
          </div>

          {/* Hidden Neurons */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-[11px] text-white/35 font-medium">Hidden Neurons</span>
              <span className="text-[11px] tabular-nums text-white/65 font-mono">{config.hiddenSize}</span>
            </div>
            <Slider min={4} max={512} step={4} value={[config.hiddenSize]} onValueChange={([v]) => onChange({ ...config, hiddenSize: v })} className="py-2" />
          </div>

          {/* Learning Rate */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-[11px] text-white/35 font-medium">Learning Rate</span>
              <span className="text-[11px] tabular-nums text-white/65 font-mono">{config.learningRate.toFixed(3)}</span>
            </div>
            <Slider min={0.005} max={0.5} step={0.005} value={[config.learningRate]} onValueChange={([v]) => onChange({ ...config, learningRate: v })} className="py-2" />
          </div>

          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-[11px] text-white/35 font-medium">Sampling Temperature</span>
              <span className="text-[11px] tabular-nums text-white/65 font-mono">{config.temperature.toFixed(2)}</span>
            </div>
            <Slider min={0.1} max={1.5} step={0.05} value={[config.temperature]} onValueChange={([v]) => onChange({ ...config, temperature: v })} className="py-2" />
            <p className="text-[10px] text-white/22 leading-relaxed">Lower = focused. Higher = creative.</p>
          </div>

          {/* Top-K */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-[11px] text-white/35 font-medium inline-flex items-center gap-1.5">
                <Filter className="size-3 text-fuchsia-400" />
                Top-K Sampling
              </span>
              <span className="text-[11px] tabular-nums text-white/65 font-mono">{config.topK}</span>
            </div>
            <Slider min={1} max={40} step={1} value={[config.topK]} onValueChange={([v]) => onChange({ ...config, topK: v })} className="py-2" />
          </div>

          {/* System Prompt */}
          <div className="space-y-2">
            <span className="text-[11px] text-white/35 font-medium inline-flex items-center gap-1.5">
              <MessageSquare className="size-3 text-sky-400" />
              System Prompt
            </span>
            <input
              type="text"
              value={config.systemPrompt}
              onChange={(e) => onChange({ ...config, systemPrompt: e.target.value })}
              placeholder="Bot: I am a helpful AI."
              className="w-full h-10 rounded-xl border border-white/[0.07] bg-[#0a0a0a] px-3 text-[11px] font-mono text-white/82 placeholder:text-white/18 focus:outline-none focus:border-[#0A84FF]/35 transition-colors"
            />
          </div>

          {/* Badges */}
          <div className="rounded-xl border border-white/[0.05] bg-[#0a0a0a] p-3 flex flex-wrap gap-1.5">
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20">vocab: {vocabSize.toLocaleString()}</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-fuchsia-500/10 text-fuchsia-400 border border-fuchsia-500/20">{isWord ? "word-level" : "char-level"}</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">softmax · cross-entropy</span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">pure TS · in-worker</span>
          </div>

          <button
            onClick={onApply}
            disabled={overBudget}
            className="w-full h-10 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-[12px] font-semibold text-white/60 hover:text-white/90 transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            <Sparkles className="size-3.5" />
            Apply & Rebuild Model
          </button>
        </Card>
      )}

      {/* ── Training Corpus Card ───────────────────────────────────────────── */}
      {showDataset && (
        <Card className="p-5 space-y-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <BookOpen className="size-4 text-amber-400" />
              <span className="text-sm font-semibold text-white/88">Training Corpus</span>
            </div>
            <span className="text-[10px] text-white/25 tabular-nums font-mono">
              {formatBytes(corpusBytes)} / {formatBytes(MAX_CORPUS_BYTES)}
            </span>
          </div>

          <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 flex gap-2.5">
            <Lightbulb className="size-4 text-amber-300 shrink-0 mt-0.5" />
            <div className="text-[11px] text-amber-100/80 leading-relaxed">
              <span className="font-semibold text-amber-200">Pro tip:</span> Format as{" "}
              <code className="mx-1 px-1.5 py-0.5 rounded bg-black/40 text-amber-200 font-mono text-[10px]">User: … Bot: …</code>
              pairs for Q&A bots.
            </div>
          </div>

          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files?.[0]; if (f) processTextUpload(f); }}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInputRef.current?.click(); } }}
            className={`rounded-xl border-2 border-dashed p-4 text-center cursor-pointer transition-all select-none min-h-[90px] flex flex-col items-center justify-center gap-1.5 ${
              dragOver ? "border-[#0A84FF]/50 bg-[#0A84FF]/5" : "border-white/[0.07] hover:border-white/[0.14] hover:bg-white/[0.02]"
            }`}
          >
            <Upload className="size-4 text-sky-400" />
            <div className="text-[11px] font-semibold text-white/60">Drop a <code className="text-sky-400">.txt</code> file</div>
            <div className="text-[10px] text-white/25">Paragraphs → User/Bot turns · max {formatBytes(MAX_CORPUS_BYTES)}</div>
            <input ref={fileInputRef} type="file" accept=".txt,text/plain" className="hidden" onChange={(e) => handleFileInputChange(fileInputRef.current, e.target.files)} />
          </div>

          {uploadError && (
            <div className="flex items-center gap-2 text-[11px] text-red-400 bg-red-400/8 border border-red-400/15 rounded-xl px-3 py-2">
              <AlertCircle className="size-3.5 shrink-0" />
              {uploadError}
            </div>
          )}
          {uploadInfo && (
            <div className="flex items-center gap-2 text-[11px] text-emerald-400 bg-emerald-400/8 border border-emerald-400/15 rounded-xl px-3 py-2">
              <CheckCircle2 className="size-3.5 shrink-0" />
              Added {uploadInfo.paragraphs} paragraphs from {uploadInfo.name}
            </div>
          )}

          {/* Active Datasets — hidden in dataset-tools mode (shown in explorer tab) */}
          {datasets.length > 0 && showDatasetList && (
            <div className="space-y-1.5">
              <span className="text-[11px] text-white/30 font-medium flex items-center gap-1.5">
                <Database className="size-3" />
                Active Datasets ({datasets.filter((d) => d.active).length}/{datasets.length})
              </span>
              {datasets.map((d) => (
                <DatasetRow
                  key={d.id}
                  dataset={d}
                  onToggle={() => updateDataset(d.id, { active: !d.active })}
                  onRemove={() => removeDataset(d.id)}
                />
              ))}
            </div>
          )}

          {/* Fact Teacher */}
          <div className="space-y-2.5">
            <div className="flex items-center gap-2">
              <GraduationCap className="size-4 text-sky-400" />
              <span className="text-[11px] font-semibold text-white/65">Fact Teacher</span>
            </div>
            <input
              type="text"
              value={factQ}
              onChange={(e) => setFactQ(e.target.value)}
              placeholder="Question topic (e.g. the capital of France)"
              className="w-full h-9 rounded-xl border border-white/[0.06] bg-[#0a0a0a] px-3 text-[11px] text-white/75 placeholder:text-white/18 focus:outline-none focus:border-[#0A84FF]/35 transition-colors"
            />
            <input
              type="text"
              value={factA}
              onChange={(e) => setFactA(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") addFactVariations(); }}
              placeholder="Answer (e.g. Paris is the capital of France)"
              className="w-full h-9 rounded-xl border border-white/[0.06] bg-[#0a0a0a] px-3 text-[11px] text-white/75 placeholder:text-white/18 focus:outline-none focus:border-[#0A84FF]/35 transition-colors"
            />
            <button
              onClick={addFactVariations}
              disabled={!factQ.trim() || !factA.trim()}
              className="w-full h-9 rounded-xl bg-[#0A84FF]/10 hover:bg-[#0A84FF]/20 border border-[#0A84FF]/20 text-[#0A84FF] text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Plus className="size-3.5" />
              Add Fact
            </button>
          </div>

          {/* Bulk formatter */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Hammer className="size-3.5 text-orange-400" />
              <span className="text-[11px] font-semibold text-white/65">Bulk Text Formatter</span>
            </div>
            <p className="text-[10px] text-white/22 leading-relaxed">Upload any prose .txt — paragraphs are auto-converted into User/Bot turns.</p>
            <button
              onClick={() => bulkFileInputRef.current?.click()}
              className="w-full h-9 rounded-xl border border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.05] text-[11px] text-white/45 hover:text-white/70 font-medium flex items-center justify-center gap-2 transition-all"
            >
              <FileCode2 className="size-3.5" />
              Choose .txt file
            </button>
            <input ref={bulkFileInputRef} type="file" accept=".txt,text/plain" className="hidden" onChange={(e) => handleFileInputChange(bulkFileInputRef.current, e.target.files)} />
          </div>

          {/* Wikipedia */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Globe className="size-3.5 text-emerald-400" />
              <span className="text-[11px] font-semibold text-white/65">Wikipedia Knowledge</span>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={wikiTopic}
                onChange={(e) => setWikiTopic(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") fetchWikipedia(); }}
                placeholder="Topic (e.g. Black holes)"
                className="flex-1 h-9 rounded-xl border border-white/[0.06] bg-[#0a0a0a] px-3 text-[11px] text-white/75 placeholder:text-white/18 focus:outline-none focus:border-[#0A84FF]/35 transition-colors"
              />
              <button
                onClick={fetchWikipedia}
                disabled={!wikiTopic.trim() || wikiStatus === "loading"}
                className="h-9 px-4 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/20 text-emerald-400 text-[11px] font-semibold flex items-center gap-1.5 transition-all disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
              >
                {wikiStatus === "loading" ? <Loader2 className="size-3.5 animate-spin" /> : <Globe className="size-3.5" />}
                Fetch
              </button>
            </div>
            {wikiMsg && (
              <div className={`flex items-center gap-2 text-[10px] rounded-xl px-3 py-2 border ${
                wikiStatus === "error"
                  ? "text-red-400 bg-red-400/8 border-red-400/15"
                  : "text-emerald-400 bg-emerald-400/8 border-emerald-400/15"
              }`}>
                {wikiStatus === "error" ? <AlertCircle className="size-3 shrink-0" /> : <CheckCircle2 className="size-3 shrink-0" />}
                {wikiMsg}
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}

function DatasetRow({
  dataset,
  onToggle,
  onRemove,
}: {
  dataset: Dataset;
  onToggle: () => void;
  onRemove: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const bytes = new Blob([dataset.text]).size;

  return (
    <div className={`rounded-xl border transition-colors ${dataset.active ? "border-white/[0.07] bg-[#0f0f0f]" : "border-white/[0.04] bg-[#0a0a0a] opacity-50"}`}>
      <div className="flex items-center gap-2 px-3 py-2.5">
        <button
          onClick={onToggle}
          className={`size-3.5 rounded-full border-2 shrink-0 transition-all ${dataset.active ? "border-[#0A84FF] bg-[#0A84FF]" : "border-white/20 bg-transparent"}`}
        />
        <span className="flex-1 text-[11px] text-white/65 font-medium truncate">{dataset.name}</span>
        <span className="text-[9px] text-white/25 tabular-nums font-mono shrink-0">{formatBytes(bytes)}</span>
        <button onClick={() => setExpanded(!expanded)} className="size-5 flex items-center justify-center text-white/20 hover:text-white/50 transition-colors shrink-0">
          {expanded ? <EyeOff className="size-3" /> : <Eye className="size-3" />}
        </button>
        <button onClick={onRemove} className="size-5 flex items-center justify-center text-white/15 hover:text-red-400 transition-colors shrink-0">
          <Trash2 className="size-3" />
        </button>
      </div>
      {expanded && (
        <div className="px-3 pb-3">
          <pre className="text-[9px] font-mono text-white/30 whitespace-pre-wrap leading-relaxed max-h-28 overflow-y-auto bg-black/30 rounded-lg p-2">
            {dataset.text.slice(0, 800)}{dataset.text.length > 800 ? "\n…" : ""}
          </pre>
        </div>
      )}
    </div>
  );
}
