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
  const [dragOver, setDragOver] = useState(false);
  const [uploadInfo, setUploadInfo] = useState<{
    name: string;
    bytes: number;
    truncated: boolean;
  } | null>(null);
  const [readError, setReadError] = useState<string | null>(null);

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

        {/* Manual fallback */}
        <div className="space-y-1.5">
          <label className="text-[10px] uppercase tracking-wider text-slate-500">
            Or paste / edit text
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
