import { Cpu, Sparkles, BookOpen } from "lucide-react";
import { Slider } from "@/components/ui/slider";

export interface LLMConfig {
  corpus: string;
  contextSize: number;
  hiddenSize: number;
  learningRate: number;
  temperature: number;
}

interface Props {
  config: LLMConfig;
  onChange: (next: LLMConfig) => void;
  onApply: () => void;
  paramCount: number;
  vocabSize: number;
  maxParams: number;
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

export function LLMArchitect({
  config,
  onChange,
  onApply,
  paramCount,
  vocabSize,
  maxParams,
}: Props) {
  const overBudget = paramCount > maxParams;

  return (
    <div className="space-y-4">
      <Card className="p-5 space-y-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu className="size-4 text-sky-400" />
            <span className="text-sm font-semibold text-slate-100">
              Char-LM Architecture
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
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Context Window</span>
            <span className="text-xs tabular-nums text-slate-200">
              {config.contextSize} chars
            </span>
          </div>
          <Slider
            min={1}
            max={6}
            step={1}
            value={[config.contextSize]}
            onValueChange={([v]) => onChange({ ...config, contextSize: v })}
            className="py-2"
          />
          <p className="text-[10px] text-slate-500">
            How many letters the model sees before predicting the next one.
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
            max={64}
            step={1}
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
            vocab: {vocabSize}
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
        <div className="flex items-center gap-2">
          <BookOpen className="size-4 text-amber-400" />
          <span className="text-sm font-semibold text-slate-100">
            Training Corpus
          </span>
        </div>
        <p className="text-[11px] text-slate-400">
          The text the model learns to imitate, character by character.
        </p>
        <textarea
          value={config.corpus}
          onChange={(e) => onChange({ ...config, corpus: e.target.value })}
          rows={6}
          className="w-full min-h-[140px] rounded-xl border border-slate-700 bg-slate-900/60 p-3 text-sm font-mono text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 resize-none"
          placeholder="hello world. hello friend…"
        />
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
