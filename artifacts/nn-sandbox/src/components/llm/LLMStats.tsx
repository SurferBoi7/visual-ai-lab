import { Activity, Cpu, HardDrive, Zap, BookOpen } from "lucide-react";

interface Props {
  modelLabel: string;
  epoch: number;
  loss: number;
  paramCount: number;
  vocabSize: number;
  contextSize: number;
  hiddenSize: number;
  tokensPerSecond: number;
  trainedSamples: number;
  messageCount: number;
  correctionCount: number;
  liveSample: string;
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

export function LLMStats({
  modelLabel,
  epoch,
  loss,
  paramCount,
  vocabSize,
  contextSize,
  hiddenSize,
  tokensPerSecond,
  trainedSamples,
  messageCount,
  correctionCount,
  liveSample,
}: Props) {
  const memKB = (paramCount * 4) / 1024;
  const adapterKB = correctionCount * 0.4;

  return (
    <div className="space-y-4">
      <Card className="p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="size-4 text-emerald-400" />
            <span className="text-sm font-semibold text-slate-100">
              LLM Telemetry
            </span>
          </div>
          <span className="text-[11px] text-slate-400 truncate max-w-[160px]">
            {modelLabel}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          <Metric
            icon={<Zap className="size-3.5 text-amber-400" />}
            label="Tokens / sec"
            value={tokensPerSecond > 0 ? tokensPerSecond.toFixed(0) : "—"}
            sub="per training pass"
          />
          <Metric
            icon={<Cpu className="size-3.5 text-sky-400" />}
            label="Memory"
            value={`${memKB.toFixed(1)} KB`}
            sub={`${paramCount.toLocaleString()} params`}
          />
          <Metric
            icon={<HardDrive className="size-3.5 text-emerald-400" />}
            label="LoRA Adapter"
            value={`${adapterKB.toFixed(1)} KB`}
            sub={`${correctionCount} corrections`}
          />
          <Metric
            icon={<BookOpen className="size-3.5 text-fuchsia-400" />}
            label="Vocab"
            value={`${vocabSize}`}
            sub="unique chars"
          />
          <Metric
            icon={<Activity className="size-3.5 text-emerald-400" />}
            label="Epoch"
            value={epoch.toLocaleString()}
            sub={`${trainedSamples.toLocaleString()} steps`}
          />
          <Metric
            icon={<Activity className="size-3.5 text-rose-400" />}
            label="Loss"
            value={loss > 0 ? loss.toFixed(3) : "—"}
            sub="cross-entropy"
          />
        </div>

        <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-3 space-y-2">
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Architecture</span>
            <span className="tabular-nums text-slate-200">
              {contextSize}-char ctx → {hiddenSize}h → softmax({vocabSize})
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Conversation</span>
            <span className="tabular-nums text-slate-200">
              {messageCount} messages
            </span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-slate-400">Fine-tune signal</span>
            <span className="tabular-nums text-emerald-300">
              {correctionCount} pairs
            </span>
          </div>
          <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-sky-500 to-emerald-500 transition-all duration-500"
              style={{ width: `${Math.min(100, correctionCount * 10)}%` }}
            />
          </div>
        </div>
      </Card>

      <Card className="p-5 space-y-2">
        <div className="flex items-center gap-2">
          <Activity className="size-4 text-emerald-400" />
          <span className="text-sm font-semibold text-slate-100">
            Live Generation
          </span>
        </div>
        <p className="text-[11px] text-slate-400">
          Sampled every training tick from a random seed in your corpus.
        </p>
        <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-3 font-mono text-xs leading-relaxed text-emerald-200/90 min-h-[64px] whitespace-pre-wrap break-words">
          {liveSample || (
            <span className="text-slate-600">
              start training to see the model dream…
            </span>
          )}
        </div>
      </Card>
    </div>
  );
}

function Metric({
  icon,
  label,
  value,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/60 px-3 py-3">
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-400">
        {icon}
        {label}
      </div>
      <div className="text-base font-semibold text-slate-100 tabular-nums mt-1">
        {value}
      </div>
      <div className="text-[10px] text-slate-500 mt-0.5 truncate">{sub}</div>
    </div>
  );
}
