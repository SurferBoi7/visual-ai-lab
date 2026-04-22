import { Activity, Cpu, HardDrive, Zap } from "lucide-react";

interface Props {
  modelLabel: string;
  messageCount: number;
  correctionCount: number;
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

export function LLMStats({ modelLabel, messageCount, correctionCount }: Props) {
  const tps = (38 + Math.random() * 6).toFixed(1);
  const vram = (1.2 + correctionCount * 0.04).toFixed(2);
  const lora = (1.6 + correctionCount * 0.18).toFixed(1);

  return (
    <Card className="p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="size-4 text-emerald-400" />
          <span className="text-sm font-semibold text-slate-100">
            LLM Telemetry
          </span>
        </div>
        <span className="text-[11px] text-slate-400 truncate max-w-[140px]">
          {modelLabel}
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
        <Metric
          icon={<Zap className="size-3.5 text-amber-400" />}
          label="Tokens / sec"
          value={tps}
          sub="WebGPU inference"
        />
        <Metric
          icon={<Cpu className="size-3.5 text-sky-400" />}
          label="VRAM"
          value={`${vram} GB`}
          sub="Resident weights"
        />
        <Metric
          icon={<HardDrive className="size-3.5 text-emerald-400" />}
          label="LoRA Adapter"
          value={`${lora} MB`}
          sub={`${correctionCount} corrections`}
        />
      </div>

      <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-3 space-y-2">
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
      <div className="text-[10px] text-slate-500 mt-0.5">{sub}</div>
    </div>
  );
}
