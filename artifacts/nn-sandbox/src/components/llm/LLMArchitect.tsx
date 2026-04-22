import { Cpu, Sparkles } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";

export interface LLMConfig {
  baseModel: string;
  systemPrompt: string;
  temperature: number;
}

interface Props {
  config: LLMConfig;
  onChange: (next: LLMConfig) => void;
}

const MODELS = [
  { id: "smollm-135m", label: "SmolLM-135M", note: "Fastest · 135M params" },
  { id: "smollm-360m", label: "SmolLM-360M", note: "Balanced · 360M params" },
  { id: "gemma-2b-q4", label: "Gemma-2B (q4)", note: "Best quality · ~1.5GB" },
  { id: "qwen-0.5b", label: "Qwen 2.5-0.5B", note: "Tiny · multilingual" },
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

export function LLMArchitect({ config, onChange }: Props) {
  const selected = MODELS.find((m) => m.id === config.baseModel) ?? MODELS[0];
  return (
    <div className="space-y-4">
      <Card className="p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Cpu className="size-4 text-sky-400" />
          <span className="text-sm font-semibold text-slate-100">
            Base Model
          </span>
        </div>

        <Select
          value={config.baseModel}
          onValueChange={(v) => onChange({ ...config, baseModel: v })}
        >
          <SelectTrigger className="min-h-[44px] rounded-xl">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {MODELS.map((m) => (
              <SelectItem key={m.id} value={m.id}>
                {m.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-3">
          <div className="text-xs text-slate-400">{selected.note}</div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-300 border border-sky-500/30">
              WebGPU
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
              LoRA fine-tunable
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/30">
              On-device
            </span>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between">
            <span className="text-xs text-slate-400">Temperature</span>
            <span className="text-xs tabular-nums text-slate-200">
              {config.temperature.toFixed(2)}
            </span>
          </div>
          <Slider
            min={0}
            max={1.5}
            step={0.01}
            value={[config.temperature]}
            onValueChange={([v]) => onChange({ ...config, temperature: v })}
            className="py-2"
          />
        </div>
      </Card>

      <Card className="p-5 space-y-3">
        <div className="flex items-center gap-2">
          <Sparkles className="size-4 text-amber-400" />
          <span className="text-sm font-semibold text-slate-100">
            System Prompt
          </span>
        </div>
        <p className="text-[11px] text-slate-400">
          Shape your assistant's personality and behavior.
        </p>
        <textarea
          value={config.systemPrompt}
          onChange={(e) =>
            onChange({ ...config, systemPrompt: e.target.value })
          }
          rows={6}
          className="w-full min-h-[140px] rounded-xl border border-slate-700 bg-slate-900/60 p-3 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 resize-none"
          placeholder="You are a helpful assistant…"
        />
      </Card>
    </div>
  );
}

export { MODELS };
