import { Cpu, Zap } from "lucide-react";

export type RuntimeMode = "lite" | "pro";

interface Props {
  mode: RuntimeMode;
  onChange: (m: RuntimeMode) => void;
}

export function RuntimeToggle({ mode, onChange }: Props) {
  return (
    <div className="relative inline-flex rounded-xl border border-slate-700 bg-slate-900/70 p-1 backdrop-blur-md">
      <div
        className={`absolute top-1 bottom-1 rounded-lg transition-all duration-300 ease-out shadow-lg ${
          mode === "lite"
            ? "bg-gradient-to-br from-slate-500 to-slate-600 shadow-slate-500/20"
            : "bg-gradient-to-br from-fuchsia-500 to-indigo-500 shadow-fuchsia-500/30"
        }`}
        style={{
          width: "calc(50% - 4px)",
          left: mode === "lite" ? "4px" : "calc(50% + 0px)",
        }}
      />
      <button
        onClick={() => onChange("lite")}
        className={`relative z-10 flex items-center gap-1.5 px-3 py-1.5 min-h-[36px] rounded-lg text-xs font-semibold transition-colors ${
          mode === "lite" ? "text-slate-50" : "text-slate-300"
        }`}
        aria-pressed={mode === "lite"}
      >
        <Cpu className="size-3.5" />
        <span className="hidden sm:inline">Lite · CPU</span>
        <span className="sm:hidden">Lite</span>
      </button>
      <button
        onClick={() => onChange("pro")}
        className={`relative z-10 flex items-center gap-1.5 px-3 py-1.5 min-h-[36px] rounded-lg text-xs font-semibold transition-colors ${
          mode === "pro" ? "text-white" : "text-slate-300"
        }`}
        aria-pressed={mode === "pro"}
      >
        <Zap className="size-3.5" />
        <span className="hidden sm:inline">Pro · WebGPU</span>
        <span className="sm:hidden">Pro</span>
      </button>
    </div>
  );
}
