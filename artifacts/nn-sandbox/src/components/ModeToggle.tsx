import { Network, MessageSquare } from "lucide-react";

export type AppMode = "mlp" | "llm";

interface Props {
  mode: AppMode;
  onChange: (m: AppMode) => void;
}

export function ModeToggle({ mode, onChange }: Props) {
  return (
    <div className="relative inline-flex rounded-xl border border-slate-700 bg-slate-900/70 p-1 backdrop-blur-md">
      <div
        className="absolute top-1 bottom-1 rounded-lg bg-gradient-to-br from-sky-500 to-emerald-500 transition-all duration-300 ease-out shadow-lg shadow-sky-500/20"
        style={{
          width: "calc(50% - 4px)",
          left: mode === "mlp" ? "4px" : "calc(50% + 0px)",
        }}
      />
      <button
        onClick={() => onChange("mlp")}
        className={`relative z-10 flex items-center gap-1.5 px-3 py-1.5 min-h-[36px] rounded-lg text-xs font-semibold transition-colors ${
          mode === "mlp" ? "text-slate-900" : "text-slate-300"
        }`}
      >
        <Network className="size-3.5" />
        <span className="hidden sm:inline">Visual MLP</span>
        <span className="sm:hidden">MLP</span>
      </button>
      <button
        onClick={() => onChange("llm")}
        className={`relative z-10 flex items-center gap-1.5 px-3 py-1.5 min-h-[36px] rounded-lg text-xs font-semibold transition-colors ${
          mode === "llm" ? "text-slate-900" : "text-slate-300"
        }`}
      >
        <MessageSquare className="size-3.5" />
        <span className="hidden sm:inline">Text LLM</span>
        <span className="sm:hidden">LLM</span>
      </button>
    </div>
  );
}
