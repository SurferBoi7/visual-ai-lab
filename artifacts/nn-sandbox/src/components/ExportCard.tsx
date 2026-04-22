import { Download, Code2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  onDownload: () => void;
  hasModel: boolean;
}

const SNIPPET = `// Load your trained model
const res = await fetch('weights.json');
const model = await res.json();

// Tiny inference helper (no deps)
function predict(input) {
  let a = input;
  for (let l = 0; l < model.weights.length; l++) {
    const W = model.weights[l];
    const b = model.biases[l];
    const z = W.map((row, j) =>
      row.reduce((s, w, i) => s + w * a[i], b[j])
    );
    const isLast = l === model.weights.length - 1;
    a = isLast
      ? z.map(v => 1 / (1 + Math.exp(-v))) // sigmoid
      : z.map(v => Math.tanh(v));          // hidden act
  }
  return a;
}

const prediction = predict([0.5, 0.8]);
console.log(prediction); // → [0.87] (Class B)`;

export function ExportCard({ onDownload, hasModel }: Props) {
  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-800/50 backdrop-blur-md p-5 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="size-9 rounded-xl bg-gradient-to-br from-emerald-500/30 to-sky-500/30 border border-emerald-400/30 flex items-center justify-center">
            <Sparkles className="size-4 text-emerald-300" />
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-100">
              Export & Use
            </div>
            <div className="text-xs text-slate-400">
              Take your trained brain anywhere.
            </div>
          </div>
        </div>
        <Button
          size="sm"
          onClick={onDownload}
          disabled={!hasModel}
          className="gap-1.5 min-h-[44px] sm:min-h-[36px]"
        >
          <Download className="size-3.5" />
          weights.json
        </Button>
      </div>

      <p className="text-xs text-slate-400 leading-relaxed">
        Your model exports as a portable JSON file containing the architecture,
        activation, and learned weights. Drop it into any web page, Node script,
        or edge function — no ML library required.
      </p>

      <div className="rounded-xl border border-slate-700/80 bg-slate-950/70 overflow-hidden">
        <div className="flex items-center justify-between px-3.5 py-2 border-b border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2">
            <Code2 className="size-3.5 text-slate-400" />
            <span className="text-[11px] text-slate-400 font-mono">
              inference.js
            </span>
          </div>
          <div className="flex gap-1">
            <span className="size-2 rounded-full bg-slate-700" />
            <span className="size-2 rounded-full bg-slate-700" />
            <span className="size-2 rounded-full bg-slate-700" />
          </div>
        </div>
        <pre className="p-4 text-[11px] leading-relaxed font-mono text-slate-300 overflow-x-auto">
          <code>{SNIPPET}</code>
        </pre>
      </div>
    </div>
  );
}
