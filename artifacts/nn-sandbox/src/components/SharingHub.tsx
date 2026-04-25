import { useRef, useState } from "react";
import {
  Download,
  Upload,
  Code2,
  Sparkles,
  Cloud,
  Loader2,
  Check,
  Copy,
  Github,
  Twitter,
  Link2,
  Globe,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";

interface Props {
  mode: "mlp" | "llm";
  onDownload: () => void;
  hasModel: boolean;
  onImport?: (json: string) => void;
}

const MLP_SNIPPET = `// Load your trained model
const res = await fetch('weights.json');
const model = await res.json();

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

const LLM_SNIPPET = `// Load your fine-tuned LoRA adapter
import { pipeline } from '@huggingface/transformers';

const chat = await pipeline('text-generation', 'smollm-135m', {
  adapter: 'adapter.safetensors',
  device: 'webgpu',
});

const reply = await chat([
  { role: 'system', content: 'You are a helpful assistant.' },
  { role: 'user',   content: 'Hello!' },
]);

console.log(reply[0].generated_text);`;

function makeId() {
  return Math.random().toString(36).slice(2, 9);
}

export function SharingHub({ mode, onDownload, hasModel, onImport }: Props) {
  const { toast } = useToast();
  const [publishing, setPublishing] = useState(false);
  const [shareId, setShareId] = useState<string | null>(null);
  const [copied, setCopied] = useState<"link" | "embed" | null>(null);
  const [importOk, setImportOk] = useState(false);
  const importRef = useRef<HTMLInputElement | null>(null);

  const snippet = mode === "mlp" ? MLP_SNIPPET : LLM_SNIPPET;
  const fileLabel = mode === "mlp" ? "weights.json" : "adapter.safetensors";
  const fileName = mode === "mlp" ? "inference.js" : "chat.ts";

  const shareUrl = shareId ? `ai-sandbox.com/model/${shareId}` : "";
  const embedCode = shareId
    ? `<iframe
  src="https://${shareUrl}/embed"
  width="480" height="360"
  style="border:0;border-radius:16px"
  allow="webgpu"
></iframe>`
    : "";

  const publish = () => {
    if (publishing) return;
    setPublishing(true);
    setTimeout(() => {
      setShareId(makeId());
      setPublishing(false);
      toast({
        title: "Published to cloud",
        description: "Your model has a shareable link.",
      });
    }, 1400);
  };

  const copy = async (text: string, kind: "link" | "embed") => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(kind);
      setTimeout(() => setCopied(null), 1500);
      toast({ title: "Copied to clipboard" });
    } catch {
      toast({ title: "Copy failed", description: "Select and copy manually." });
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const json = ev.target?.result as string;
      if (!json) return;
      onImport?.(json);
      setImportOk(true);
      setTimeout(() => setImportOk(false), 2000);
    };
    reader.readAsText(file);
    e.target.value = "";
  };

  const tweetUrl = shareId
    ? `https://twitter.com/intent/tweet?text=${encodeURIComponent(
        `I trained a tiny ${mode === "mlp" ? "neural network" : "LLM"} in my browser →`,
      )}&url=${encodeURIComponent(`https://${shareUrl}`)}`
    : "#";
  const githubUrl = "https://github.com";

  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-800/50 backdrop-blur-md p-5 space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="size-9 rounded-xl bg-gradient-to-br from-emerald-500/30 to-sky-500/30 border border-emerald-400/30 flex items-center justify-center">
            <Sparkles className="size-4 text-emerald-300" />
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-100">
              Export & Share
            </div>
            <div className="text-xs text-slate-400">
              {mode === "mlp"
                ? "Take your trained brain anywhere."
                : "Ship your fine-tuned chatbot."}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {/* Import Model */}
          {onImport && (
            <>
              <input
                ref={importRef}
                type="file"
                accept=".json"
                className="hidden"
                onChange={handleFileChange}
              />
              <Button
                size="sm"
                variant="secondary"
                onClick={() => importRef.current?.click()}
                className={`gap-1.5 min-h-[44px] sm:min-h-[36px] rounded-xl transition-colors ${
                  importOk
                    ? "bg-emerald-600 hover:bg-emerald-600 text-white border-emerald-500"
                    : ""
                }`}
              >
                {importOk ? (
                  <>
                    <Check className="size-3.5" />
                    Model Loaded!
                  </>
                ) : (
                  <>
                    <Upload className="size-3.5" />
                    Import
                  </>
                )}
              </Button>
            </>
          )}
          {/* Export / Download */}
          <Button
            size="sm"
            onClick={onDownload}
            disabled={!hasModel}
            className="gap-1.5 min-h-[44px] sm:min-h-[36px] rounded-xl"
          >
            <Download className="size-3.5" />
            {fileLabel}
          </Button>
        </div>
      </div>

      {/* Code snippet */}
      <div className="rounded-xl border border-slate-700/80 bg-slate-950/70 overflow-hidden">
        <div className="flex items-center justify-between px-3.5 py-2 border-b border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-2">
            <Code2 className="size-3.5 text-slate-400" />
            <span className="text-[11px] text-slate-400 font-mono">
              {fileName}
            </span>
          </div>
          <div className="flex gap-1">
            <span className="size-2 rounded-full bg-slate-700" />
            <span className="size-2 rounded-full bg-slate-700" />
            <span className="size-2 rounded-full bg-slate-700" />
          </div>
        </div>
        <pre className="p-4 text-[11px] leading-relaxed font-mono text-slate-300 overflow-x-auto">
          <code>{snippet}</code>
        </pre>
      </div>

      {/* Publish to Cloud */}
      <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-4 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Cloud className="size-4 text-sky-400" />
            <span className="text-sm font-semibold text-slate-100">
              Publish to Cloud
            </span>
          </div>
          {!shareId && (
            <Button
              size="sm"
              onClick={publish}
              disabled={publishing || !hasModel}
              className="gap-1.5 min-h-[40px] rounded-xl"
            >
              {publishing ? (
                <>
                  <Loader2 className="size-3.5 animate-spin" />
                  Publishing…
                </>
              ) : (
                <>
                  <Cloud className="size-3.5" />
                  Publish
                </>
              )}
            </Button>
          )}
          {shareId && (
            <span className="text-[10px] px-2 py-1 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 inline-flex items-center gap-1">
              <Check className="size-3" /> Live
            </span>
          )}
        </div>

        {!shareId && (
          <p className="text-[11px] text-slate-400">
            Get a public, shareable link in seconds. Your model runs on the
            visitor's device — no server, no cost.
          </p>
        )}

        {shareId && (
          <>
            <div className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-950/70 px-3 py-2">
              <Globe className="size-3.5 text-sky-300 shrink-0" />
              <span className="text-xs font-mono text-slate-200 truncate flex-1">
                {shareUrl}
              </span>
              <button
                onClick={() => copy(`https://${shareUrl}`, "link")}
                className="text-slate-400 hover:text-sky-300 transition-colors min-h-[32px] min-w-[32px] flex items-center justify-center rounded-md"
                aria-label="Copy link"
              >
                {copied === "link" ? (
                  <Check className="size-3.5 text-emerald-400" />
                ) : (
                  <Copy className="size-3.5" />
                )}
              </button>
            </div>

            <div className="flex items-center gap-2 pt-1">
              <a
                href={tweetUrl}
                target="_blank"
                rel="noreferrer"
                className="size-10 rounded-xl border border-slate-700 bg-slate-950/70 flex items-center justify-center text-slate-300 hover:text-sky-300 hover:border-sky-500/40 transition-colors"
                aria-label="Share on X"
              >
                <Twitter className="size-4" />
              </a>
              <a
                href={githubUrl}
                target="_blank"
                rel="noreferrer"
                className="size-10 rounded-xl border border-slate-700 bg-slate-950/70 flex items-center justify-center text-slate-300 hover:text-slate-100 hover:border-slate-500 transition-colors"
                aria-label="Open GitHub"
              >
                <Github className="size-4" />
              </a>
              <button
                onClick={() => copy(`https://${shareUrl}`, "link")}
                className="size-10 rounded-xl border border-slate-700 bg-slate-950/70 flex items-center justify-center text-slate-300 hover:text-emerald-300 hover:border-emerald-500/40 transition-colors"
                aria-label="Copy link"
              >
                <Link2 className="size-4" />
              </button>
            </div>

            {/* Embed iframe */}
            <div className="rounded-xl border border-slate-700/80 bg-slate-950/70 overflow-hidden mt-2">
              <div className="flex items-center justify-between px-3 py-2 border-b border-slate-800 bg-slate-900/60">
                <div className="flex items-center gap-2">
                  <Code2 className="size-3.5 text-slate-400" />
                  <span className="text-[11px] text-slate-400 font-mono">
                    Embed as widget
                  </span>
                </div>
                <button
                  onClick={() => copy(embedCode, "embed")}
                  className="text-slate-400 hover:text-sky-300 transition-colors min-h-[28px] px-1.5 flex items-center gap-1 text-[10px]"
                >
                  {copied === "embed" ? (
                    <>
                      <Check className="size-3 text-emerald-400" /> Copied
                    </>
                  ) : (
                    <>
                      <Copy className="size-3" /> Copy
                    </>
                  )}
                </button>
              </div>
              <pre className="p-3 text-[11px] leading-relaxed font-mono text-slate-300 overflow-x-auto">
                <code>{embedCode}</code>
              </pre>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
