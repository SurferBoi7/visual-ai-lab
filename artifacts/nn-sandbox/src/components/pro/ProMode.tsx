import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Zap,
  Cpu,
  AlertTriangle,
  Download,
  Send,
  Square,
  CheckCircle2,
  Loader2,
  Terminal,
  Gauge,
  Box,
  Layers,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface BaseModel {
  id: string;
  label: string;
  size: string;
  description: string;
}

const BASE_MODELS: BaseModel[] = [
  {
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    label: "SmolLM2-135M",
    size: "~80MB",
    description:
      "Instruction-tuned 135M parameter transformer. Fast, tiny, runs fully on-device.",
  },
];

type ChatRole = "user" | "assistant";

interface ProMessage {
  id: string;
  role: ChatRole;
  content: string;
  tokensPerSecond?: number;
  tokens?: number;
  streaming?: boolean;
  error?: boolean;
}

type LoadStage =
  | "idle"
  | "device"
  | "downloading"
  | "initializing"
  | "ready"
  | "error";

interface ProgressEntry {
  file: string;
  progress: number; // 0..100
  loaded: number;
  total: number;
}

interface TelemetryState {
  device: "webgpu" | "wasm" | null;
  hasWebGPU: boolean;
  stage: LoadStage;
  statusMessage: string;
  lastTokensPerSecond: number;
  totalTokens: number;
  lastLatencyMs: number;
}

function formatBytes(n: number) {
  if (!n || n <= 0) return "–";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function ProMode() {
  const workerRef = useRef<Worker | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>(BASE_MODELS[0].id);
  const [loadStage, setLoadStage] = useState<LoadStage>("idle");
  const [progress, setProgress] = useState<Record<string, ProgressEntry>>({});
  const [messages, setMessages] = useState<ProMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [telemetry, setTelemetry] = useState<TelemetryState>({
    device: null,
    hasWebGPU:
      typeof navigator !== "undefined" &&
      typeof (navigator as Navigator & { gpu?: unknown }).gpu !== "undefined",
    stage: "idle",
    statusMessage: "Select a base model and click Load.",
    lastTokensPerSecond: 0,
    totalTokens: 0,
    lastLatencyMs: 0,
  });
  const [genStartedAt, setGenStartedAt] = useState<number | null>(null);
  const [runtimeError, setRuntimeError] = useState<string | null>(null);

  const hasWebGPU = telemetry.hasWebGPU;

  // Spin up worker on mount.
  useEffect(() => {
    const worker = new Worker(
      new URL("@/lib/webgpu.worker.ts", import.meta.url),
      { type: "module" },
    );
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      switch (msg.type) {
        case "status": {
          if (msg.stage === "device") {
            setTelemetry((t) => ({
              ...t,
              device: msg.device,
              stage: "downloading",
              statusMessage: msg.message ?? "",
              hasWebGPU: !!msg.hasWebGPU,
            }));
            setLoadStage("downloading");
          }
          break;
        }
        case "progress": {
          const file = msg.file ?? "model";
          setProgress((prev) => ({
            ...prev,
            [file]: {
              file,
              progress: typeof msg.progress === "number" ? msg.progress : 0,
              loaded: msg.loaded ?? 0,
              total: msg.total ?? 0,
            },
          }));
          if (msg.status === "ready") {
            // individual file ready
          }
          setTelemetry((t) => ({
            ...t,
            stage: t.stage === "ready" ? "ready" : "downloading",
          }));
          break;
        }
        case "ready": {
          setLoadStage("ready");
          setTelemetry((t) => ({
            ...t,
            stage: "ready",
            device: msg.device,
            statusMessage: `${msg.modelId} ready on ${msg.device.toUpperCase()}.`,
          }));
          break;
        }
        case "token": {
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last && last.role === "assistant" && last.streaming) {
              next[next.length - 1] = {
                ...last,
                content: msg.text,
                tokens: msg.tokensGenerated,
              };
            }
            return next;
          });
          break;
        }
        case "done": {
          const tps: number = msg.tokensPerSecond ?? 0;
          const tokens: number = msg.tokensGenerated ?? 0;
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            if (last && last.role === "assistant" && last.streaming) {
              next[next.length - 1] = {
                ...last,
                content: msg.text ?? last.content,
                streaming: false,
                tokens,
                tokensPerSecond: tps,
              };
            }
            return next;
          });
          setStreaming(false);
          setTelemetry((t) => ({
            ...t,
            lastTokensPerSecond: tps,
            totalTokens: t.totalTokens + tokens,
            lastLatencyMs: genStartedAt
              ? performance.now() - genStartedAt
              : t.lastLatencyMs,
          }));
          break;
        }
        case "error": {
          const detail: string =
            msg.payload || msg.message || "Unknown WebGPU Error";
          setLoadStage((prev) => (prev === "ready" ? "ready" : "error"));
          setTelemetry((t) => ({
            ...t,
            stage: t.stage === "ready" ? "ready" : "error",
            statusMessage: detail,
          }));
          setStreaming(false);
          setRuntimeError(detail);
          setMessages((prev) => {
            const next = [...prev];
            const last = next[next.length - 1];
            // Always replace the streaming placeholder with a clearly marked
            // error bubble so the user is never left staring at "...".
            if (last && last.role === "assistant" && last.streaming) {
              next[next.length - 1] = {
                ...last,
                streaming: false,
                error: true,
                content: `Generation failed: ${detail}`,
              };
            }
            return next;
          });
          break;
        }
      }
    };

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [genStartedAt]);

  const loadModel = useCallback(() => {
    if (!workerRef.current) return;
    setProgress({});
    setLoadStage("device");
    setTelemetry((t) => ({
      ...t,
      stage: "device",
      statusMessage: "Requesting accelerated device…",
    }));
    workerRef.current.postMessage({
      type: "load",
      modelId: selectedModel,
      device: hasWebGPU ? "webgpu" : "wasm",
    });
  }, [selectedModel, hasWebGPU]);

  const sendPrompt = useCallback(() => {
    const text = input.trim();
    if (!text || streaming || loadStage !== "ready") return;
    if (!workerRef.current) return;
    const id = `a-${Date.now()}`;
    const userMsg: ProMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
    };
    const placeholder: ProMessage = {
      id,
      role: "assistant",
      content: "",
      streaming: true,
    };
    const history = messages
      .filter((m) => !m.streaming)
      .map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [...prev, userMsg, placeholder]);
    setInput("");
    setStreaming(true);
    setRuntimeError(null);
    setGenStartedAt(performance.now());
    workerRef.current.postMessage({
      type: "generate",
      id,
      prompt: text,
      history,
      maxNewTokens: 256,
      temperature: 0.7,
      topP: 0.9,
    });
  }, [input, streaming, loadStage, messages]);

  const stopGeneration = useCallback(() => {
    workerRef.current?.postMessage({ type: "stop" });
  }, []);

  const totalBytes = useMemo(() => {
    let loaded = 0;
    let total = 0;
    for (const p of Object.values(progress)) {
      loaded += p.loaded || 0;
      total += p.total || 0;
    }
    return { loaded, total };
  }, [progress]);

  const overallProgress = useMemo(() => {
    const entries = Object.values(progress);
    if (entries.length === 0) return 0;
    const avg =
      entries.reduce((acc, p) => acc + (p.progress || 0), 0) / entries.length;
    return avg;
  }, [progress]);

  const ready = loadStage === "ready";
  const downloading =
    loadStage === "downloading" ||
    loadStage === "device" ||
    loadStage === "initializing";

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-mono">
      {/* Subtle grid backdrop */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.07]"
        style={{
          backgroundImage:
            "linear-gradient(to right, #a855f7 1px, transparent 1px), linear-gradient(to bottom, #a855f7 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      <div className="relative grid grid-cols-12 gap-0 h-screen">
        {/* LEFT SIDEBAR — BASE MODELS */}
        <aside className="col-span-3 lg:col-span-2 border-r border-zinc-800 bg-zinc-950/80 backdrop-blur-md flex flex-col">
          <div className="px-4 py-3 border-b border-zinc-800 flex items-center gap-2">
            <div className="size-7 rounded-md bg-gradient-to-br from-fuchsia-500 to-indigo-500 flex items-center justify-center">
              <Box className="size-3.5 text-white" />
            </div>
            <div className="text-xs uppercase tracking-widest text-zinc-400">
              Base Models
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {BASE_MODELS.map((m) => {
              const active = selectedModel === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => setSelectedModel(m.id)}
                  disabled={downloading || streaming}
                  className={`w-full text-left rounded-lg border px-3 py-2.5 transition-colors ${
                    active
                      ? "bg-gradient-to-br from-fuchsia-500/15 to-indigo-500/15 border-fuchsia-500/40"
                      : "border-zinc-800 hover:border-zinc-700 hover:bg-zinc-900/60"
                  } disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Layers
                      className={`size-3.5 ${
                        active ? "text-fuchsia-300" : "text-zinc-500"
                      }`}
                    />
                    <span className="text-[13px] font-semibold text-zinc-100">
                      {m.label}
                    </span>
                    <span className="ml-auto text-[10px] text-zinc-500">
                      {m.size}
                    </span>
                  </div>
                  <div className="text-[11px] text-zinc-400 leading-snug">
                    {m.description}
                  </div>
                </button>
              );
            })}
          </div>

          <div className="p-3 border-t border-zinc-800 space-y-2">
            <Button
              onClick={loadModel}
              disabled={downloading}
              className="w-full gap-2 bg-gradient-to-br from-fuchsia-500 to-indigo-500 text-white hover:opacity-90 min-h-[38px] rounded-lg"
            >
              {ready ? (
                <>
                  <CheckCircle2 className="size-3.5" />
                  Reload
                </>
              ) : downloading ? (
                <>
                  <Loader2 className="size-3.5 animate-spin" />
                  Loading…
                </>
              ) : (
                <>
                  <Download className="size-3.5" />
                  Load Model
                </>
              )}
            </Button>
            <div className="text-[10px] text-zinc-500 leading-snug">
              Weights cached in your browser after first download.
            </div>
          </div>
        </aside>

        {/* CENTER — CHAT TERMINAL */}
        <main className="col-span-9 lg:col-span-7 flex flex-col min-h-0">
          <div className="px-4 py-3 border-b border-zinc-800 bg-zinc-950/90 backdrop-blur-md flex items-center justify-between">
            <div className="flex items-center gap-2 min-w-0">
              <Terminal className="size-4 text-fuchsia-400" />
              <span className="text-[13px] font-semibold text-zinc-100 truncate">
                {selectedModel}
              </span>
              <span
                className={`ml-2 text-[10px] px-2 py-0.5 rounded-full border uppercase tracking-wider ${
                  ready
                    ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                    : downloading
                      ? "bg-indigo-500/10 text-indigo-300 border-indigo-500/30 animate-pulse"
                      : "bg-zinc-800/50 text-zinc-400 border-zinc-700"
                }`}
              >
                {ready
                  ? `online · ${telemetry.device?.toUpperCase() ?? ""}`
                  : downloading
                    ? "booting"
                    : "offline"}
              </span>
            </div>
            {streaming && (
              <Button
                size="sm"
                variant="secondary"
                onClick={stopGeneration}
                className="gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900 text-zinc-100 hover:bg-zinc-800"
              >
                <Square className="size-3" /> Stop
              </Button>
            )}
          </div>

          {runtimeError && (
            <div className="mx-4 mt-3 rounded-lg border border-red-500/50 bg-red-500/10 px-3 py-2.5 flex items-start gap-2">
              <AlertTriangle className="size-4 text-red-300 shrink-0 mt-0.5" />
              <div className="flex-1 text-[11px] text-red-100 leading-snug">
                <div className="font-semibold mb-0.5 uppercase tracking-wider text-red-200">
                  Runtime Error
                </div>
                <div className="font-mono break-words">{runtimeError}</div>
              </div>
              <button
                onClick={() => setRuntimeError(null)}
                className="text-[10px] uppercase tracking-wider text-red-200/70 hover:text-red-100 px-2 py-1 rounded border border-red-500/40 hover:border-red-500/70"
              >
                Dismiss
              </button>
            </div>
          )}

          {!hasWebGPU && (
            <div className="mx-4 mt-3 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2.5 flex items-start gap-2">
              <AlertTriangle className="size-4 text-amber-300 shrink-0 mt-0.5" />
              <div className="text-[11px] text-amber-100 leading-snug">
                <div className="font-semibold mb-0.5">
                  WebGPU unavailable on this device.
                </div>
                Pro Mode will fall back to WebAssembly (CPU). Inference works
                but will be noticeably slower — expect single-digit tokens/sec.
              </div>
            </div>
          )}

          {/* Download progress */}
          {downloading && (
            <div className="mx-4 mt-3 rounded-lg border border-indigo-500/30 bg-indigo-500/5 px-3 py-3">
              <div className="flex items-center justify-between mb-2">
                <div className="text-[11px] uppercase tracking-wider text-indigo-300 flex items-center gap-1.5">
                  <Download className="size-3" /> Fetching weights
                </div>
                <div className="text-[11px] tabular-nums text-zinc-400">
                  {formatBytes(totalBytes.loaded)} /{" "}
                  {formatBytes(totalBytes.total)}
                </div>
              </div>
              <div className="h-1.5 rounded-full bg-zinc-800 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-fuchsia-500 to-indigo-500 transition-all"
                  style={{ width: `${Math.min(100, overallProgress)}%` }}
                />
              </div>
              <div className="mt-2 space-y-1 max-h-20 overflow-y-auto">
                {Object.values(progress).map((p) => (
                  <div
                    key={p.file}
                    className="flex items-center justify-between text-[10px] text-zinc-500"
                  >
                    <span className="truncate max-w-[60%]">{p.file}</span>
                    <span className="tabular-nums">
                      {Math.round(p.progress)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
            {messages.length === 0 && (
              <div className="h-full flex flex-col items-center justify-center text-center px-6">
                <div className="size-14 rounded-2xl bg-gradient-to-br from-fuchsia-500/20 to-indigo-500/20 border border-fuchsia-500/30 flex items-center justify-center mb-4">
                  <Sparkles className="size-6 text-fuchsia-300" />
                </div>
                <div className="text-sm font-semibold text-zinc-100">
                  {ready
                    ? "Ready. Ask me anything."
                    : "Load a base model to begin."}
                </div>
                <div className="text-[11px] text-zinc-500 mt-1 max-w-sm">
                  {ready
                    ? "Runs fully in your browser — no server, no API keys. Your conversation never leaves this tab."
                    : "Models are downloaded once and cached. Subsequent loads are instant."}
                </div>
              </div>
            )}

            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex gap-3 ${
                  m.role === "user" ? "flex-row-reverse" : ""
                }`}
              >
                <div
                  className={`size-7 rounded-md flex items-center justify-center shrink-0 border ${
                    m.role === "user"
                      ? "bg-indigo-500/15 border-indigo-500/30 text-indigo-200"
                      : "bg-fuchsia-500/15 border-fuchsia-500/30 text-fuchsia-200"
                  }`}
                >
                  {m.role === "user" ? (
                    <span className="text-[10px] font-semibold">YOU</span>
                  ) : (
                    <Zap className="size-3.5" />
                  )}
                </div>
                <div
                  className={`max-w-[78%] rounded-lg px-3.5 py-2.5 text-[13px] leading-relaxed whitespace-pre-wrap break-words border ${
                    m.error
                      ? "bg-red-500/10 border-red-500/40 text-red-100 font-mono"
                      : m.role === "user"
                        ? "bg-indigo-500/10 border-indigo-500/30 text-indigo-50"
                        : "bg-zinc-900/80 border-zinc-800 text-zinc-100"
                  }`}
                >
                  {m.content ||
                    (m.streaming ? (
                      <span className="inline-flex gap-1 items-center text-zinc-500">
                        <span className="size-1.5 rounded-full bg-fuchsia-400 animate-pulse" />
                        <span
                          className="size-1.5 rounded-full bg-fuchsia-400 animate-pulse"
                          style={{ animationDelay: "150ms" }}
                        />
                        <span
                          className="size-1.5 rounded-full bg-fuchsia-400 animate-pulse"
                          style={{ animationDelay: "300ms" }}
                        />
                      </span>
                    ) : (
                      ""
                    ))}
                  {m.role === "assistant" &&
                    !m.streaming &&
                    typeof m.tokensPerSecond === "number" &&
                    m.tokens ? (
                    <div className="mt-2 pt-2 border-t border-zinc-800 text-[10px] text-zinc-500 tabular-nums flex items-center gap-3">
                      <span>{m.tokens} tokens</span>
                      <span>·</span>
                      <span>{m.tokensPerSecond.toFixed(1)} tok/s</span>
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>

          <div className="border-t border-zinc-800 bg-zinc-950/90 p-3">
            <div className="flex items-end gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendPrompt();
                  }
                }}
                rows={1}
                placeholder={
                  ready
                    ? "Message SmolLM…"
                    : "Load a model to enable prompt input."
                }
                disabled={!ready || streaming}
                className="flex-1 min-h-[44px] max-h-40 rounded-lg border border-zinc-800 bg-zinc-900/80 px-3 py-2.5 text-[13px] text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-fuchsia-500/40 resize-none font-mono disabled:opacity-60"
              />
              <Button
                onClick={sendPrompt}
                disabled={!ready || streaming || !input.trim()}
                className="min-h-[44px] min-w-[44px] rounded-lg gap-1.5 bg-gradient-to-br from-fuchsia-500 to-indigo-500 text-white hover:opacity-90"
              >
                {streaming ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Send className="size-4" />
                )}
              </Button>
            </div>
          </div>
        </main>

        {/* RIGHT PANEL — GPU TELEMETRY */}
        <aside className="hidden lg:flex col-span-3 border-l border-zinc-800 bg-zinc-950/80 backdrop-blur-md flex-col">
          <div className="px-4 py-3 border-b border-zinc-800 flex items-center gap-2">
            <Gauge className="size-4 text-fuchsia-400" />
            <div className="text-xs uppercase tracking-widest text-zinc-400">
              Runtime Telemetry
            </div>
          </div>

          <div className="p-3 space-y-3 overflow-y-auto">
            <TelemetryCard
              title="Compute Device"
              icon={
                telemetry.device === "webgpu" ? (
                  <Zap className="size-3.5 text-fuchsia-300" />
                ) : (
                  <Cpu className="size-3.5 text-amber-300" />
                )
              }
            >
              <div className="text-[13px] font-semibold text-zinc-100">
                {telemetry.device
                  ? telemetry.device.toUpperCase()
                  : hasWebGPU
                    ? "WebGPU (pending)"
                    : "WASM (fallback)"}
              </div>
              <div className="text-[10px] text-zinc-500 mt-0.5">
                {hasWebGPU
                  ? "navigator.gpu detected"
                  : "navigator.gpu unavailable"}
              </div>
            </TelemetryCard>

            <TelemetryCard title="Status">
              <div
                className={`text-[12px] leading-snug ${
                  telemetry.stage === "error"
                    ? "text-rose-300"
                    : telemetry.stage === "ready"
                      ? "text-emerald-300"
                      : "text-zinc-300"
                }`}
              >
                {telemetry.statusMessage}
              </div>
            </TelemetryCard>

            <TelemetryCard title="Weights">
              <div className="text-[13px] font-semibold text-zinc-100 tabular-nums">
                {formatBytes(totalBytes.loaded)}
              </div>
              <div className="text-[10px] text-zinc-500 mt-0.5 tabular-nums">
                of {formatBytes(totalBytes.total)} downloaded
              </div>
              <div className="mt-2 h-1 rounded-full bg-zinc-800 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-fuchsia-500 to-indigo-500"
                  style={{
                    width: `${Math.min(100, overallProgress)}%`,
                  }}
                />
              </div>
            </TelemetryCard>

            <TelemetryCard title="Throughput">
              <div className="text-[13px] font-semibold text-zinc-100 tabular-nums">
                {telemetry.lastTokensPerSecond > 0
                  ? `${telemetry.lastTokensPerSecond.toFixed(1)} tok/s`
                  : "—"}
              </div>
              <div className="text-[10px] text-zinc-500 mt-0.5 tabular-nums">
                last completion
              </div>
            </TelemetryCard>

            <TelemetryCard title="Session">
              <div className="grid grid-cols-2 gap-2">
                <MetricTile
                  label="Tokens"
                  value={telemetry.totalTokens.toString()}
                />
                <MetricTile
                  label="Last (ms)"
                  value={
                    telemetry.lastLatencyMs
                      ? Math.round(telemetry.lastLatencyMs).toString()
                      : "—"
                  }
                />
                <MetricTile label="Msgs" value={messages.length.toString()} />
                <MetricTile
                  label="WebGPU"
                  value={hasWebGPU ? "yes" : "no"}
                  accent={hasWebGPU ? "ok" : "warn"}
                />
              </div>
            </TelemetryCard>
          </div>
        </aside>
      </div>
    </div>
  );
}

function TelemetryCard({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2.5">
      <div className="flex items-center gap-1.5 mb-1.5">
        {icon}
        <div className="text-[10px] uppercase tracking-widest text-zinc-500">
          {title}
        </div>
      </div>
      {children}
    </div>
  );
}

function MetricTile({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "ok" | "warn";
}) {
  const tone =
    accent === "ok"
      ? "text-emerald-300"
      : accent === "warn"
        ? "text-amber-300"
        : "text-zinc-100";
  return (
    <div className="rounded-md border border-zinc-800 bg-zinc-950/50 px-2 py-2 text-center">
      <div className="text-[9px] uppercase tracking-wider text-zinc-500">
        {label}
      </div>
      <div className={`text-xs font-semibold tabular-nums mt-0.5 ${tone}`}>
        {value}
      </div>
    </div>
  );
}
