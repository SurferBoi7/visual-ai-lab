import { useState } from "react";
import { Copy, Check, Key, Globe2, Brain, AlertCircle, Rocket } from "lucide-react";

interface Props {
  modelLabel: string;
  hasModel: boolean;
  epoch: number;
  loss: number;
}

export function DeployHub({ modelLabel, hasModel, epoch, loss }: Props) {
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  const generateKey = () => {
    const seg = () => Math.random().toString(36).slice(2, 10);
    setApiKey(`sk-tiny-ai-${seg()}-${seg()}`);
  };

  const copy = (text: string, id: string) => {
    navigator.clipboard.writeText(text).catch(() => {});
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const appName = window.location.hostname.replace(/\./g, "-").slice(0, 40);
  const endpointUrl = `wss://${appName}/api/chat`;

  return (
    <div className="max-w-2xl mx-auto py-10 px-4 md:px-0 space-y-5">

      {/* Header */}
      <div className="mb-2">
        <div className="flex items-center gap-3 mb-1">
          <div className="size-9 rounded-xl bg-[#0A84FF]/10 border border-[#0A84FF]/15 flex items-center justify-center shrink-0">
            <Rocket className="size-[18px] text-[#0A84FF]" />
          </div>
          <h1 className="text-xl font-semibold tracking-tight text-white">Deploy Hub</h1>
        </div>
        <p className="text-sm text-white/40 leading-relaxed pl-12">
          Securely bridge your trained model to external applications via a WebSocket API.
        </p>
      </div>

      {/* Active model status */}
      <div className="rounded-2xl border border-white/[0.06] bg-[#141414] p-5">
        <div className="text-[10px] uppercase tracking-[0.12em] text-white/30 font-medium mb-3">
          Active Model
        </div>
        <div className="flex items-center gap-3">
          <div className="size-10 rounded-xl bg-[#0A84FF]/10 border border-[#0A84FF]/15 flex items-center justify-center shrink-0">
            <Brain className="size-5 text-[#0A84FF]" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-[14px] font-semibold text-white leading-tight font-mono truncate">
              {modelLabel}
            </div>
            <div className="text-[11px] text-white/40 mt-0.5">
              {hasModel
                ? `Epoch ${epoch} · Loss ${loss.toFixed(3)}`
                : "No weights loaded — train first"}
            </div>
          </div>
          <div
            className={`size-2 rounded-full shrink-0 ${
              hasModel ? "bg-[#30D158]" : "bg-white/15"
            }`}
            style={hasModel ? { boxShadow: "0 0 6px #30D158" } : undefined}
          />
        </div>
      </div>

      {/* API Key */}
      <div className="rounded-2xl border border-white/[0.06] bg-[#141414] p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Key className="size-4 text-[#0A84FF]" />
          <span className="text-sm font-semibold text-white">API Key</span>
        </div>

        {apiKey ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2 bg-[#0a0a0a] border border-white/[0.08] rounded-xl px-4 py-3">
              <code className="flex-1 text-[12px] font-mono text-[#30D158] truncate">
                {apiKey}
              </code>
              <button
                onClick={() => copy(apiKey, "key")}
                className="shrink-0 size-7 rounded-lg flex items-center justify-center text-white/40 hover:text-white/80 hover:bg-white/[0.06] transition-colors"
              >
                {copied === "key" ? (
                  <Check className="size-3.5 text-[#30D158]" />
                ) : (
                  <Copy className="size-3.5" />
                )}
              </button>
            </div>
            <p className="text-[11px] text-white/30">
              Store this key securely — it will not be shown again once you regenerate.
            </p>
            <button
              onClick={generateKey}
              className="text-[11px] text-white/35 hover:text-white/60 transition-colors underline underline-offset-2"
            >
              Regenerate
            </button>
          </div>
        ) : (
          <button
            onClick={generateKey}
            className="w-full h-11 rounded-xl bg-[#0A84FF] hover:bg-[#409CFF] text-white text-sm font-semibold transition-colors flex items-center justify-center gap-2"
          >
            <Key className="size-4" />
            Generate API Key
          </button>
        )}
      </div>

      {/* Endpoint URL */}
      <div className="rounded-2xl border border-white/[0.06] bg-[#141414] p-5 space-y-4">
        <div className="flex items-center gap-2">
          <Globe2 className="size-4 text-[#0A84FF]" />
          <span className="text-sm font-semibold text-white">Endpoint URL</span>
        </div>
        <div className="flex items-center gap-2 bg-[#0a0a0a] border border-white/[0.08] rounded-xl px-4 py-3">
          <code className="flex-1 text-[12px] font-mono text-white/50 truncate">
            {endpointUrl}
          </code>
          <button
            onClick={() => copy(endpointUrl, "url")}
            className="shrink-0 size-7 rounded-lg flex items-center justify-center text-white/40 hover:text-white/80 hover:bg-white/[0.06] transition-colors"
          >
            {copied === "url" ? (
              <Check className="size-3.5 text-[#30D158]" />
            ) : (
              <Copy className="size-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* How it works */}
      <div className="rounded-2xl border border-white/[0.04] bg-[#0e0e0e] p-5 space-y-3">
        <div className="text-[10px] uppercase tracking-[0.12em] text-white/25 font-medium">
          How it works
        </div>
        <p className="text-[13px] text-white/40 leading-relaxed">
          Use this API Key in your consumer chat application to securely bridge
          to this model's inference engine. The endpoint accepts WebSocket
          connections authenticated with the{" "}
          <code className="text-white/55 bg-white/[0.06] px-1.5 py-0.5 rounded text-[11px] font-mono">
            Bearer
          </code>{" "}
          token. Send{" "}
          <code className="text-white/55 bg-white/[0.06] px-1.5 py-0.5 rounded text-[11px] font-mono">
            {"{ message: string }"}
          </code>{" "}
          frames and receive streamed completions back in real time.
        </p>
        <div className="rounded-xl bg-[#0a0a0a] border border-white/[0.06] p-4 font-mono text-[11px] text-white/40 leading-relaxed space-y-1">
          <div>
            <span className="text-white/20">// Example</span>
          </div>
          <div>
            <span className="text-[#0A84FF]/70">const</span>{" "}
            <span className="text-white/60">ws</span>{" "}
            <span className="text-white/30">=</span>{" "}
            <span className="text-[#30D158]/70">new</span>{" "}
            <span className="text-white/50">WebSocket</span>
            <span className="text-white/30">(endpointUrl);</span>
          </div>
          <div>
            <span className="text-white/50">ws.setRequestHeader</span>
            <span className="text-white/30">(</span>
            <span className="text-[#FF9F0A]/70">'Authorization'</span>
            <span className="text-white/30">, </span>
            <span className="text-[#FF9F0A]/70">`Bearer {"${apiKey}"}`</span>
            <span className="text-white/30">);</span>
          </div>
          <div>
            <span className="text-white/50">ws.send</span>
            <span className="text-white/30">(JSON.stringify({"{ message: 'hello' }"}));</span>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-2.5 px-1">
        <AlertCircle className="size-3.5 text-white/20 mt-0.5 shrink-0" />
        <p className="text-[11px] text-white/20 leading-relaxed">
          This is a mock Deploy Hub for demonstration purposes. The model runs
          entirely in your browser — there is no persistent server. A real
          deployment would bundle the trained weights with a serverless
          inference runtime.
        </p>
      </div>
    </div>
  );
}
