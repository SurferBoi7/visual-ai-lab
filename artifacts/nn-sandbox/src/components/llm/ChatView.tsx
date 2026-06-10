import { useEffect, useRef, useState } from "react";
import {
  Send,
  Bot,
  User,
  Loader2,
  Sparkles,
  Wand2,
  Trash2,
  Plus,
  MessageSquare,
  ChevronDown,
} from "lucide-react";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export interface ModelOption {
  id: string;
  label: string;
}

interface ChatSession {
  id: string;
  label: string;
  timestamp: number;
}

interface Props {
  modelLabel: string;
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  loading: boolean;
  setLoading: React.Dispatch<React.SetStateAction<boolean>>;
  generate: (seed: string) => Promise<string>;
  liveSample: string;
  epoch: number;
  loss: number;
  isTraining: boolean;
  modelOptions?: ModelOption[];
  onSelectModel?: (id: string) => void;
}

export function ChatView({
  modelLabel,
  messages,
  setMessages,
  loading,
  setLoading,
  generate,
  liveSample,
  epoch,
  loss,
  isTraining,
  modelOptions,
  onSelectModel,
}: Props) {
  const [input, setInput] = useState("");
  const [activeSessionId] = useState<string>(() => `s-${Date.now()}`);
  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: `s-${Date.now()}`, label: "Session 1", timestamp: Date.now() },
  ]);
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [selectedModelLabel, setSelectedModelLabel] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const selectorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (selectorRef.current && !selectorRef.current.contains(e.target as Node)) {
        setModelSelectorOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const startNewSession = () => {
    const newId = `s-${Date.now()}`;
    const newLabel = `Session ${sessions.length + 1}`;
    setSessions((prev) => [...prev, { id: newId, label: newLabel, timestamp: Date.now() }]);
    setMessages([]);
  };

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    try {
      const reply = await generate(text);
      const aiMsg: ChatMessage = { id: `a-${Date.now()}`, role: "assistant", content: reply.trim() || "(silence)" };
      setMessages((prev) => [...prev, aiMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectModel = (id: string, label: string) => {
    if (onSelectModel) onSelectModel(id);
    setSelectedModelLabel(label);
    setModelSelectorOpen(false);
  };

  const activeModelDisplay = selectedModelLabel ?? modelLabel;

  return (
    <div className="h-full flex overflow-hidden bg-[#080808]">

      {/* ── Slim Chat History Sidebar (56px) ─────────────────────────────── */}
      <aside className="w-14 shrink-0 border-r border-white/[0.05] bg-[#080808] flex flex-col items-center pt-3 pb-3 gap-2">

        {/* New Chat */}
        <button
          onClick={startNewSession}
          title="New conversation"
          className="size-9 rounded-xl flex items-center justify-center text-white/25 hover:text-[#0A84FF] hover:bg-[#0A84FF]/10 border border-white/[0.05] hover:border-[#0A84FF]/20 transition-all"
        >
          <Plus className="size-3.5" />
        </button>

        <div className="w-7 h-px bg-white/[0.05]" />

        {/* Session list */}
        <div className="flex flex-col gap-1.5 flex-1 overflow-y-auto w-full items-center py-1">
          {sessions.map((s, i) => {
            const isActive = s.id === activeSessionId;
            return (
              <button
                key={s.id}
                title={s.label}
                className={`size-9 rounded-xl flex items-center justify-center transition-all shrink-0 ${
                  isActive
                    ? "bg-[#0A84FF]/12 text-[#0A84FF] border border-[#0A84FF]/20"
                    : "text-white/20 hover:text-white/50 hover:bg-white/[0.04] border border-transparent"
                }`}
              >
                <MessageSquare className="size-3.5" />
              </button>
            );
          })}
        </div>

        {/* Bottom: live training indicator */}
        {isTraining && (
          <div className="flex flex-col items-center gap-1 pb-1" title="Training in progress">
            <span className="relative flex size-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#0A84FF] opacity-60" />
              <span className="relative inline-flex rounded-full size-2 bg-[#0A84FF]" />
            </span>
          </div>
        )}
      </aside>

      {/* ── Main Chat Canvas ──────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Live generation ticker */}
        {liveSample && (
          <div className="shrink-0 px-4 py-2 border-b border-white/[0.04] flex items-center gap-2.5 overflow-hidden">
            <span className="size-1.5 rounded-full bg-[#30D158] shrink-0 animate-pulse" />
            <Wand2 className="size-3 text-white/20 shrink-0" />
            <code className="text-[10px] font-mono text-[#30D158]/55 truncate leading-tight">{liveSample}</code>
          </div>
        )}

        {/* Messages thread */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {messages.length === 0 && !loading && (
            <div className="h-full flex flex-col items-center justify-center pb-16 px-8">
              <div className="size-16 rounded-2xl bg-[#101010] border border-white/[0.06] flex items-center justify-center mb-5">
                <Sparkles className="size-7 text-[#0A84FF]/45" />
              </div>
              <div className="text-[15px] font-semibold text-white/55 text-center mb-2">
                {epoch > 0 ? "Ready to chat" : "Train first"}
              </div>
              <div className="text-[12px] text-white/20 text-center max-w-[240px] leading-relaxed">
                {epoch > 0
                  ? `${modelLabel} — send a message to begin.`
                  : "Head to Training Lab, press Train, then come back here."}
              </div>
            </div>
          )}

          <div className="px-4 py-5 space-y-5 max-w-2xl mx-auto w-full">
            {messages.map((m) => (
              <div key={m.id} className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}>
                <div className={`size-7 rounded-xl flex items-center justify-center shrink-0 mt-0.5 ${
                  m.role === "user"
                    ? "bg-[#0A84FF]/12 border border-[#0A84FF]/18"
                    : "bg-white/[0.04] border border-white/[0.07]"
                }`}>
                  {m.role === "user" ? (
                    <User className="size-3.5 text-[#0A84FF]" />
                  ) : (
                    <Bot className="size-3.5 text-white/35" />
                  )}
                </div>
                <div className={`max-w-[82%] rounded-2xl px-4 py-2.5 text-[13px] leading-relaxed font-mono ${
                  m.role === "user"
                    ? "bg-[#0A84FF]/10 border border-[#0A84FF]/15 text-white/85 rounded-tr-sm"
                    : "bg-[#101010] border border-white/[0.06] text-white/72 rounded-tl-sm"
                }`}>
                  {m.content}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex gap-3">
                <div className="size-7 rounded-xl bg-white/[0.04] border border-white/[0.07] flex items-center justify-center shrink-0 mt-0.5">
                  <Bot className="size-3.5 text-white/35" />
                </div>
                <div className="rounded-2xl rounded-tl-sm px-4 py-3 bg-[#101010] border border-white/[0.06] flex gap-1.5 items-center">
                  <span className="size-1.5 rounded-full bg-white/22 animate-pulse" />
                  <span className="size-1.5 rounded-full bg-white/22 animate-pulse" style={{ animationDelay: "180ms" }} />
                  <span className="size-1.5 rounded-full bg-white/22 animate-pulse" style={{ animationDelay: "360ms" }} />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Input Bar ──────────────────────────────────────────────────── */}
        <div className="shrink-0 px-4 pb-4 pt-2">
          <div className="max-w-2xl mx-auto space-y-2">

            {/* Model selector + clear chat row — always rendered */}
            <div className="flex items-center gap-2 px-1">
              <span className="text-[10px] text-white/22 font-medium shrink-0">Model:</span>
              {modelOptions && modelOptions.length > 0 ? (
                <div className="relative" ref={selectorRef}>
                  <button
                    onClick={() => setModelSelectorOpen(!modelSelectorOpen)}
                    className="flex items-center gap-1.5 h-6 px-2.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.07] border border-white/[0.06] text-[10px] font-mono text-white/45 hover:text-white/70 transition-all"
                  >
                    <span className="max-w-[160px] truncate">{activeModelDisplay}</span>
                    <ChevronDown className="size-3 shrink-0" />
                  </button>
                  {modelSelectorOpen && (
                    <div className="absolute bottom-full left-0 mb-1.5 w-52 bg-[#111111] border border-white/[0.08] rounded-xl shadow-2xl overflow-hidden z-50">
                      <div className="py-1">
                        {modelOptions.map((o) => (
                          <button
                            key={o.id}
                            onClick={() => handleSelectModel(o.id, o.label)}
                            className="w-full text-left px-3 py-2 text-[11px] text-white/60 hover:text-white/90 hover:bg-white/[0.06] transition-colors truncate"
                          >
                            {o.label}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <span className="h-6 px-2.5 flex items-center rounded-lg bg-white/[0.02] border border-white/[0.04] text-[10px] font-mono text-white/18 select-none cursor-default">
                  No models available
                </span>
              )}

              <div className="flex-1" />

              {messages.length > 0 && (
                <button
                  onClick={() => setMessages([])}
                  className="flex items-center gap-1 h-6 px-2.5 rounded-lg text-[10px] text-white/20 hover:text-red-400/70 hover:bg-red-400/8 border border-transparent hover:border-red-400/15 transition-all"
                >
                  <Trash2 className="size-2.5" />
                  Clear chat
                </button>
              )}
            </div>

            {/* Main input */}
            <div className={`flex items-end bg-[#0d0d0d] rounded-2xl border transition-colors ${
              isTraining ? "border-[#0A84FF]/15" : "border-white/[0.07] focus-within:border-white/[0.12]"
            }`}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
                }}
                rows={1}
                placeholder={isTraining ? "Training in progress — you can still chat…" : "Send a message…"}
                className="flex-1 bg-transparent px-4 py-3 text-[13px] font-mono text-white/80 placeholder:text-white/18 focus:outline-none resize-none leading-relaxed min-h-[20px] max-h-28"
              />
              <div className="px-2 py-2 shrink-0">
                <button
                  onClick={send}
                  disabled={!input.trim() || loading}
                  className={`size-8 rounded-xl flex items-center justify-center transition-all ${
                    input.trim() && !loading
                      ? "bg-[#0A84FF] text-white hover:bg-[#409CFF] shadow-[0_0_14px_rgba(10,132,255,0.28)]"
                      : "bg-white/[0.04] text-white/18"
                  }`}
                >
                  {loading ? <Loader2 className="size-3.5 animate-spin" /> : <Send className="size-3.5" />}
                </button>
              </div>
            </div>

            {/* Status strip */}
            <div className="flex items-center justify-between px-1">
              <div className="text-[10px] text-white/14 font-mono tabular-nums">
                {epoch > 0 ? `ep ${epoch} · loss ${loss.toFixed(4)}` : "untrained"}
                {isTraining && <span className="text-[#0A84FF]/45 ml-1.5">· training…</span>}
              </div>
              {/* Show clear button here if no model options (fallback) */}
              {(!modelOptions || modelOptions.length === 0) && messages.length > 0 && (
                <button
                  onClick={() => setMessages([])}
                  className="flex items-center gap-1 text-[10px] text-white/14 hover:text-red-400/60 transition-colors"
                >
                  <Trash2 className="size-2.5" />
                  clear
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
