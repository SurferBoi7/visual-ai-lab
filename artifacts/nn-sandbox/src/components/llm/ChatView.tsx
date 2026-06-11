import { useEffect, useRef, useState, useCallback } from "react";
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
  X,
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

interface Thread {
  id: string;
  name: string;
  messages: ChatMessage[];
  modelId: string | null;
  modelLabel: string;
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

function makeThread(index: number): Thread {
  return { id: `t-${Date.now()}-${index}`, name: `Session ${index}`, messages: [], modelId: null, modelLabel: "" };
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
  const [threads, setThreads] = useState<Thread[]>(() => [makeThread(1)]);
  const [activeId, setActiveId] = useState<string>(() => {
    const t = makeThread(1);
    return t.id;
  });
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedModelLabel, setSelectedModelLabel] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const selectorRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // Initialize threads with a stable first thread
  const [initialized, setInitialized] = useState(false);
  useEffect(() => {
    if (!initialized) {
      const first = makeThread(1);
      setThreads([first]);
      setActiveId(first.id);
      setInitialized(true);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeThread = threads.find((t) => t.id === activeId) ?? threads[0];

  // Sync active thread messages back to App.tsx for LLMStats
  useEffect(() => {
    if (activeThread) {
      setMessages(activeThread.messages);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeThread?.messages]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [activeThread?.messages, loading]);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (selectorRef.current && !selectorRef.current.contains(e.target as Node)) {
        setModelSelectorOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const updateActiveMessages = useCallback((updater: (prev: ChatMessage[]) => ChatMessage[]) => {
    setThreads((prev) =>
      prev.map((t) => (t.id === activeId ? { ...t, messages: updater(t.messages) } : t)),
    );
  }, [activeId]);

  const launchNewSession = () => {
    const n = threads.length + 1;
    const t = makeThread(n);
    setThreads((prev) => [...prev, t]);
    setActiveId(t.id);
    inputRef.current?.focus();
  };

  const deleteSession = (id: string) => {
    setThreads((prev) => {
      const next = prev.filter((t) => t.id !== id);
      if (next.length === 0) {
        const fresh = makeThread(1);
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) {
        setActiveId(next[next.length - 1].id);
      }
      return next;
    });
  };

  const clearCurrentSession = () => {
    setThreads((prev) =>
      prev.map((t) => (t.id === activeId ? { ...t, messages: [] } : t)),
    );
  };

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: "user", content: text };
    updateActiveMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    try {
      const reply = await generate(text);
      const aiMsg: ChatMessage = { id: `a-${Date.now()}`, role: "assistant", content: reply.trim() || "(silence)" };
      updateActiveMessages((prev) => [...prev, aiMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectModel = (id: string, label: string) => {
    if (onSelectModel) onSelectModel(id);
    setSelectedModelId(id);
    setSelectedModelLabel(label);
    setModelSelectorOpen(false);
    // tag the active thread with this model
    setThreads((prev) =>
      prev.map((t) => (t.id === activeId ? { ...t, modelId: id, modelLabel: label } : t)),
    );
  };

  const activeModelDisplay = selectedModelLabel ?? modelLabel;
  const activeMessages = activeThread?.messages ?? [];

  return (
    <div className="h-full flex overflow-hidden bg-[#000000]">

      {/* ── Multi-Thread Sidebar ──────────────────────────────────────────── */}
      <aside className="w-52 shrink-0 border-r border-white/[0.05] bg-[#030303] flex flex-col overflow-hidden">

        {/* Sidebar header */}
        <div className="h-11 shrink-0 border-b border-white/[0.05] flex items-center justify-between px-3">
          <span className="text-[10px] font-semibold text-white/28 tracking-[0.12em] uppercase">Chat Matrix</span>
          {isTraining && (
            <span className="relative flex size-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#0A84FF] opacity-60" />
              <span className="relative inline-flex rounded-full size-1.5 bg-[#0A84FF]" />
            </span>
          )}
        </div>

        {/* Launch New Session button */}
        <div className="px-2.5 pt-2.5 pb-2 shrink-0">
          <button
            onClick={launchNewSession}
            className="w-full h-8 rounded-xl border border-dashed border-white/[0.08] hover:border-[#0A84FF]/30 hover:bg-[#0A84FF]/[0.04] text-white/28 hover:text-[#0A84FF]/70 text-[11px] font-medium flex items-center justify-center gap-1.5 transition-all"
          >
            <Plus className="size-3" />
            Launch New Session
          </button>
        </div>

        {/* Thread list */}
        <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-px">
          {threads.map((t) => {
            const isActive = t.id === activeId;
            return (
              <div
                key={t.id}
                className={`group relative rounded-xl transition-all ${
                  isActive
                    ? "bg-[#0A84FF]/10 border border-[#0A84FF]/15"
                    : "border border-transparent hover:bg-white/[0.025]"
                }`}
              >
                <button
                  onClick={() => setActiveId(t.id)}
                  className="w-full text-left flex items-center gap-2 px-2.5 py-2"
                >
                  <MessageSquare className={`size-3 shrink-0 ${isActive ? "text-[#0A84FF]/70" : "text-white/20"}`} />
                  <div className="flex-1 min-w-0">
                    <div className={`text-[11px] font-medium truncate ${isActive ? "text-[#0A84FF]/90" : "text-white/40"}`}>
                      {t.name}
                    </div>
                    <div className="text-[9px] text-white/18 tabular-nums mt-0.5">
                      {t.messages.length} msg{t.messages.length !== 1 ? "s" : ""}
                      {t.modelLabel ? ` · ${t.modelLabel.split("·")[0].trim()}` : ""}
                    </div>
                  </div>
                </button>

                {/* Per-session actions */}
                <div className={`absolute right-1.5 top-1/2 -translate-y-1/2 flex items-center gap-0.5 ${isActive ? "opacity-100" : "opacity-0 group-hover:opacity-100"} transition-opacity`}>
                  {isActive && t.messages.length > 0 && (
                    <button
                      onClick={(e) => { e.stopPropagation(); clearCurrentSession(); }}
                      title="Clear session"
                      className="size-5 rounded-md flex items-center justify-center text-white/20 hover:text-amber-400/70 hover:bg-amber-400/8 transition-all"
                    >
                      <Trash2 className="size-2.5" />
                    </button>
                  )}
                  {threads.length > 1 && (
                    <button
                      onClick={(e) => { e.stopPropagation(); deleteSession(t.id); }}
                      title="Delete session"
                      className="size-5 rounded-md flex items-center justify-center text-white/20 hover:text-red-400/70 hover:bg-red-400/8 transition-all"
                    >
                      <X className="size-2.5" />
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Sidebar footer stats */}
        <div className="border-t border-white/[0.05] px-3 py-2.5 shrink-0">
          <div className="text-[9px] text-white/18 font-mono tabular-nums leading-relaxed">
            <div>ep {epoch > 0 ? epoch : "—"} · loss {loss > 0 ? loss.toFixed(4) : "—"}</div>
            <div className="mt-0.5">{threads.length} active thread{threads.length !== 1 ? "s" : ""}</div>
          </div>
        </div>
      </aside>

      {/* ── Main Chat Canvas ──────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* Session header bar */}
        <div className="h-11 shrink-0 border-b border-white/[0.05] bg-[#000000] flex items-center px-4 gap-3">
          <MessageSquare className="size-3.5 text-[#0A84FF]/55 shrink-0" />
          <span className="text-[12px] font-medium text-white/55 truncate">{activeThread?.name ?? "Session"}</span>
          {activeMessages.length > 0 && (
            <span className="text-[10px] text-white/20 tabular-nums shrink-0">{activeMessages.length} messages</span>
          )}
          <div className="flex-1" />
          {/* Live training ticker */}
          {liveSample && (
            <div className="hidden sm:flex items-center gap-1.5 max-w-[220px] overflow-hidden">
              <span className="size-1.5 rounded-full bg-[#30D158] shrink-0 animate-pulse" />
              <Wand2 className="size-3 text-white/18 shrink-0" />
              <code className="text-[9px] font-mono text-[#30D158]/45 truncate">{liveSample}</code>
            </div>
          )}
        </div>

        {/* Messages thread */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {activeMessages.length === 0 && !loading && (
            <div className="h-full flex flex-col items-center justify-center pb-16 px-8">
              <div className="size-16 rounded-2xl bg-[#0a0a0a] border border-white/[0.06] flex items-center justify-center mb-5">
                <Sparkles className="size-7 text-[#0A84FF]/40" />
              </div>
              <div className="text-[15px] font-semibold text-white/45 text-center mb-2">
                {epoch > 0 ? "Ready to chat" : "Train first"}
              </div>
              <div className="text-[12px] text-white/18 text-center max-w-[240px] leading-relaxed">
                {epoch > 0
                  ? `${modelLabel} — send a message to begin.`
                  : "Head to Training Lab, start training, then return here."}
              </div>
            </div>
          )}

          <div className="px-4 py-5 space-y-5 max-w-2xl mx-auto w-full">
            {activeMessages.map((m) => (
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
                    : "bg-[#0d0d0d] border border-white/[0.06] text-white/72 rounded-tl-sm"
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
                <div className="rounded-2xl rounded-tl-sm px-4 py-3 bg-[#0d0d0d] border border-white/[0.06] flex gap-1.5 items-center">
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

            {/* Model selector row — always rendered */}
            <div className="flex items-center gap-2 px-1">
              <span className="text-[10px] text-white/20 font-medium shrink-0">Profile:</span>
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
                    <div className="absolute bottom-full left-0 mb-1.5 w-56 bg-[#0e0e0e] border border-white/[0.08] rounded-xl shadow-2xl overflow-hidden z-50">
                      <div className="px-3 py-2 border-b border-white/[0.05]">
                        <span className="text-[9px] text-white/25 font-medium tracking-[0.12em] uppercase">Model Profiles</span>
                      </div>
                      <div className="py-1">
                        {modelOptions.map((o) => (
                          <button
                            key={o.id}
                            onClick={() => handleSelectModel(o.id, o.label)}
                            className={`w-full text-left px-3 py-2 text-[11px] hover:bg-white/[0.06] transition-colors truncate flex items-center gap-2 ${
                              selectedModelId === o.id ? "text-[#0A84FF]/90" : "text-white/55 hover:text-white/90"
                            }`}
                          >
                            {selectedModelId === o.id && <span className="size-1.5 rounded-full bg-[#0A84FF] shrink-0" />}
                            <span className="truncate">{o.label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <span className="h-6 px-2.5 flex items-center rounded-lg bg-white/[0.02] border border-white/[0.04] text-[10px] font-mono text-white/16 select-none cursor-default">
                  No Profiles Configured
                </span>
              )}

              <div className="flex-1" />

              {activeMessages.length > 0 && (
                <button
                  onClick={clearCurrentSession}
                  className="flex items-center gap-1 h-6 px-2.5 rounded-lg text-[10px] text-white/18 hover:text-red-400/65 hover:bg-red-400/8 border border-transparent hover:border-red-400/12 transition-all"
                >
                  <Trash2 className="size-2.5" />
                  Clear Session
                </button>
              )}
            </div>

            {/* Main input */}
            <div className={`flex items-end bg-[#0a0a0a] rounded-2xl border transition-colors ${
              isTraining ? "border-[#0A84FF]/15" : "border-white/[0.07] focus-within:border-white/[0.12]"
            }`}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
                }}
                rows={1}
                placeholder={epoch > 0 ? "Send a message…" : "Train a model first…"}
                className="flex-1 bg-transparent px-4 py-3 text-[13px] font-mono text-white/80 placeholder:text-white/15 focus:outline-none resize-none leading-relaxed min-h-[20px] max-h-28"
              />
              <div className="px-2 py-2 shrink-0">
                <button
                  onClick={send}
                  disabled={!input.trim() || loading}
                  className={`size-8 rounded-xl flex items-center justify-center transition-all ${
                    input.trim() && !loading
                      ? "bg-[#0A84FF] text-white hover:bg-[#409CFF] shadow-[0_0_14px_rgba(10,132,255,0.28)]"
                      : "bg-white/[0.04] text-white/16"
                  }`}
                >
                  {loading ? <Loader2 className="size-3.5 animate-spin" /> : <Send className="size-3.5" />}
                </button>
              </div>
            </div>

            {/* Status strip */}
            <div className="flex items-center justify-between px-1">
              <div className="text-[9px] text-white/12 font-mono tabular-nums">
                {epoch > 0 ? `ep ${epoch} · loss ${loss.toFixed(4)}` : "untrained"}
                {isTraining && <span className="text-[#0A84FF]/40 ml-1.5">· training…</span>}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
