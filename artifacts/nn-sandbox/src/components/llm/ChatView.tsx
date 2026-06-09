import { useEffect, useRef, useState } from "react";
import { Send, Bot, User, Loader2, Sparkles, Wand2, Trash2 } from "lucide-react";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
}

export interface ModelOption {
  id: string;
  label: string;
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
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;
    const userMsg: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    try {
      const reply = await generate(text);
      const aiMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: reply.trim() || "(silence)",
      };
      setMessages((prev) => [...prev, aiMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-full flex flex-col bg-[#080808] overflow-hidden">

      {/* Live generation ticker */}
      {liveSample && (
        <div className="shrink-0 px-5 py-2 border-b border-white/[0.04] flex items-center gap-2.5 overflow-hidden">
          <span className="size-1.5 rounded-full bg-[#30D158] shrink-0 animate-pulse" />
          <Wand2 className="size-3 text-white/20 shrink-0" />
          <code className="text-[10px] font-mono text-[#30D158]/55 truncate leading-tight">
            {liveSample}
          </code>
        </div>
      )}

      {/* Messages thread */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">

        {/* Empty state */}
        {messages.length === 0 && !loading && (
          <div className="h-full flex flex-col items-center justify-center pb-16 px-8">
            <div className="size-16 rounded-2xl bg-[#141414] border border-white/[0.06] flex items-center justify-center mb-5">
              <Sparkles className="size-7 text-[#0A84FF]/50" />
            </div>
            <div className="text-[15px] font-semibold text-white/60 text-center mb-2">
              {epoch > 0 ? "Ready to chat" : "Train first"}
            </div>
            <div className="text-[12px] text-white/22 text-center max-w-[260px] leading-relaxed">
              {epoch > 0
                ? `${modelLabel} — send a message and the model will continue your text.`
                : "Head to Training Lab, press Train, then come back here to converse."}
            </div>
          </div>
        )}

        <div className="px-4 py-4 space-y-4">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex gap-3 ${m.role === "user" ? "flex-row-reverse" : ""}`}
            >
              {/* Avatar */}
              <div
                className={`size-7 rounded-xl flex items-center justify-center shrink-0 mt-0.5 ${
                  m.role === "user"
                    ? "bg-[#0A84FF]/12 border border-[#0A84FF]/18"
                    : "bg-white/[0.04] border border-white/[0.07]"
                }`}
              >
                {m.role === "user" ? (
                  <User className="size-3.5 text-[#0A84FF]" />
                ) : (
                  <Bot className="size-3.5 text-white/40" />
                )}
              </div>
              {/* Bubble */}
              <div
                className={`max-w-[82%] rounded-2xl px-4 py-2.5 text-[13px] leading-relaxed font-mono ${
                  m.role === "user"
                    ? "bg-[#0A84FF]/10 border border-[#0A84FF]/18 text-white/88 rounded-tr-sm"
                    : "bg-[#141414] border border-white/[0.06] text-white/75 rounded-tl-sm"
                }`}
              >
                {m.content}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex gap-3">
              <div className="size-7 rounded-xl bg-white/[0.04] border border-white/[0.07] flex items-center justify-center shrink-0 mt-0.5">
                <Bot className="size-3.5 text-white/40" />
              </div>
              <div className="rounded-2xl rounded-tl-sm px-4 py-3 bg-[#141414] border border-white/[0.06] flex gap-1.5 items-center">
                <span className="size-1.5 rounded-full bg-white/25 animate-pulse" />
                <span
                  className="size-1.5 rounded-full bg-white/25 animate-pulse"
                  style={{ animationDelay: "180ms" }}
                />
                <span
                  className="size-1.5 rounded-full bg-white/25 animate-pulse"
                  style={{ animationDelay: "360ms" }}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input bar */}
      <div className="shrink-0 px-3 pb-3 pt-2 bg-[#080808]">
        <div
          className={`flex items-end bg-[#111111] rounded-2xl border transition-colors ${
            isTraining
              ? "border-[#0A84FF]/18"
              : "border-white/[0.07] focus-within:border-white/[0.13]"
          }`}
        >
          {/* Model selector */}
          <div className="flex items-center pl-3 py-3 shrink-0">
            {modelOptions && modelOptions.length > 0 ? (
              <>
                <select
                  className="bg-transparent text-[10px] font-mono text-white/22 hover:text-white/50 focus:outline-none cursor-pointer max-w-[110px] leading-none"
                  onChange={(e) => {
                    const val = e.target.value;
                    if (val && onSelectModel) onSelectModel(val);
                    e.target.value = "";
                  }}
                  defaultValue=""
                >
                  <option value="" disabled className="bg-[#1a1a1a] text-white/50">
                    load model
                  </option>
                  {modelOptions.map((o) => (
                    <option
                      key={o.id}
                      value={o.id}
                      className="bg-[#1a1a1a] text-white/80"
                    >
                      {o.label}
                    </option>
                  ))}
                </select>
                <div className="w-px h-3 bg-white/[0.08] mx-2.5 shrink-0" />
              </>
            ) : null}
          </div>

          {/* Textarea */}
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            rows={1}
            placeholder={
              isTraining
                ? "Model is training — you can still send…"
                : "Send a message…"
            }
            className="flex-1 bg-transparent py-3 text-[13px] font-mono text-white/82 placeholder:text-white/18 focus:outline-none resize-none leading-relaxed min-h-[20px] max-h-28 pl-0 pr-0"
          />

          {/* Send button */}
          <div className="px-2 py-2 shrink-0">
            <button
              onClick={send}
              disabled={!input.trim() || loading}
              className={`size-8 rounded-xl flex items-center justify-center transition-all ${
                input.trim() && !loading
                  ? "bg-[#0A84FF] text-white hover:bg-[#409CFF] shadow-[0_0_16px_rgba(10,132,255,0.3)]"
                  : "bg-white/[0.04] text-white/18"
              }`}
            >
              {loading ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <Send className="size-3.5" />
              )}
            </button>
          </div>
        </div>

        {/* Status strip */}
        <div className="flex items-center justify-between px-1.5 mt-1.5">
          <div className="text-[10px] text-white/14 font-mono tabular-nums">
            {epoch > 0 ? `ep ${epoch} · loss ${loss.toFixed(4)}` : "untrained"}
            {isTraining && (
              <span className="text-[#0A84FF]/50 ml-1.5">· training…</span>
            )}
          </div>
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="text-[10px] text-white/14 hover:text-red-400/60 transition-colors flex items-center gap-1"
            >
              <Trash2 className="size-2.5" />
              clear
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
