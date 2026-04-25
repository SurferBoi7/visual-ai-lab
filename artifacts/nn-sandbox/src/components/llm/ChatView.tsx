import { useEffect, useRef, useState } from "react";
import { Send, Bot, User, Loader2, Sparkles, Wand2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
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
    <div className="rounded-2xl border border-slate-700 bg-slate-800/50 backdrop-blur-md flex flex-col h-[calc(100vh-220px)] md:h-[600px] overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/80 bg-slate-900/40">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="size-8 rounded-lg bg-gradient-to-br from-sky-500/30 to-emerald-500/30 border border-sky-400/30 flex items-center justify-center shrink-0">
            <Bot className="size-4 text-sky-300" />
          </div>
          <div className="min-w-0">
            <div className="text-sm font-semibold text-slate-100 truncate">
              {modelLabel}
            </div>
            <div className="text-[11px] text-slate-400 truncate tabular-nums">
              epoch {epoch} · loss {loss.toFixed(3)}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="flex items-center gap-1 text-[11px] text-apple-mid hover:text-red-400 transition-colors px-2 py-1 rounded-lg hover:bg-red-500/10"
              aria-label="Clear chat"
            >
              <Trash2 className="size-3" />
              Clear
            </button>
          )}
          <span
            className={`text-[10px] px-2 py-0.5 rounded-full border ${
              isTraining
                ? "bg-sky-500/15 text-sky-300 border-sky-500/30 animate-pulse"
                : "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
            }`}
          >
            {isTraining ? "training…" : "idle"}
          </span>
        </div>
      </div>

      {/* Live generation banner */}
      <div className="px-4 py-2.5 border-b border-slate-700/60 bg-slate-950/40">
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-slate-500 mb-1">
          <Wand2 className="size-3 text-amber-400" />
          Live Generation
        </div>
        <div className="font-mono text-[11px] text-emerald-200/90 leading-relaxed break-words min-h-[28px] whitespace-pre-wrap">
          {liveSample || (
            <span className="text-slate-600">
              start training to see the model dream…
            </span>
          )}
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center px-6 py-10">
            <div className="size-12 rounded-2xl bg-gradient-to-br from-sky-500/20 to-emerald-500/20 border border-slate-700 flex items-center justify-center mb-3">
              <Sparkles className="size-5 text-sky-300" />
            </div>
            <div className="text-sm font-semibold text-slate-100">
              Train, then chat
            </div>
            <div className="text-xs text-slate-400 mt-1 max-w-xs">
              Hit <span className="text-sky-300 font-semibold">Train</span> to
              fit the model on your corpus, then send a prompt — the network
              continues your text character by character.
            </div>
          </div>
        )}

        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex gap-2.5 ${m.role === "user" ? "flex-row-reverse" : ""}`}
          >
            <div
              className={`size-8 rounded-lg flex items-center justify-center shrink-0 ${
                m.role === "user"
                  ? "bg-sky-500/20 border border-sky-400/30"
                  : "bg-slate-700/60 border border-slate-600"
              }`}
            >
              {m.role === "user" ? (
                <User className="size-4 text-sky-300" />
              ) : (
                <Bot className="size-4 text-slate-200" />
              )}
            </div>
            <div className={`max-w-[78%]`}>
              <div
                className={`rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed font-mono ${
                  m.role === "user"
                    ? "bg-sky-500/15 border border-sky-400/30 text-sky-50 rounded-tr-sm"
                    : "bg-slate-900/70 border border-slate-700 text-slate-100 rounded-tl-sm"
                }`}
              >
                {m.content}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex gap-2.5">
            <div className="size-8 rounded-lg bg-slate-700/60 border border-slate-600 flex items-center justify-center">
              <Bot className="size-4 text-slate-200" />
            </div>
            <div className="rounded-2xl rounded-tl-sm px-3.5 py-3 bg-slate-900/70 border border-slate-700 flex gap-1">
              <span className="size-1.5 rounded-full bg-slate-400 animate-pulse" />
              <span
                className="size-1.5 rounded-full bg-slate-400 animate-pulse"
                style={{ animationDelay: "150ms" }}
              />
              <span
                className="size-1.5 rounded-full bg-slate-400 animate-pulse"
                style={{ animationDelay: "300ms" }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="border-t border-slate-700/80 bg-slate-900/40 p-3">
        <div className="flex items-end gap-2">
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
            placeholder="Type a prompt — the model continues it…"
            className="flex-1 min-h-[44px] max-h-32 rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2.5 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40 resize-none font-mono"
          />
          <Button
            onClick={send}
            disabled={!input.trim() || loading}
            className="min-h-[44px] min-w-[44px] rounded-xl gap-1.5"
          >
            {loading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <Send className="size-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
