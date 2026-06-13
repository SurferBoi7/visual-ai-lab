import { useEffect, useState } from "react";
import { X, Sparkles, ArrowRight, Loader2 } from "lucide-react";

const EMOJI_PRESETS = ["🧠", "⚡", "🤖", "💡", "🔬", "🎯", "🌊", "🔥", "🚀", "💎", "🦾", "🌐"];

interface Props {
  open: boolean;
  onClose: () => void;
  onSubmit: (name: string, emoji: string, description: string) => Promise<void>;
}

export function IdentityModal({ open, onClose, onSubmit }: Props) {
  const [name, setName] = useState("");
  const [emoji, setEmoji] = useState("🧠");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (open) {
      setName("");
      setEmoji("🧠");
      setDescription("");
      setBusy(false);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  const canSubmit = name.trim().length > 0 && !busy;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setBusy(true);
    try {
      await onSubmit(name.trim(), emoji, description.trim());
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-2xl border border-white/[0.08] bg-[#0d0d0d] shadow-2xl shadow-black/60 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-white/[0.06]">
          <div className="flex items-center gap-2.5">
            <div className="size-9 rounded-xl bg-[#0A84FF]/12 border border-[#0A84FF]/20 flex items-center justify-center">
              <Sparkles className="size-4 text-[#0A84FF]" />
            </div>
            <div>
              <div className="text-sm font-semibold text-white/88">Name your model</div>
              <div className="text-[11px] text-white/28">Give it an identity before training begins</div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="size-9 rounded-lg text-white/25 hover:text-white/65 hover:bg-white/[0.05] flex items-center justify-center transition-all"
            aria-label="Close"
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {/* Emoji picker */}
          <div className="space-y-2">
            <label className="text-[11px] text-white/35 font-medium">Icon</label>
            <div className="flex gap-1.5 flex-wrap">
              {EMOJI_PRESETS.map((e) => (
                <button
                  key={e}
                  onClick={() => setEmoji(e)}
                  className={`size-9 rounded-lg text-lg flex items-center justify-center transition-all ${
                    emoji === e
                      ? "bg-[#0A84FF]/18 border border-[#0A84FF]/35 ring-1 ring-[#0A84FF]/25"
                      : "bg-white/[0.04] border border-white/[0.06] hover:bg-white/[0.08]"
                  }`}
                >
                  {e}
                </button>
              ))}
            </div>
          </div>

          {/* Name */}
          <div className="space-y-1.5">
            <label className="text-[11px] text-white/35 font-medium" htmlFor="identity-name">
              Name
            </label>
            <input
              id="identity-name"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSubmit();
              }}
              placeholder="e.g. Galileo-7, Codex Mini, SciFi Bot"
              maxLength={60}
              className="w-full h-11 rounded-xl border border-white/[0.06] bg-[#080808] px-3 text-[13px] text-white/80 placeholder:text-white/18 focus:outline-none focus:border-[#0A84FF]/38 transition-colors"
            />
          </div>

          {/* Description */}
          <div className="space-y-1.5">
            <label className="text-[11px] text-white/35 font-medium" htmlFor="identity-desc">
              Description{" "}
              <span className="text-white/18">(optional)</span>
            </label>
            <input
              id="identity-desc"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleSubmit();
              }}
              placeholder="What will this model learn?"
              maxLength={120}
              className="w-full h-11 rounded-xl border border-white/[0.06] bg-[#080808] px-3 text-[13px] text-white/80 placeholder:text-white/18 focus:outline-none focus:border-[#0A84FF]/38 transition-colors"
            />
          </div>

          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="w-full h-12 rounded-xl bg-[#0A84FF] hover:bg-[#409CFF] text-white font-semibold text-[13px] flex items-center justify-center gap-2 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {busy ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <>
                <ArrowRight className="size-4" />
                Initialize &amp; Enter Training Lab
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
