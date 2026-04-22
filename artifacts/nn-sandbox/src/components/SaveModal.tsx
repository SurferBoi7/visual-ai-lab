import { useEffect, useState } from "react";
import { X, Save, Library, Download, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  open: boolean;
  defaultName: string;
  modeLabel: string;
  hasModel: boolean;
  onClose: () => void;
  onSaveToLibrary: (name: string) => Promise<void> | void;
  onExportFile: (name: string) => Promise<void> | void;
}

export function SaveModal({
  open,
  defaultName,
  modeLabel,
  hasModel,
  onClose,
  onSaveToLibrary,
  onExportFile,
}: Props) {
  const [name, setName] = useState(defaultName);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (open) {
      setName(defaultName);
      setBusy(false);
    }
  }, [open, defaultName]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  const trimmed = name.trim();
  const canSave = hasModel && trimmed.length > 0 && !busy;

  const wrap = async (fn: (n: string) => Promise<void> | void) => {
    if (!canSave) return;
    setBusy(true);
    try {
      await fn(trimmed);
      onClose();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-2xl border border-slate-700 bg-slate-900/95 backdrop-blur-md shadow-2xl shadow-sky-500/10 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <div className="flex items-center gap-2.5">
            <div className="size-9 rounded-xl bg-gradient-to-br from-sky-500/30 to-emerald-500/30 border border-sky-400/30 flex items-center justify-center">
              <Sparkles className="size-4 text-sky-300" />
            </div>
            <div>
              <div className="text-sm font-semibold text-slate-100">
                Save your model
              </div>
              <div className="text-[11px] text-slate-400">
                Mode: {modeLabel}
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            className="size-9 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 flex items-center justify-center"
            aria-label="Close"
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          <div className="space-y-2">
            <label className="text-xs text-slate-400" htmlFor="save-name">
              Name
            </label>
            <input
              id="save-name"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. XOR Master"
              className="w-full min-h-[44px] rounded-xl border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/40"
              maxLength={60}
            />
          </div>

          {!hasModel && (
            <p className="text-[11px] text-amber-300/90">
              Train the model at least one epoch before saving.
            </p>
          )}

          <div className="flex flex-col sm:flex-row gap-2 pt-1">
            <Button
              onClick={() => wrap(onSaveToLibrary)}
              disabled={!canSave}
              className="flex-1 gap-1.5 min-h-[44px] rounded-xl"
            >
              <Library className="size-4" />
              Save to Library
            </Button>
            <Button
              onClick={() => wrap(onExportFile)}
              disabled={!canSave}
              variant="secondary"
              className="flex-1 gap-1.5 min-h-[44px] rounded-xl"
            >
              <Download className="size-4" />
              Export as File
            </Button>
          </div>

          <p className="text-[10px] text-slate-500 leading-relaxed pt-1 inline-flex items-start gap-1.5">
            <Save className="size-3 mt-0.5 shrink-0" />
            Library models live in your browser (IndexedDB). Exported files
            move anywhere — share, version, or import elsewhere.
          </p>
        </div>
      </div>
    </div>
  );
}
