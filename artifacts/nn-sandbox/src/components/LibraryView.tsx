import { useEffect, useState } from "react";
import {
  Library as LibraryIcon,
  Trash2,
  Upload,
  Network,
  MessageSquare,
  Inbox,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { getModels, deleteModel, type SavedModel } from "@/lib/storage";

interface Props {
  refreshKey: number;
  onLoad: (model: SavedModel) => void;
  onDeleted: () => void;
}

function formatDate(ms: number): string {
  const d = new Date(ms);
  const today = new Date();
  const sameDay =
    d.getFullYear() === today.getFullYear() &&
    d.getMonth() === today.getMonth() &&
    d.getDate() === today.getDate();
  if (sameDay) {
    return d.toLocaleTimeString(undefined, {
      hour: "numeric",
      minute: "2-digit",
    });
  }
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "2-digit",
  });
}

export function LibraryView({ refreshKey, onLoad, onDeleted }: Props) {
  const [models, setModels] = useState<SavedModel[] | null>(null);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setModels(null);
    getModels().then((m) => {
      if (!cancelled) setModels(m);
    });
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  const handleDelete = async (id: string) => {
    setPendingDelete(id);
    await deleteModel(id);
    setPendingDelete(null);
    setModels((prev) => (prev ? prev.filter((m) => m.id !== id) : prev));
    onDeleted();
  };

  return (
    <div className="rounded-2xl border border-slate-700 bg-slate-800/50 backdrop-blur-md p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="size-9 rounded-xl bg-gradient-to-br from-fuchsia-500/30 to-sky-500/30 border border-fuchsia-400/30 flex items-center justify-center">
            <LibraryIcon className="size-4 text-fuchsia-300" />
          </div>
          <div>
            <div className="text-sm font-semibold text-slate-100">
              Model Library
            </div>
            <div className="text-[11px] text-slate-400">
              Saved locally in your browser.
            </div>
          </div>
        </div>
        <span className="text-[11px] text-slate-400 tabular-nums">
          {models ? `${models.length} saved` : ""}
        </span>
      </div>

      {models === null && (
        <div className="flex items-center justify-center py-12 text-slate-500">
          <Loader2 className="size-5 animate-spin" />
        </div>
      )}

      {models !== null && models.length === 0 && (
        <div className="flex flex-col items-center justify-center text-center py-10">
          <div className="size-12 rounded-2xl bg-slate-900/60 border border-slate-700 flex items-center justify-center mb-3">
            <Inbox className="size-5 text-slate-500" />
          </div>
          <div className="text-sm font-semibold text-slate-200">
            No saved models yet
          </div>
          <div className="text-xs text-slate-400 mt-1 max-w-sm">
            Train something cool, hit <span className="text-sky-300">Save</span>{" "}
            in the header, name it, and stash it here for later.
          </div>
        </div>
      )}

      {models && models.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {models.map((m) => {
            const isMLP = m.type === "MLP";
            const Icon = isMLP ? Network : MessageSquare;
            const tone = isMLP
              ? "text-sky-300 bg-sky-500/15 border-sky-500/30"
              : "text-emerald-300 bg-emerald-500/15 border-emerald-500/30";
            return (
              <div
                key={m.id}
                className="rounded-xl border border-slate-700 bg-slate-900/60 p-4 space-y-3 flex flex-col"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-semibold text-slate-100 truncate">
                      {m.name}
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {formatDate(m.date)}
                    </div>
                  </div>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-full border inline-flex items-center gap-1 shrink-0 ${tone}`}
                  >
                    <Icon className="size-3" />
                    {m.type}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  <Stat label="Params" value={m.paramsCount.toLocaleString()} />
                  <Stat
                    label="Loss"
                    value={m.loss > 0 ? m.loss.toFixed(3) : "—"}
                  />
                  <Stat label="Epoch" value={m.epoch.toLocaleString()} />
                </div>

                <div className="flex gap-2 mt-auto">
                  <Button
                    size="sm"
                    onClick={() => onLoad(m)}
                    className="flex-1 gap-1.5 min-h-[40px] rounded-lg"
                  >
                    <Upload className="size-3.5" />
                    Load
                  </Button>
                  <Button
                    size="icon"
                    variant="secondary"
                    onClick={() => handleDelete(m.id)}
                    disabled={pendingDelete === m.id}
                    aria-label="Delete"
                    className="min-h-[40px] min-w-[40px] rounded-lg text-rose-300 hover:text-rose-200 hover:bg-rose-500/10"
                  >
                    {pendingDelete === m.id ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <Trash2 className="size-4" />
                    )}
                  </Button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/60 px-2 py-1.5 text-center">
      <div className="text-[9px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className="text-xs font-semibold tabular-nums text-slate-100 mt-0.5 truncate">
        {value}
      </div>
    </div>
  );
}
