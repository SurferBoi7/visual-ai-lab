// Native IndexedDB wrapper for the AI Sandbox model library.

export type SavedModelType = "MLP" | "Char-LM";

export interface MLPWeights {
  layers: number[];
  activation: string;
  weights: number[][][];
  biases: number[][];
  learningRate: number;
}

// Mirrors the Dataset shape in LLMArchitect.tsx. Stored alongside the
// flattened corpus so a save → load round trip can restore the user's
// individual datasets in the Dataset Manager UI instead of collapsing
// everything into a single "Base Training" file.
export interface SavedDataset {
  name: string;
  text: string;
  active: boolean;
}

export interface CharLMWeights {
  vocab: string[];
  tokenization?: "char" | "word";
  config: {
    vocabSize: number;
    contextSize: number;
    hiddenSize: number;
    learningRate: number;
  };
  weights: {
    W1: number[][];
    b1: number[];
    W2: number[][];
    b2: number[];
  };
  // Original corpus + temperature so the model can be used immediately on load.
  corpus?: string;
  temperature?: number;
  // Optional snapshot of the Dataset Manager state at save time. Older saves
  // pre-date this field, so it must remain optional and load paths must
  // tolerate its absence (and never derive datasets from `corpus`).
  datasets?: SavedDataset[];
}

export interface SavedModel {
  id: string;
  name: string;
  emoji?: string;
  description?: string;
  type: SavedModelType;
  date: number;
  paramsCount: number;
  loss: number;
  epoch: number;
  weights: MLPWeights | CharLMWeights;
}

const DB_NAME = "ai-sandbox";
const DB_VERSION = 1;
const STORE = "models";

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex("date", "date", { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(
  mode: IDBTransactionMode,
  fn: (store: IDBObjectStore) => IDBRequest<T> | Promise<T>,
): Promise<T> {
  return new Promise(async (resolve, reject) => {
    try {
      const db = await openDB();
      const t = db.transaction(STORE, mode);
      const store = t.objectStore(STORE);
      const result = fn(store);
      if (result instanceof Promise) {
        result.then(resolve, reject);
      } else {
        result.onsuccess = () => resolve(result.result);
        result.onerror = () => reject(result.error);
      }
      t.oncomplete = () => db.close();
    } catch (err) {
      reject(err);
    }
  });
}

export async function saveModel(model: SavedModel): Promise<void> {
  await tx<IDBValidKey>("readwrite", (store) => store.put(model));
}

export async function getModels(): Promise<SavedModel[]> {
  return tx<SavedModel[]>(
    "readonly",
    (store) =>
      new Promise<SavedModel[]>((resolve, reject) => {
        const req = store.getAll();
        req.onsuccess = () => {
          const all = (req.result as SavedModel[]) ?? [];
          all.sort((a, b) => b.date - a.date);
          resolve(all);
        };
        req.onerror = () => reject(req.error);
      }) as unknown as IDBRequest<SavedModel[]>,
  );
}

export async function loadModel(id: string): Promise<SavedModel | null> {
  return tx<SavedModel | null>(
    "readonly",
    (store) =>
      new Promise<SavedModel | null>((resolve, reject) => {
        const req = store.get(id);
        req.onsuccess = () => resolve((req.result as SavedModel) ?? null);
        req.onerror = () => reject(req.error);
      }) as unknown as IDBRequest<SavedModel | null>,
  );
}

export async function deleteModel(id: string): Promise<void> {
  await tx<undefined>("readwrite", (store) => store.delete(id));
}

export function makeId(): string {
  return `m-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
}
