/// <reference lib="webworker" />
// Pro Mode inference worker. Loads a small instruction-tuned LLM via
// @huggingface/transformers and streams tokens back to the UI.
//
// Messages (UI -> worker):
//   { type: "load", modelId, device }
//   { type: "generate", id, prompt, maxNewTokens?, temperature?, topP? }
//   { type: "stop" }
//
// Messages (worker -> UI):
//   { type: "status", stage, message? }         // device decision, ready, error
//   { type: "progress", file, progress, loaded, total }
//   { type: "ready", modelId, device }
//   { type: "token", id, delta, text }          // streaming token
//   { type: "done", id, text, tokensPerSecond, tokensGenerated }
//   { type: "error", id?, message }

import {
  pipeline,
  TextStreamer,
  env,
  type TextGenerationPipeline,
  type ProgressInfo,
} from "@huggingface/transformers";

// We fetch model files from the HF Hub directly. Disable local model lookup
// which would otherwise attempt to resolve /models/... against our own origin.
env.allowLocalModels = false;
env.allowRemoteModels = true;

type Device = "webgpu" | "wasm";

interface LoadMsg {
  type: "load";
  modelId: string;
  device?: Device;
}

interface GenerateMsg {
  type: "generate";
  id: string;
  prompt: string;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  history?: { role: "user" | "assistant" | "system"; content: string }[];
}

interface StopMsg {
  type: "stop";
}

type InMsg = LoadMsg | GenerateMsg | StopMsg;

let generator: TextGenerationPipeline | null = null;
let activeDevice: Device = "wasm";
let currentModelId = "";
let stopRequested = false;

function post(msg: unknown, transfer: Transferable[] = []) {
  (self as unknown as Worker).postMessage(msg, transfer);
}

async function loadModel(modelId: string, preferred: Device) {
  currentModelId = modelId;
  // Detect WebGPU availability. `navigator.gpu` is exposed inside workers when
  // supported; otherwise we transparently fall back to WASM.
  const hasWebGPU =
    typeof (navigator as Navigator & { gpu?: unknown }).gpu !== "undefined";
  const device: Device = preferred === "webgpu" && hasWebGPU ? "webgpu" : "wasm";
  activeDevice = device;

  post({
    type: "status",
    stage: "device",
    device,
    hasWebGPU,
    message:
      device === "webgpu"
        ? "WebGPU detected — initializing accelerated runtime."
        : "WebGPU unavailable — falling back to WASM (CPU).",
  });

  try {
    generator = (await pipeline("text-generation", modelId, {
      device,
      // q4f16 = 4-bit weights with fp16 activations. Required for compatibility
      // with AMD/Intel Mac GPUs; plain "q4" silently hangs on first prompt on
      // those devices. WASM fallback uses q8 which is the most stable CPU dtype.
      dtype: device === "webgpu" ? "q4f16" : "q8",
      progress_callback: (p: ProgressInfo) => {
        // Stream download / init progress back to the UI.
        const anyP = p as unknown as {
          status?: string;
          file?: string;
          progress?: number;
          loaded?: number;
          total?: number;
          name?: string;
        };
        post({
          type: "progress",
          status: anyP.status,
          file: anyP.file ?? anyP.name,
          progress: typeof anyP.progress === "number" ? anyP.progress : 0,
          loaded: anyP.loaded ?? 0,
          total: anyP.total ?? 0,
        });
      },
    })) as unknown as TextGenerationPipeline;

    post({ type: "ready", modelId, device });
  } catch (err) {
    post({
      type: "error",
      message: `Failed to load model: ${
        err instanceof Error ? err.message : String(err)
      }`,
    });
  }
}

async function generate(msg: GenerateMsg) {
  if (!generator) {
    post({ type: "error", id: msg.id, message: "Model not loaded yet." });
    return;
  }
  stopRequested = false;

  const messages = [
    {
      role: "system" as const,
      content:
        "You are a helpful, concise assistant running entirely inside the user's browser.",
    },
    ...(msg.history ?? []),
    { role: "user" as const, content: msg.prompt },
  ];

  const tokenizer = (generator as unknown as {
    tokenizer: {
      apply_chat_template: (
        messages: unknown,
        opts: { tokenize: boolean; add_generation_prompt: boolean },
      ) => string;
      decode: (ids: number[] | bigint[], opts?: { skip_special_tokens?: boolean }) => string;
    };
  }).tokenizer;

  const inputs = tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
  });

  const startedAt = performance.now();
  let tokensGenerated = 0;
  let accumulated = "";

  const streamer = new TextStreamer(tokenizer as never, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text: string) => {
      if (!text) return;
      tokensGenerated++;
      accumulated += text;
      post({
        type: "token",
        id: msg.id,
        delta: text,
        text: accumulated,
        tokensGenerated,
      });
    },
  });

  try {
    const result = await generator(inputs, {
      max_new_tokens: msg.maxNewTokens ?? 256,
      do_sample: true,
      temperature: msg.temperature ?? 0.7,
      top_p: msg.topP ?? 0.9,
      streamer,
      // Stop when we have produced the full answer or user interrupted.
      stopping_criteria: {
        _call: () => stopRequested,
      } as never,
    });

    // Extract final text from pipeline output as a fallback to whatever we
    // accumulated from the streamer.
    let finalText = accumulated;
    if (Array.isArray(result) && result[0]) {
      const first = result[0] as { generated_text?: unknown };
      const gt = first.generated_text;
      if (typeof gt === "string" && gt.length > accumulated.length) {
        finalText = gt;
      } else if (Array.isArray(gt)) {
        const last = gt[gt.length - 1] as { content?: string } | undefined;
        if (last && typeof last.content === "string") finalText = last.content;
      }
    }

    const elapsedSec = (performance.now() - startedAt) / 1000;
    const tps = elapsedSec > 0 ? tokensGenerated / elapsedSec : 0;

    post({
      type: "done",
      id: msg.id,
      text: finalText,
      tokensGenerated,
      tokensPerSecond: tps,
      device: activeDevice,
      modelId: currentModelId,
    });
  } catch (err) {
    post({
      type: "error",
      id: msg.id,
      message: `Generation failed: ${
        err instanceof Error ? err.message : String(err)
      }`,
    });
  }
}

self.addEventListener("message", (e: MessageEvent<InMsg>) => {
  const msg = e.data;
  if (!msg || !("type" in msg)) return;
  switch (msg.type) {
    case "load":
      void loadModel(msg.modelId, msg.device ?? "webgpu");
      break;
    case "generate":
      void generate(msg);
      break;
    case "stop":
      stopRequested = true;
      break;
  }
});

export {};
