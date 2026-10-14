// Thin client for ComfyUI's HTTP and websocket API. Fletch is served by ComfyUI itself,
// so every call is same-origin (the Vite dev server proxies them in development).

import type { ApiWorkflow } from "./workflow";

export interface ImageRef {
  filename: string;
  subfolder: string;
  type: string;
}

export type RunEvent =
  | { kind: "progress"; value: number; max: number }
  | { kind: "preview"; url: string }
  | { kind: "image"; image: ImageRef }
  | { kind: "done" }
  | { kind: "error"; message: string };

export const clientId = crypto.randomUUID();

// A combo input is ["a", "b"] in older ComfyUI and ["COMBO", { options: [...] }] in newer builds.
function comboOptions(spec: unknown): string[] {
  if (!Array.isArray(spec)) return [];
  const [first, second] = spec;
  if (Array.isArray(first)) return first;
  if (first === "COMBO" && second && Array.isArray(second.options)) return second.options;
  return [];
}

export async function listCheckpoints(): Promise<string[]> {
  const res = await fetch("/object_info/CheckpointLoaderSimple");
  if (!res.ok) throw new Error(`ComfyUI answered ${res.status}`);
  const info = await res.json();
  return comboOptions(info.CheckpointLoaderSimple?.input?.required?.ckpt_name);
}

export async function queuePrompt(prompt: ApiWorkflow): Promise<string> {
  const res = await fetch("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, client_id: clientId }),
  });
  const body = await res.json();
  if (!res.ok) {
    const detail = body?.error?.message ?? JSON.stringify(body);
    throw new Error(`ComfyUI rejected the workflow: ${detail}`);
  }
  return body.prompt_id;
}

export const imageUrl = (img: ImageRef) =>
  `/view?${new URLSearchParams({ filename: img.filename, subfolder: img.subfolder, type: img.type })}`;

// One websocket for the page; run listeners are keyed by prompt id.
type Listener = (e: RunEvent) => void;
const listeners = new Map<string, Listener>();
let current: string | null = null; // the prompt ComfyUI is executing now, for preview frames

let socket: WebSocket | null = null;

// Opens the page's websocket once (later calls only swap the status callback) and reconnects on drop.
export function connect(onStatus: (connected: boolean) => void) {
  statusCallback = onStatus;
  if (!socket) open();
}

let statusCallback: (connected: boolean) => void = () => {};

function open() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = (socket = new WebSocket(`${proto}://${location.host}/ws?clientId=${clientId}`));
  ws.binaryType = "blob";
  ws.onopen = () => statusCallback(true);
  ws.onclose = () => {
    statusCallback(false);
    setTimeout(open, 2000);
  };
  ws.onmessage = (msg) => {
    if (msg.data instanceof Blob) {
      // Binary frames: 4-byte event type (1 = preview image), 4-byte format, then the image.
      if (current) listeners.get(current)?.({ kind: "preview", url: URL.createObjectURL(msg.data.slice(8)) });
      return;
    }
    const { type, data } = JSON.parse(msg.data);
    const id: string | undefined = data?.prompt_id;
    if (type === "execution_start") current = id ?? null;
    const emit = id ? listeners.get(id) : undefined;
    if (!emit) return;
    if (type === "progress") emit({ kind: "progress", value: data.value, max: data.max });
    else if (type === "executed") for (const image of data.output?.images ?? []) emit({ kind: "image", image });
    else if (type === "execution_error") emit({ kind: "error", message: data.exception_message ?? "Generation failed" });
    else if ((type === "executing" && data.node === null) || type === "execution_success") emit({ kind: "done" });
  };
}

export function watch(promptId: string, listener: Listener) {
  listeners.set(promptId, listener);
  return () => listeners.delete(promptId);
}
