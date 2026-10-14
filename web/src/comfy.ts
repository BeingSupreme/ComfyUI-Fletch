// Thin client for ComfyUI's HTTP and websocket API. Fletch is served by ComfyUI itself,
// so every call is same-origin (the Vite dev server proxies them in development).

import type { ApiWorkflow, InputImage, NodeInfo, NodeInfos } from "./workflow";

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

async function getJson(url: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`ComfyUI answered ${res.status} for ${url}`);
  return res.json();
}

/** The choices ComfyUI offers for one node input, e.g. every file in models/checkpoints. */
export async function listOptions(classType: string, input: string): Promise<string[]> {
  try {
    const info = await getJson(`/object_info/${classType}`);
    return comboOptions(info[classType]?.input?.required?.[input]);
  } catch {
    return []; // node not installed
  }
}

export interface Catalog {
  checkpoints: string[];
  diffusionModels: string[];
  textEncoders: string[];
  vaes: string[];
  samplers: string[];
  schedulers: string[];
  upscaleModels: string[];
  /** Detection models for face fix (Impact Subpack's UltralyticsDetectorProvider). */
  detectors: string[];
  /** /object_info for the custom nodes add-ons use; missing means not installed. */
  nodes: NodeInfos;
}

/** Custom nodes whose presence (and defaults) add-ons depend on. */
const OPTIONAL_NODES = [
  "FaceDetailer",
  "UltralyticsDetectorProvider",
  "DWPreprocessor",
  "OpenposePreprocessor",
  "DepthAnythingV2Preprocessor",
  "DepthAnythingPreprocessor",
  "MiDaS-DepthMapPreprocessor",
];

async function nodeInfo(classType: string): Promise<NodeInfo | undefined> {
  try {
    return (await getJson(`/object_info/${encodeURIComponent(classType)}`))[classType];
  } catch {
    return undefined;
  }
}

export async function loadCatalog(): Promise<Catalog> {
  const [checkpoints, diffusionModels, textEncoders, vaes, samplers, schedulers, upscaleModels, detectors, infos] =
    await Promise.all([
      listOptions("CheckpointLoaderSimple", "ckpt_name"),
      listOptions("UNETLoader", "unet_name"),
      listOptions("DualCLIPLoader", "clip_name1"),
      listOptions("VAELoader", "vae_name"),
      listOptions("KSampler", "sampler_name"),
      listOptions("KSampler", "scheduler"),
      listOptions("UpscaleModelLoader", "model_name"),
      listOptions("UltralyticsDetectorProvider", "model_name"),
      Promise.all(OPTIONAL_NODES.map(nodeInfo)),
    ]);
  const nodes = Object.fromEntries(OPTIONAL_NODES.map((n, i) => [n, infos[i]]));
  return { checkpoints, diffusionModels, textEncoders, vaes, samplers, schedulers, upscaleModels, detectors, nodes };
}

/** Upload a picture to ComfyUI's input folder (under fletch/) so LoadImage can read it. */
export async function uploadImage(file: Blob, filename: string): Promise<InputImage> {
  const size = await new Promise<{ width: number; height: number }>((ok, fail) => {
    const img = new Image();
    img.onload = () => ok({ width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => fail(new Error("That file isn't an image Fletch can read."));
    img.src = URL.createObjectURL(file);
  });
  const form = new FormData();
  form.append("image", file, filename);
  form.append("subfolder", "fletch");
  form.append("type", "input");
  const res = await fetch("/upload/image", { method: "POST", body: form });
  if (!res.ok) throw new Error(`ComfyUI couldn't save the image (${res.status})`);
  const body = await res.json();
  return { name: body.subfolder ? `${body.subfolder}/${body.name}` : body.name, ...size };
}

export const inputImageUrl = (img: InputImage) => {
  const i = img.name.lastIndexOf("/");
  return imageUrl({ filename: img.name.slice(i + 1), subfolder: i < 0 ? "" : img.name.slice(0, i), type: "input" });
};

export interface ModelInfo {
  family: string | null;
  variant?: string | null;
  has_vae?: boolean;
  has_text_encoders?: boolean;
  reason?: string;
}

export const modelInfo = (folder: string, name: string): Promise<ModelInfo> =>
  getJson(`/fletch/api/model_info?${new URLSearchParams({ folder, name })}`);

export interface LoraInfo {
  name: string;
  family: "sd15" | "sdxl" | "flux" | "flux2" | null;
}

export const listLoras = (): Promise<LoraInfo[]> => getJson("/fletch/api/loras");

export const listControlnets = (): Promise<LoraInfo[]> => getJson("/fletch/api/controlnets");

/** Queue a workflow. `extra` is written into the saved PNG's metadata by SaveImage. */
export async function queuePrompt(prompt: ApiWorkflow, extra?: Record<string, unknown>): Promise<string> {
  const res = await fetch("/prompt", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, client_id: clientId, extra_data: extra ? { extra_pnginfo: extra } : undefined }),
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
