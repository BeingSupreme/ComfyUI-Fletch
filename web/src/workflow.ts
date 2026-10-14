// Compiles the app's settings into a ComfyUI API-format workflow (the JSON POST /prompt accepts).
//
// The base pipeline comes from the model family's profile. Each stage reads and rewrites a
// small set of named slots (MODEL, CLIP, VAE, ...), so a feature such as a LoRA only has to
// say where it plugs in and everything downstream picks it up.

import { PROFILES, type Family, type Quality, type Shape } from "./profiles";

export type Link = [nodeId: string, outputIndex: number];

export interface ApiNode {
  class_type: string;
  inputs: Record<string, unknown>;
}

export type ApiWorkflow = Record<string, ApiNode>;

export type ModelFolder = "checkpoints" | "diffusion_models";

export interface LoraChoice {
  name: string;
  weight: number;
}

/** Overrides from the Advanced panel; anything left undefined comes from the profile. */
export interface Advanced {
  steps?: number;
  cfg?: number;
  sampler?: string;
  scheduler?: string;
  clipSkip?: number;
}

export interface Settings {
  model: { folder: ModelFolder; name: string };
  family: Family;
  positive: string;
  negative: string;
  shape: Shape;
  quality: Quality;
  loras: LoraChoice[];
  seed: number;
  advanced: Advanced;
}

/** Separate files a diffusion-only model needs (Flux shipped without encoders or VAE). */
export interface Parts {
  t5?: string;
  clipL?: string;
  vae?: string;
}

export const randomSeed = () => Math.floor(Math.random() * 2 ** 48);

/** The numbers a run will actually use: profile defaults with Advanced overrides applied. */
export function resolve(s: Pick<Settings, "family" | "shape" | "quality" | "advanced">) {
  const p = PROFILES[s.family];
  const [width, height] = p.sizes[s.shape];
  return {
    width,
    height,
    steps: s.advanced.steps ?? p.steps[s.quality],
    cfg: s.advanced.cfg ?? p.cfg,
    sampler: s.advanced.sampler ?? p.sampler,
    scheduler: s.advanced.scheduler ?? p.scheduler,
    clipSkip: s.advanced.clipSkip ?? p.clipSkip,
  };
}

const withPrefix = (prefix: string | undefined, text: string) =>
  prefix ? (text.trim() ? `${prefix}, ${text.trim()}` : prefix) : text.trim();

const basename = (path: string) => (path.split(/[\\/]/).pop() ?? path).toLowerCase();

/** Pick the text encoders and VAE a diffusion-only Flux model needs from what's on disk. */
export function pickFluxParts(textEncoders: string[], vaes: string[]): Parts {
  const t5s = textEncoders.filter((n) => /t5/i.test(n));
  return {
    // Prefer the fp8 T5 when both exist: same results for most prompts at half the memory.
    t5: t5s.find((n) => /fp8/i.test(n)) ?? t5s[0],
    clipL: textEncoders.find((n) => /clip_l/i.test(n)),
    vae: vaes.find((n) => /^ae\./.test(basename(n))) ?? vaes.find((n) => /flux/.test(basename(n))),
  };
}

/** Files still missing for a model, in words a person can act on. */
export function missingParts(s: Pick<Settings, "model" | "family">, parts: Parts): string[] {
  if (s.model.folder === "checkpoints") return [];
  if (!s.family.startsWith("flux")) return ["a full checkpoint (this file has no text encoders or VAE)"];
  const missing: string[] = [];
  if (!parts.t5) missing.push("a T5 text encoder (t5xxl) in models/text_encoders");
  if (!parts.clipL) missing.push("a CLIP-L text encoder (clip_l) in models/text_encoders");
  if (!parts.vae) missing.push("the Flux VAE (ae.safetensors) in models/vae");
  return missing;
}

class Graph {
  nodes: ApiWorkflow = {};
  private next = 1;
  add(class_type: string, inputs: Record<string, unknown>): string {
    const id = String(this.next++);
    this.nodes[id] = { class_type, inputs };
    return id;
  }
}

export function compile(s: Settings, parts: Parts = {}): ApiWorkflow {
  const missing = missingParts(s, parts);
  if (missing.length) throw new Error(`This model also needs ${missing.join(", ")}.`);

  const p = PROFILES[s.family];
  const r = resolve(s);
  const g = new Graph();
  let MODEL: Link, CLIP: Link, VAE: Link;

  // 1. Load
  if (s.model.folder === "checkpoints") {
    const ckpt = g.add("CheckpointLoaderSimple", { ckpt_name: s.model.name });
    [MODEL, CLIP, VAE] = [[ckpt, 0], [ckpt, 1], [ckpt, 2]];
  } else {
    MODEL = [g.add("UNETLoader", { unet_name: s.model.name, weight_dtype: "default" }), 0];
    CLIP = [g.add("DualCLIPLoader", { clip_name1: parts.t5, clip_name2: parts.clipL, type: "flux" }), 0];
    VAE = [g.add("VAELoader", { vae_name: parts.vae }), 0];
  }

  // 2. Patch the model: LoRAs chain in order
  for (const lora of s.loras) {
    if (!lora.name) continue;
    const n = g.add("LoraLoader", {
      model: MODEL,
      clip: CLIP,
      lora_name: lora.name,
      strength_model: lora.weight,
      strength_clip: lora.weight,
    });
    [MODEL, CLIP] = [[n, 0], [n, 1]];
  }
  if (r.clipSkip > 1) CLIP = [g.add("CLIPSetLastLayer", { clip: CLIP, stop_at_clip_layer: -r.clipSkip }), 0];

  // 3. Conditioning
  let POSITIVE: Link = [g.add("CLIPTextEncode", { clip: CLIP, text: withPrefix(p.positivePrefix, s.positive) }), 0];
  if (p.guidance !== undefined) POSITIVE = [g.add("FluxGuidance", { conditioning: POSITIVE, guidance: p.guidance }), 0];
  const negativeText = p.negative ? withPrefix(p.negativePrefix, s.negative) : "";
  const NEGATIVE: Link = [g.add("CLIPTextEncode", { clip: CLIP, text: negativeText }), 0];

  // 4. Latent source
  const LATENT: Link = [g.add(p.latent, { width: r.width, height: r.height, batch_size: 1 }), 0];

  // 5. Sample, 6. decode, 7. save
  const sampled = g.add("KSampler", {
    model: MODEL,
    positive: POSITIVE,
    negative: NEGATIVE,
    latent_image: LATENT,
    seed: s.seed,
    steps: r.steps,
    cfg: r.cfg,
    sampler_name: r.sampler,
    scheduler: r.scheduler,
    denoise: 1,
  });
  const IMAGE: Link = [g.add("VAEDecode", { samples: [sampled, 0], vae: VAE }), 0];
  g.add("SaveImage", { images: IMAGE, filename_prefix: "fletch" });
  return g.nodes;
}
