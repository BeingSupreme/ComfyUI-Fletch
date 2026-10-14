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
  /** FLUX.2 Klein size ("4b" or "9b"), which decides its text encoder. */
  variant?: string | null;
  positive: string;
  negative: string;
  shape: Shape;
  quality: Quality;
  loras: LoraChoice[];
  seed: number;
  advanced: Advanced;
}

/** Separate files a diffusion-only model needs (Flux ships without encoders or VAE). */
export interface Parts {
  t5?: string;
  clipL?: string;
  qwen?: string;
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

const QWEN = { "4b": /qwen.?3.?4b/i, "9b": /qwen.?3.?8b/i } as Record<string, RegExp>;
const QWEN_FILE = { "4b": "qwen_3_4b.safetensors", "9b": "qwen_3_8b_fp8mixed.safetensors" } as Record<string, string>;

/** Pick the text encoders and VAE a diffusion-only Flux model needs from what's on disk. */
export function pickParts(family: Family, variant: string | null | undefined, textEncoders: string[], vaes: string[]): Parts {
  if (PROFILES[family].arch === "flux2") {
    const pattern = (variant && QWEN[variant]) || /qwen.?3/i;
    return {
      qwen: textEncoders.find((n) => pattern.test(basename(n))),
      vae: vaes.find((n) => /flux.?2/.test(basename(n))),
    };
  }
  const t5s = textEncoders.filter((n) => /t5/i.test(n));
  return {
    // Prefer the fp8 T5 when both exist: same results for most prompts at half the memory.
    t5: t5s.find((n) => /fp8/i.test(n)) ?? t5s[0],
    clipL: textEncoders.find((n) => /clip_l/i.test(n)),
    vae: vaes.find((n) => /^ae\./.test(basename(n))) ?? vaes.find((n) => /flux/.test(basename(n)) && !/flux.?2/.test(basename(n))),
  };
}

/**
 * Why a model can't run yet, as sentences a person can act on (empty when it's ready).
 * `hasTextEncoders` comes from detection; a checkpoint without them must be loaded as a
 * diffusion model, which ComfyUI only lists from models/diffusion_models.
 */
export function modelProblems(
  s: Pick<Settings, "model" | "family" | "variant">,
  parts: Parts,
  hasTextEncoders?: boolean,
): string[] {
  const arch = PROFILES[s.family].arch;
  if (s.model.folder === "checkpoints") {
    if (hasTextEncoders !== false || arch === "sd") return [];
    return [
      "This file has no text encoder built in, so it has to load as a diffusion model. Move it to models/diffusion_models, restart ComfyUI, and pick it under Diffusion models.",
    ];
  }
  if (arch === "sd") return ["This file has no text encoders or VAE. Use the full checkpoint for this model instead."];
  const problems: string[] = [];
  if (arch === "flux2") {
    const want = (s.variant && QWEN_FILE[s.variant]) || "qwen_3_4b.safetensors or qwen_3_8b_fp8mixed.safetensors";
    if (!parts.qwen) problems.push(`Add the Qwen3 text encoder (${want}) to models/text_encoders.`);
    if (!parts.vae) problems.push("Add the Flux 2 VAE (flux2-vae.safetensors) to models/vae.");
    return problems;
  }
  if (!parts.t5) problems.push("Add a T5 text encoder (t5xxl) to models/text_encoders.");
  if (!parts.clipL) problems.push("Add a CLIP-L text encoder (clip_l) to models/text_encoders.");
  if (!parts.vae) problems.push("Add the Flux VAE (ae.safetensors) to models/vae.");
  return problems;
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

export function compile(s: Settings, parts: Parts = {}, hasTextEncoders?: boolean): ApiWorkflow {
  const problems = modelProblems(s, parts, hasTextEncoders);
  if (problems.length) throw new Error(problems.join(" "));

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
    CLIP =
      p.arch === "flux2"
        ? [g.add("CLIPLoader", { clip_name: parts.qwen, type: "flux2", device: "default" }), 0]
        : [g.add("DualCLIPLoader", { clip_name1: parts.t5, clip_name2: parts.clipL, type: "flux" }), 0];
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
  let NEGATIVE: Link;
  if (p.negative) NEGATIVE = [g.add("CLIPTextEncode", { clip: CLIP, text: withPrefix(p.negativePrefix, s.negative) }), 0];
  else if (p.arch === "flux2") NEGATIVE = [g.add("ConditioningZeroOut", { conditioning: POSITIVE }), 0];
  else NEGATIVE = [g.add("CLIPTextEncode", { clip: CLIP, text: "" }), 0];

  // 4. Latent source
  const LATENT: Link = [g.add(p.latent, { width: r.width, height: r.height, batch_size: 1 }), 0];

  // 5. Sample
  let SAMPLES: Link;
  if (p.arch === "flux2") {
    // FLUX.2 shifts its noise schedule by resolution, which Flux2Scheduler works out.
    SAMPLES = [
      g.add("SamplerCustomAdvanced", {
        noise: [g.add("RandomNoise", { noise_seed: s.seed }), 0],
        guider: [g.add("CFGGuider", { model: MODEL, positive: POSITIVE, negative: NEGATIVE, cfg: r.cfg }), 0],
        sampler: [g.add("KSamplerSelect", { sampler_name: r.sampler }), 0],
        sigmas: [g.add("Flux2Scheduler", { steps: r.steps, width: r.width, height: r.height }), 0],
        latent_image: LATENT,
      }),
      0,
    ];
  } else {
    SAMPLES = [
      g.add("KSampler", {
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
      }),
      0,
    ];
  }

  // 6. Decode, 7. save
  const IMAGE: Link = [g.add("VAEDecode", { samples: SAMPLES, vae: VAE }), 0];
  g.add("SaveImage", { images: IMAGE, filename_prefix: "fletch" });
  return g.nodes;
}
