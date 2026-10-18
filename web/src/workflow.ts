// Compiles the app's settings into a ComfyUI API-format workflow (the JSON POST /prompt accepts).
//
// The base pipeline comes from the model family's profile. Each stage reads and rewrites a
// small set of named slots (MODEL, CLIP, VAE, ...), so a feature such as a LoRA only has to
// say where it plugs in and everything downstream picks it up.

import { PROFILES, type Family, type Quality, type Shape } from "./profiles";
import { compileVideo } from "./video";

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

/** An image uploaded to ComfyUI's input folder, by the name LoadImage takes. */
export interface InputImage {
  name: string;
  width: number;
  height: number;
  /** Animated GIFs: how many frames, and how fast they play. */
  frames?: number;
  fps?: number;
}

/** Start from an image (img2img), optionally repainting only a masked area (inpaint). */
export interface SourceAddon {
  image: InputImage;
  /** Same picture with the area to repaint made transparent; LoadImage reads that as the mask. */
  masked?: InputImage;
  /** How much to change: 0 keeps the image, 1 ignores it. */
  strength: number;
}

export interface HiresAddon {
  scale: number;
  strength: number;
}

export interface UpscaleAddon {
  model: string;
}

export interface FaceFixAddon {
  detector: string;
  strength: number;
}

export type GuideMode = "canny" | "pose" | "depth" | "asis";

export interface GuideAddon {
  image: InputImage;
  mode: GuideMode;
  model: string;
  strength: number;
}

export interface Addons {
  source?: SourceAddon;
  hires?: HiresAddon;
  upscale?: UpscaleAddon;
  faceFix?: FaceFixAddon;
  guide?: GuideAddon;
}

/**
 * Node classes some add-ons need beyond ComfyUI's core, with the preprocessor each guide mode
 * uses. Custom nodes have many inputs that change between versions, so their values come from
 * the node's own defaults (ComfyUI's /object_info) and only the ones Fletch cares about are set.
 */
export const GUIDE_PREPROCESSORS: Record<GuideMode, string[]> = {
  canny: ["Canny"],
  pose: ["DWPreprocessor", "OpenposePreprocessor"],
  depth: ["DepthAnythingV2Preprocessor", "DepthAnythingPreprocessor", "MiDaS-DepthMapPreprocessor"],
  asis: [],
};

/** ComfyUI's description of one node class, as /object_info/<class> returns it. */
export interface NodeInfo {
  input?: { required?: Record<string, unknown[]>; optional?: Record<string, unknown[]> };
}

export type NodeInfos = Record<string, NodeInfo | undefined>;

/** Every widget input of a node at its default value; links are left for the caller. */
export function nodeDefaults(info: NodeInfo | undefined): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [name, spec] of Object.entries(info?.input?.required ?? {})) {
    const [type, opts] = spec as [unknown, Record<string, unknown> | undefined];
    if (Array.isArray(type)) out[name] = type[0];
    else if (type === "COMBO") out[name] = opts?.default ?? (opts?.options as unknown[] | undefined)?.[0];
    else if (["INT", "FLOAT", "BOOLEAN", "STRING"].includes(type as string)) out[name] = opts?.default ?? (type === "STRING" ? "" : 0);
  }
  return out;
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
  addons?: Addons;
  /** Video models: length in seconds. */
  seconds?: number;
  /** Wan 2.2 14B speed LoRAs (lightx2v), used for Fast: 4 steps at CFG 1. */
  speedLora?: { high: string; low: string };
}

/** Separate files a diffusion-only model needs (Flux ships without encoders or VAE). */
export interface Parts {
  t5?: string;
  clipL?: string;
  qwen?: string;
  vae?: string;
  /** UMT5 text encoder (Wan). */
  umt5?: string;
  /** Wan 2.2 14B: the two halves of the model pair. */
  high?: string;
  low?: string;
  /** LTX-2.3: Gemma 3 text encoder, distilled LoRA and 2x latent upscaler. */
  gemma?: string;
  distillLora?: string;
  upscaler?: string;
}

/** Files beyond text encoders and VAEs that some families need. */
export interface ExtraFiles {
  loras?: string[];
  latentUpscalers?: string[];
}

export const randomSeed = () => Math.floor(Math.random() * 2 ** 48);

const snap = (n: number) => Math.max(64, Math.round(n / 64) * 64);

/**
 * Output size. With a source image the size follows its aspect ratio at the family's native
 * pixel count (so inpaint masks line up); otherwise it comes from the chosen Shape.
 */
export function sizeFor(family: Family, shape: Shape, source?: { width: number; height: number }): [number, number] {
  const p = PROFILES[family];
  if (!source) return p.sizes[shape];
  const [w, h] = p.sizes.square;
  const ratio = source.width / source.height;
  return [snap(Math.sqrt(w * h * ratio)), snap(Math.sqrt((w * h) / ratio))];
}

/** The numbers a run will actually use: profile defaults with Advanced overrides applied. */
export function resolve(s: Pick<Settings, "family" | "shape" | "quality" | "advanced" | "addons" | "seconds" | "speedLora">) {
  const p = PROFILES[s.family];
  const [width, height] = sizeFor(s.family, s.shape, s.addons?.source?.image);
  const speed = !!s.speedLora;
  const v = p.video;
  const seconds = v ? (s.seconds && v.seconds.includes(s.seconds) ? s.seconds : v.seconds.includes(5) ? 5 : v.seconds[v.seconds.length - 1]) : 0;
  return {
    width,
    height,
    steps: s.advanced.steps ?? (speed ? 4 : p.steps[s.quality]),
    cfg: s.advanced.cfg ?? (speed ? 1 : p.cfg),
    seconds,
    frames: v ? seconds * v.fps + 1 : 1,
    sampler: s.advanced.sampler ?? p.sampler,
    scheduler: s.advanced.scheduler ?? p.scheduler,
    clipSkip: s.advanced.clipSkip ?? p.clipSkip,
  };
}

export const withPrefix = (prefix: string | undefined, text: string) =>
  prefix ? (text.trim() ? `${prefix}, ${text.trim()}` : prefix) : text.trim();

export const basename = (path: string) => (path.split(/[\\/]/).pop() ?? path).toLowerCase();

const QWEN = { "4b": /qwen.?3.?4b/i, "9b": /qwen.?3.?8b/i } as Record<string, RegExp>;
const QWEN_VL = /qwen.?3.?vl/i;
const QWEN_FILE = { "4b": "qwen_3_4b.safetensors", "9b": "qwen_3_8b_fp8mixed.safetensors" } as Record<string, string>;

/** The other half of a Wan 2.2 14B pair: same file name with "high" and "low" swapped. */
export function partnerName(name: string, from: "high" | "low"): string | null {
  const cut = Math.max(name.lastIndexOf("/"), name.lastIndexOf("\\")) + 1;
  const [dir, base] = [name.slice(0, cut), name.slice(cut)];
  const to = from === "high" ? "low" : "high";
  const swap = (word: string) => (word === word.toUpperCase() ? to.toUpperCase() : word[0] === word[0].toUpperCase() ? to[0].toUpperCase() + to.slice(1) : to);
  const near = new RegExp(`${from}(?=[\\s_.-]*noise)`, "i");
  const any = new RegExp(from, "i");
  const re = near.test(base) ? near : any.test(base) ? any : null;
  return re ? dir + base.replace(re, swap) : null;
}

/**
 * Pick the separate files a diffusion-only model needs from what's on disk: text encoders and
 * VAE, plus for Wan 2.2 14B the other model of its high/low-noise pair.
 */
export function pickParts(
  family: Family,
  variant: string | null | undefined,
  textEncoders: string[],
  vaes: string[],
  model?: string,
  diffusionModels: string[] = [],
  extra: ExtraFiles = {},
): Parts {
  const p = PROFILES[family];
  const arch = p.arch;
  if (arch === "ltx23") {
    const gemmas = textEncoders.filter((n) => /gemma.?3/i.test(basename(n)));
    const distills = (extra.loras ?? []).filter((n) => /ltx.?2.?3/.test(basename(n)) && /distill/.test(basename(n)));
    const ups = extra.latentUpscalers ?? [];
    return {
      gemma: gemmas.find((n) => /12b/i.test(n)) ?? gemmas[0],
      distillLora: distills[0],
      upscaler: ups.find((n) => /ltx.?2.?3/.test(basename(n)) && /spatial|upscal/.test(basename(n))) ?? ups.find((n) => /ltx/.test(basename(n))),
    };
  }
  if (arch === "wan22") {
    const umt5s = textEncoders.filter((n) => /umt5/i.test(n));
    const parts: Parts = {
      umt5: umt5s.find((n) => /fp8/i.test(n)) ?? umt5s[0],
      vae: vaes.find((n) => p.video!.vae!.test(basename(n))),
    };
    if (p.video!.experts && model) {
      const isLow = /low/i.test(basename(model)) && !/high/i.test(basename(model));
      const other = partnerName(model, isLow ? "low" : "high");
      const found = other && diffusionModels.includes(other) ? other : undefined;
      [parts.high, parts.low] = isLow ? [found, model] : [model, found];
    }
    return parts;
  }
  if (arch === "krea2") {
    return {
      qwen: textEncoders.find((n) => QWEN_VL.test(basename(n))),
      vae: vaes.find((n) => /qwen.?image/.test(basename(n))),
    };
  }
  if (arch === "flux2") {
    const pattern = (variant && QWEN[variant]) || /qwen.?3/i;
    return {
      qwen: textEncoders.find((n) => pattern.test(basename(n)) && !QWEN_VL.test(basename(n))),
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
  if (arch === "ltx23") {
    const problems: string[] = [];
    if (s.model.folder !== "checkpoints") problems.push("LTX-2.3 loads as a full checkpoint. Put it in models/checkpoints and pick it there.");
    if (!parts.gemma) problems.push("Add the Gemma 3 text encoder (gemma_3_12B_it_fp4_mixed.safetensors) to models/text_encoders.");
    if (!parts.distillLora)
      problems.push("Add LTX-2.3's distilled LoRA (ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors) to models/loras.");
    if (!parts.upscaler) problems.push("Add LTX-2.3's upscaler (ltx-2.3-spatial-upscaler-x2-1.1.safetensors) to models/latent_upscale_models.");
    return problems;
  }
  if (s.model.folder === "checkpoints") {
    if (hasTextEncoders !== false || arch === "sd") return [];
    return [
      "This file has no text encoder built in, so it has to load as a diffusion model. Move it to models/diffusion_models, restart ComfyUI, and pick it under Diffusion models.",
    ];
  }
  if (arch === "sd") return ["This file has no text encoders or VAE. Use the full checkpoint for this model instead."];
  const problems: string[] = [];
  if (arch === "wan22") {
    const v = PROFILES[s.family].video!;
    if (!parts.umt5) problems.push("Add the UMT5 text encoder (umt5_xxl_fp8_e4m3fn_scaled.safetensors) to models/text_encoders.");
    if (!parts.vae) problems.push(`Add the Wan VAE (${v.vaeFile}) to models/vae.`);
    if (v.experts && !(parts.high && parts.low)) {
      const want = partnerName(s.model.name, parts.low === s.model.name ? "low" : "high");
      problems.push(
        `Wan 2.2 14B comes as two models, high noise and low noise. Put the other one${want ? ` (${want.split(/[\\/]/).pop()})` : ""} in models/diffusion_models too.`,
      );
    }
    return problems;
  }
  if (arch === "krea2") {
    if (!parts.qwen) problems.push("Add the Qwen3-VL text encoder (qwen3vl_4b_fp8_scaled.safetensors) to models/text_encoders.");
    if (!parts.vae) problems.push("Add the Qwen Image VAE (qwen_image_vae.safetensors) to models/vae.");
    return problems;
  }
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

export class Graph {
  nodes: ApiWorkflow = {};
  private next = 1;
  add(class_type: string, inputs: Record<string, unknown>): string {
    const id = String(this.next++);
    this.nodes[id] = { class_type, inputs };
    return id;
  }
}

export const loadImage = (g: Graph, img: InputImage) => g.add("LoadImage", { image: img.name });

/** Where to save: a folder inside ComfyUI's output (`subfolder`) or anywhere else (`folder`). */
export interface Destination {
  subfolder?: string;
  folder?: string;
}

/** Everything stopping a run, the model's needs first, then the start picture for video. */
export function runProblems(s: Pick<Settings, "model" | "family" | "variant" | "addons">, parts: Parts, hasTextEncoders?: boolean): string[] {
  const problems = modelProblems(s, parts, hasTextEncoders);
  const v = PROFILES[s.family].video;
  const source = s.addons?.source;
  const gif = isAnimated(source?.image);
  if (!v && gif)
    problems.push("Animated GIFs work with the video models (Wan 2.2 or LTX-2.3). Pick one of those, or start from a still picture.");
  if (v?.image === "required" && !source)
    problems.push("This model turns a picture into a video. Turn on Start from an image and add one.");
  if (v?.image === "required" && gif)
    problems.push("The image-to-video pair animates a still picture. To restyle a GIF, pick the text-to-video pair, Wan 2.2 5B or LTX-2.3.");
  if (v?.image === "none" && source && !gif)
    problems.push("This model makes video from text, or restyles an animated GIF. Turn off Start from an image, or pick the image-to-video model.");
  return problems;
}

/** An animated GIF to restyle, rather than a still picture to start from. */
export const isAnimated = (img?: InputImage) => (img?.frames ?? 1) > 1;

export function compile(
  s: Settings,
  parts: Parts = {},
  hasTextEncoders?: boolean,
  nodes: NodeInfos = {},
  dest: Destination = {},
): ApiWorkflow {
  const problems = runProblems(s, parts, hasTextEncoders);
  if (problems.length) throw new Error(problems.join(" "));
  if (PROFILES[s.family].video) return compileVideo(s, parts, nodes, dest);

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
      p.arch === "flux2" || p.arch === "krea2"
        ? [g.add("CLIPLoader", { clip_name: parts.qwen, type: p.arch, device: "default" }), 0]
        : [g.add("DualCLIPLoader", { clip_name1: parts.t5, clip_name2: parts.clipL, type: "flux" }), 0];
    VAE = [g.add("VAELoader", { vae_name: parts.vae }), 0];
  }

  // 2. Patch the model: LoRAs chain in order
  for (const lora of s.loras) {
    if (!lora.name) continue;
    // Krea 2 LoRAs only touch the diffusion model, as in ComfyUI's template.
    if (p.arch === "krea2") {
      MODEL = [g.add("LoraLoaderModelOnly", { model: MODEL, lora_name: lora.name, strength_model: lora.weight }), 0];
      continue;
    }
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
  // Distilled models skip the negative at CFG 1; above that, a typed negative is used.
  if (p.negative || (r.cfg > 1 && s.negative.trim())) NEGATIVE = [g.add("CLIPTextEncode", { clip: CLIP, text: withPrefix(p.negativePrefix, s.negative) }), 0];
  else if (p.arch === "flux2" || p.arch === "krea2") NEGATIVE = [g.add("ConditioningZeroOut", { conditioning: POSITIVE }), 0];
  else NEGATIVE = [g.add("CLIPTextEncode", { clip: CLIP, text: "" }), 0];

  const addons = s.addons ?? {};

  // 3b. Pose or edge guide (ControlNet), applied to both prompts
  if (addons.guide) {
    const gd = addons.guide;
    let hint: Link = [loadImage(g, gd.image), 0];
    const pre = GUIDE_PREPROCESSORS[gd.mode].find((c) => c === "Canny" || nodes[c]);
    if (pre === "Canny") hint = [g.add("Canny", { image: hint, low_threshold: 0.4, high_threshold: 0.8 }), 0];
    else if (pre) hint = [g.add(pre, { ...nodeDefaults(nodes[pre]), image: hint }), 0];
    const applied = g.add("ControlNetApplyAdvanced", {
      positive: POSITIVE,
      negative: NEGATIVE,
      control_net: [g.add("ControlNetLoader", { control_net_name: gd.model }), 0],
      image: hint,
      strength: gd.strength,
      start_percent: 0,
      end_percent: 1,
      vae: VAE,
    });
    [POSITIVE, NEGATIVE] = [[applied, 0], [applied, 1]];
  }

  // 4. Latent source: empty, or the source image (masked for inpaint)
  let LATENT: Link;
  let denoise = 1;
  if (addons.source) {
    const src = addons.source;
    const loaded = loadImage(g, src.masked ?? src.image);
    const pixels: Link = [
      g.add("ImageScale", { image: [loaded, 0], upscale_method: "lanczos", width: r.width, height: r.height, crop: "disabled" }),
      0,
    ];
    LATENT = [g.add("VAEEncode", { pixels, vae: VAE }), 0];
    if (src.masked) {
      const mask: Link = [g.add("GrowMask", { mask: [loaded, 1], expand: 6, tapered_corners: true }), 0];
      LATENT = [g.add("SetLatentNoiseMask", { samples: LATENT, mask }), 0];
    }
    denoise = src.strength;
  } else {
    LATENT = [g.add(p.latent, { width: r.width, height: r.height, batch_size: 1 }), 0];
  }

  // 5. Sample (and again at a larger size for hires fix)
  const sample = (latent: Link, width: number, height: number, amount: number): Link => {
    if (p.arch === "flux2") {
      // FLUX.2 shifts its noise schedule by resolution, which Flux2Scheduler works out.
      // A partial pass keeps only the tail of that schedule.
      let sigmas: Link = [g.add("Flux2Scheduler", { steps: r.steps, width, height }), 0];
      if (amount < 1) sigmas = [g.add("SplitSigmasDenoise", { sigmas, denoise: amount }), 1];
      return [
        g.add("SamplerCustomAdvanced", {
          noise: [g.add("RandomNoise", { noise_seed: s.seed }), 0],
          guider: [g.add("CFGGuider", { model: MODEL, positive: POSITIVE, negative: NEGATIVE, cfg: r.cfg }), 0],
          sampler: [g.add("KSamplerSelect", { sampler_name: r.sampler }), 0],
          sigmas,
          latent_image: latent,
        }),
        0,
      ];
    }
    return [
      g.add("KSampler", {
        model: MODEL,
        positive: POSITIVE,
        negative: NEGATIVE,
        latent_image: latent,
        seed: s.seed,
        steps: r.steps,
        cfg: r.cfg,
        sampler_name: r.sampler,
        scheduler: r.scheduler,
        denoise: amount,
      }),
      0,
    ];
  };
  let SAMPLES = sample(LATENT, r.width, r.height, denoise);

  // 6. Decode
  let IMAGE: Link = [g.add("VAEDecode", { samples: SAMPLES, vae: VAE }), 0];

  // 6b. Hires fix: enlarge in pixel space (works for every latent format), then refine
  if (addons.hires) {
    const { scale, strength } = addons.hires;
    const big: Link = [g.add("ImageScaleBy", { image: IMAGE, upscale_method: "lanczos", scale_by: scale }), 0];
    const w = Math.round((r.width * scale) / 8) * 8;
    const h = Math.round((r.height * scale) / 8) * 8;
    SAMPLES = sample([g.add("VAEEncode", { pixels: big, vae: VAE }), 0], w, h, strength);
    IMAGE = [g.add("VAEDecode", { samples: SAMPLES, vae: VAE }), 0];
  }

  // 6c. Face fix (Impact Pack's FaceDetailer)
  if (addons.faceFix) {
    const detector = g.add("UltralyticsDetectorProvider", {
      ...nodeDefaults(nodes.UltralyticsDetectorProvider),
      model_name: addons.faceFix.detector,
    });
    IMAGE = [
      g.add("FaceDetailer", {
        ...nodeDefaults(nodes.FaceDetailer),
        image: IMAGE,
        model: MODEL,
        clip: CLIP,
        vae: VAE,
        positive: POSITIVE,
        negative: NEGATIVE,
        bbox_detector: [detector, 0],
        seed: s.seed,
        steps: Math.min(r.steps, 20),
        cfg: r.cfg,
        sampler_name: r.sampler,
        scheduler: r.scheduler,
        denoise: addons.faceFix.strength,
      }),
      0,
    ];
  }

  // 6d. Upscale with a model (ESRGAN and friends)
  if (addons.upscale) {
    const model = g.add("UpscaleModelLoader", { model_name: addons.upscale.model });
    IMAGE = [g.add("ImageUpscaleWithModel", { upscale_model: [model, 0], image: IMAGE }), 0];
  }

  // 7. Save
  if (dest.folder) g.add("FletchSaveImage", { images: IMAGE, folder: dest.folder, filename_prefix: "fletch" });
  else g.add("SaveImage", { images: IMAGE, filename_prefix: dest.subfolder ? `${dest.subfolder}/fletch` : "fletch" });
  return g.nodes;
}
