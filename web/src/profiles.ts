// One profile per model family: how to load it and the defaults that make it look good.
// Adding a family means adding an entry here, not changing the compiler.

export type Family =
  | "sd15"
  | "sdxl"
  | "pony"
  | "illustrious"
  | "flux_dev"
  | "flux_schnell"
  | "flux2_klein"
  | "flux2_klein_base"
  | "krea2_turbo"
  | "wan22_t2v"
  | "wan22_i2v"
  | "wan22_5b";
export type LoraBase = "sd15" | "sdxl" | "flux" | "flux2" | "krea2" | "wan";
export type Shape = "square" | "portrait" | "wide";
export type Quality = "fast" | "balanced" | "best";

export interface Profile {
  label: string;
  /**
   * How the model is wired. "sd" and "flux1" sample with KSampler; "flux2" uses ComfyUI's
   * FLUX.2 graph (Flux2Scheduler + CFGGuider + SamplerCustomAdvanced) and a Qwen3 encoder.
   * "krea2" samples with KSampler, a Qwen3-VL encoder and the Qwen Image VAE.
   * "wan22" is video: UMT5 encoder, Wan VAE, and (for 14B) a high-noise and a low-noise model.
   */
  arch: "sd" | "flux1" | "flux2" | "krea2" | "wan22";
  /** Architecture its LoRAs are trained on. */
  loraBase: LoraBase;
  sizes: Record<Shape, [width: number, height: number]>;
  steps: Record<Quality, number>;
  cfg: number;
  sampler: string;
  scheduler: string;
  /** 1 = use every CLIP layer; 2 = skip the last one (CLIPSetLastLayer -2). */
  clipSkip: number;
  /** False for distilled models, which ignore the negative prompt unless CFG is raised above 1. */
  negative: boolean;
  /** Flux distilled guidance, applied with FluxGuidance. */
  guidance?: number;
  /** Tags the family expects, added in front of the user's prompts. */
  positivePrefix?: string;
  negativePrefix?: string;
  /** Latent node for text-to-image (or text-to-video). */
  latent: "EmptyLatentImage" | "EmptySD3LatentImage" | "EmptyFlux2LatentImage" | "EmptyHunyuanLatentVideo" | "Wan22ImageToVideoLatent";
  /** Set for video models. */
  video?: VideoProfile;
}

export interface VideoProfile {
  fps: number;
  /** Lengths offered, in seconds; the last is the longest the model handles well. */
  seconds: number[];
  /** ModelSamplingSD3 shift. */
  shift: number;
  /** Wan 2.2 14B: sample the first half with the high-noise model, the rest with the low-noise one. */
  experts: boolean;
  /** "required": animates a start picture; "optional": can start from one; "none": text only. */
  image: "required" | "optional" | "none";
  /** VAE file the model expects, matched by name. */
  vae: RegExp;
  vaeFile: string;
}

/** Wan's own default negative prompt (from ComfyUI's templates), in the language it was trained on. */
export const WAN_NEGATIVE =
  "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

const SDXL_SIZES: Profile["sizes"] = { square: [1024, 1024], portrait: [832, 1216], wide: [1216, 832] };

const WAN14: Profile = {
  arch: "wan22",
  label: "Wan 2.2 14B",
  loraBase: "wan",
  sizes: { square: [640, 640], portrait: [480, 832], wide: [832, 480] },
  steps: { fast: 14, balanced: 20, best: 30 },
  cfg: 3.5,
  sampler: "euler",
  scheduler: "simple",
  clipSkip: 1,
  negative: true,
  negativePrefix: WAN_NEGATIVE,
  latent: "EmptyHunyuanLatentVideo",
  video: { fps: 16, seconds: [2, 3, 5], shift: 5, experts: true, image: "none", vae: /wan.?2\.?1.?vae/, vaeFile: "wan_2.1_vae.safetensors" },
};

export const PROFILES: Record<Family, Profile> = {
  sd15: {
    arch: "sd",
    label: "SD 1.5",
    loraBase: "sd15",
    sizes: { square: [512, 512], portrait: [512, 768], wide: [768, 512] },
    steps: { fast: 15, balanced: 25, best: 35 },
    cfg: 7,
    sampler: "dpmpp_2m",
    scheduler: "karras",
    clipSkip: 1,
    negative: true,
    latent: "EmptyLatentImage",
  },
  sdxl: {
    arch: "sd",
    label: "SDXL",
    loraBase: "sdxl",
    sizes: SDXL_SIZES,
    steps: { fast: 20, balanced: 30, best: 40 },
    cfg: 6,
    sampler: "dpmpp_2m",
    scheduler: "karras",
    clipSkip: 1,
    negative: true,
    latent: "EmptyLatentImage",
  },
  pony: {
    arch: "sd",
    label: "Pony",
    loraBase: "sdxl",
    sizes: SDXL_SIZES,
    steps: { fast: 18, balanced: 25, best: 35 },
    cfg: 7,
    sampler: "euler_ancestral",
    scheduler: "normal",
    clipSkip: 2,
    negative: true,
    positivePrefix: "score_9, score_8_up, score_7_up",
    negativePrefix: "score_6, score_5, score_4",
    latent: "EmptyLatentImage",
  },
  illustrious: {
    arch: "sd",
    label: "Illustrious",
    loraBase: "sdxl",
    sizes: SDXL_SIZES,
    steps: { fast: 20, balanced: 28, best: 36 },
    cfg: 5.5,
    sampler: "euler_ancestral",
    scheduler: "normal",
    clipSkip: 2,
    negative: true,
    positivePrefix: "masterpiece, best quality",
    negativePrefix: "worst quality, low quality",
    latent: "EmptyLatentImage",
  },
  flux_dev: {
    arch: "flux1",
    label: "Flux dev",
    loraBase: "flux",
    sizes: SDXL_SIZES,
    steps: { fast: 12, balanced: 20, best: 28 },
    cfg: 1,
    sampler: "euler",
    scheduler: "simple",
    clipSkip: 1,
    negative: false,
    guidance: 3.5,
    latent: "EmptySD3LatentImage",
  },
  flux_schnell: {
    arch: "flux1",
    label: "Flux schnell",
    loraBase: "flux",
    sizes: SDXL_SIZES,
    steps: { fast: 4, balanced: 4, best: 6 },
    cfg: 1,
    sampler: "euler",
    scheduler: "simple",
    clipSkip: 1,
    negative: false,
    latent: "EmptySD3LatentImage",
  },
  // Defaults follow ComfyUI's own FLUX.2 Klein templates: distilled at 4 steps and CFG 1
  // (no negative prompt), base at 20 steps and CFG 5.
  flux2_klein: {
    arch: "flux2",
    label: "Flux.2 Klein",
    loraBase: "flux2",
    sizes: SDXL_SIZES,
    steps: { fast: 4, balanced: 4, best: 6 },
    cfg: 1,
    sampler: "euler",
    scheduler: "simple",
    clipSkip: 1,
    negative: false,
    latent: "EmptyFlux2LatentImage",
  },
  flux2_klein_base: {
    arch: "flux2",
    label: "Flux.2 Klein base",
    loraBase: "flux2",
    sizes: SDXL_SIZES,
    steps: { fast: 14, balanced: 20, best: 32 },
    cfg: 5,
    sampler: "euler",
    scheduler: "simple",
    clipSkip: 1,
    negative: true,
    latent: "EmptyFlux2LatentImage",
  },
  // Follows ComfyUI's Krea 2 Turbo template: distilled for 8 steps at CFG 1, euler/simple.
  krea2_turbo: {
    arch: "krea2",
    label: "Krea 2 Turbo",
    loraBase: "krea2",
    sizes: SDXL_SIZES,
    steps: { fast: 6, balanced: 8, best: 10 },
    cfg: 1,
    sampler: "euler",
    scheduler: "simple",
    clipSkip: 1,
    negative: false,
    latent: "EmptyLatentImage",
  },
  // Wan 2.2 follows ComfyUI's templates: 14B splits sampling between its two models at the
  // halfway step with shift 5 and CFG 3.5 (the 4-step speed LoRAs, when present, make Fast
  // 4 steps at CFG 1); 5B runs alone at shift 8, CFG 5, uni_pc.
  wan22_t2v: {
    ...WAN14,
    label: "Wan 2.2 14B text to video",
    latent: "EmptyHunyuanLatentVideo",
    video: { ...WAN14.video!, image: "none" },
  },
  wan22_i2v: {
    ...WAN14,
    label: "Wan 2.2 14B image to video",
    latent: "Wan22ImageToVideoLatent",
    video: { ...WAN14.video!, image: "required" },
  },
  wan22_5b: {
    arch: "wan22",
    label: "Wan 2.2 5B",
    loraBase: "wan",
    sizes: { square: [960, 960], portrait: [704, 1280], wide: [1280, 704] },
    steps: { fast: 14, balanced: 20, best: 30 },
    cfg: 5,
    sampler: "uni_pc",
    scheduler: "simple",
    clipSkip: 1,
    negative: true,
    negativePrefix: WAN_NEGATIVE,
    latent: "Wan22ImageToVideoLatent",
    video: { fps: 24, seconds: [2, 3, 5], shift: 8, experts: false, image: "optional", vae: /wan.?2\.?2.?vae/, vaeFile: "wan2.2_vae.safetensors" },
  },
};

/** Families the detector recognises but Fletch can't run yet, in words. */
export const UNSUPPORTED: Record<string, string> = {
  sd2: "SD 2",
  sd3: "SD 3.5",
  flux2: "FLUX.2 dev",
  wan21: "Wan 2.1",
  krea2_raw: "Krea 2 RAW (made for training LoRAs; use Krea 2 Turbo to make images)",
};

export const FAMILIES = Object.keys(PROFILES) as Family[];

export const isFamily = (f: unknown): f is Family => typeof f === "string" && f in PROFILES;
