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
  | "flux2_klein_base";
export type LoraBase = "sd15" | "sdxl" | "flux" | "flux2";
export type Shape = "square" | "portrait" | "wide";
export type Quality = "fast" | "balanced" | "best";

export interface Profile {
  label: string;
  /**
   * How the model is wired. "sd" and "flux1" sample with KSampler; "flux2" uses ComfyUI's
   * FLUX.2 graph (Flux2Scheduler + CFGGuider + SamplerCustomAdvanced) and a Qwen3 encoder.
   */
  arch: "sd" | "flux1" | "flux2";
  /** Architecture its LoRAs are trained on. */
  loraBase: LoraBase;
  sizes: Record<Shape, [width: number, height: number]>;
  steps: Record<Quality, number>;
  cfg: number;
  sampler: string;
  scheduler: string;
  /** 1 = use every CLIP layer; 2 = skip the last one (CLIPSetLastLayer -2). */
  clipSkip: number;
  /** False hides the "Avoid" box (Flux ignores negative prompts at CFG 1). */
  negative: boolean;
  /** Flux distilled guidance, applied with FluxGuidance. */
  guidance?: number;
  /** Tags the family expects, added in front of the user's prompts. */
  positivePrefix?: string;
  negativePrefix?: string;
  /** Latent node for text-to-image. */
  latent: "EmptyLatentImage" | "EmptySD3LatentImage" | "EmptyFlux2LatentImage";
}

const SDXL_SIZES: Profile["sizes"] = { square: [1024, 1024], portrait: [832, 1216], wide: [1216, 832] };

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
};

/** Families the detector recognises but Fletch can't run yet, in words. */
export const UNSUPPORTED: Record<string, string> = { sd2: "SD 2", sd3: "SD 3.5", flux2: "FLUX.2 dev" };

export const FAMILIES = Object.keys(PROFILES) as Family[];

export const isFamily = (f: unknown): f is Family => typeof f === "string" && f in PROFILES;
