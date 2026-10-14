// Turns the app's settings into a ComfyUI API-format workflow (the JSON POST /prompt accepts).
// Phase 0 builds one fixed text-to-image graph for a full SDXL checkpoint.

export type Link = [nodeId: string, outputIndex: number];

export interface ApiNode {
  class_type: string;
  inputs: Record<string, unknown>;
}

export type ApiWorkflow = Record<string, ApiNode>;

export interface Txt2ImgSettings {
  checkpoint: string;
  positive: string;
  negative: string;
  width: number;
  height: number;
  seed: number;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
}

export const SDXL_DEFAULTS = {
  width: 1024,
  height: 1024,
  steps: 30,
  cfg: 6,
  sampler: "dpmpp_2m",
  scheduler: "karras",
} satisfies Partial<Txt2ImgSettings>;

export const randomSeed = () => Math.floor(Math.random() * 2 ** 48);

export function buildTxt2Img(s: Txt2ImgSettings): ApiWorkflow {
  const model: Link = ["1", 0];
  const clip: Link = ["1", 1];
  const vae: Link = ["1", 2];
  return {
    "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: s.checkpoint } },
    "2": { class_type: "CLIPTextEncode", inputs: { clip, text: s.positive } },
    "3": { class_type: "CLIPTextEncode", inputs: { clip, text: s.negative } },
    "4": {
      class_type: "EmptyLatentImage",
      inputs: { width: s.width, height: s.height, batch_size: 1 },
    },
    "5": {
      class_type: "KSampler",
      inputs: {
        model,
        positive: ["2", 0],
        negative: ["3", 0],
        latent_image: ["4", 0],
        seed: s.seed,
        steps: s.steps,
        cfg: s.cfg,
        sampler_name: s.sampler,
        scheduler: s.scheduler,
        denoise: 1,
      },
    },
    "6": { class_type: "VAEDecode", inputs: { samples: ["5", 0], vae } },
    "7": { class_type: "SaveImage", inputs: { images: ["6", 0], filename_prefix: "fletch" } },
  };
}
