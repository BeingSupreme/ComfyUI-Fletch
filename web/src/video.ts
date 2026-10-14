// Video workflows (Wan 2.2), built the way ComfyUI's own templates wire them.
//
// Wan 2.2 14B is two models: the high-noise one lays out motion and composition for the first
// half of the steps, then the low-noise one finishes the detail. Each gets its own LoRAs and
// ModelSamplingSD3 shift, and two KSamplerAdvanced nodes hand the latent from one to the other.

import { PROFILES } from "./profiles";
import { Graph, loadImage, nodeDefaults, resolve, withPrefix, type ApiWorkflow, type Destination, type Link, type NodeInfos, type Parts, type Settings } from "./workflow";

/** Which half of a Wan 2.2 14B pair a LoRA belongs to, from its name; "both" if it doesn't say. */
export function loraTarget(name: string): "high" | "low" | "both" {
  const base = (name.split(/[\\/]/).pop() ?? name).toLowerCase();
  if (/high.?noise|_high[_.]|\bhn\b/.test(base)) return "high";
  if (/low.?noise|_low[_.]|\bln\b/.test(base)) return "low";
  return "both";
}

/**
 * SaveVideo's codec moved into a nested input (format.codec) in newer ComfyUI; older versions
 * take format and codec side by side. Ask the node which it is.
 */
function saveVideoInputs(nodes: NodeInfos): Record<string, unknown> {
  const required = nodes.SaveVideo?.input?.required ?? {};
  return "codec" in required ? { format: "auto", codec: "auto" } : { format: "auto", "format.codec": "auto" };
}

export function compileVideo(s: Settings, parts: Parts, nodes: NodeInfos = {}, dest: Destination = {}): ApiWorkflow {
  if (PROFILES[s.family].arch === "ltx23") return compileLtx(s, parts, nodes, dest);
  const p = PROFILES[s.family];
  const v = p.video!;
  const r = resolve(s);
  const g = new Graph();

  const CLIP: Link = [g.add("CLIPLoader", { clip_name: parts.umt5, type: "wan", device: "default" }), 0];
  const VAE: Link = [g.add("VAELoader", { vae_name: parts.vae }), 0];
  let POSITIVE: Link = [g.add("CLIPTextEncode", { clip: CLIP, text: withPrefix(p.positivePrefix, s.positive) }), 0];
  let NEGATIVE: Link = [g.add("CLIPTextEncode", { clip: CLIP, text: withPrefix(p.negativePrefix, s.negative) }), 0];

  const start = s.addons?.source?.image;
  const size = { width: r.width, height: r.height, length: r.frames, batch_size: 1 };
  let LATENT: Link;
  if (p.latent === "EmptyHunyuanLatentVideo") {
    LATENT = [g.add("EmptyHunyuanLatentVideo", size), 0];
  } else if (v.experts) {
    // 14B image to video: the start picture conditions both prompts and seeds the latent.
    const i2v = g.add("WanImageToVideo", { positive: POSITIVE, negative: NEGATIVE, vae: VAE, start_image: [loadImage(g, start!), 0], ...size });
    [POSITIVE, NEGATIVE, LATENT] = [[i2v, 0], [i2v, 1], [i2v, 2]];
  } else {
    LATENT = [
      g.add("Wan22ImageToVideoLatent", { vae: VAE, ...size, ...(start ? { start_image: [loadImage(g, start), 0] } : {}) }),
      0,
    ];
  }

  // Load a model, apply the LoRAs meant for it, then the shift every Wan template sets.
  const model = (name: string | undefined, half: "high" | "low" | null): Link => {
    let m: Link = [g.add("UNETLoader", { unet_name: name, weight_dtype: "default" }), 0];
    const loras = [
      ...(s.speedLora && half ? [{ name: s.speedLora[half], weight: 1 }] : []),
      ...s.loras.filter((l) => l.name && (!half || loraTarget(l.name) === "both" || loraTarget(l.name) === half)),
    ];
    for (const l of loras) m = [g.add("LoraLoaderModelOnly", { model: m, lora_name: l.name, strength_model: l.weight }), 0];
    return [g.add("ModelSamplingSD3", { model: m, shift: v.shift }), 0];
  };

  const common = { cfg: r.cfg, sampler_name: r.sampler, scheduler: r.scheduler, positive: POSITIVE, negative: NEGATIVE };
  let SAMPLES: Link;
  if (v.experts) {
    const split = Math.max(1, Math.floor(r.steps / 2));
    const first = g.add("KSamplerAdvanced", {
      model: model(parts.high, "high"),
      add_noise: "enable",
      noise_seed: s.seed,
      steps: r.steps,
      ...common,
      latent_image: LATENT,
      start_at_step: 0,
      end_at_step: split,
      return_with_leftover_noise: "enable",
    });
    SAMPLES = [
      g.add("KSamplerAdvanced", {
        model: model(parts.low, "low"),
        add_noise: "disable",
        noise_seed: 0,
        steps: r.steps,
        ...common,
        latent_image: [first, 0],
        start_at_step: split,
        end_at_step: 10000,
        return_with_leftover_noise: "disable",
      }),
      0,
    ];
  } else {
    SAMPLES = [
      g.add("KSampler", { model: model(s.model.name, null), seed: s.seed, steps: r.steps, ...common, latent_image: LATENT, denoise: 1 }),
      0,
    ];
  }

  const frames: Link = [g.add("VAEDecode", { samples: SAMPLES, vae: VAE }), 0];
  save(g, [g.add("CreateVideo", { ...nodeDefaults(nodes.CreateVideo), images: frames, fps: v.fps }), 0], nodes, dest);
  return g.nodes;
}

/** Newer CreateVideo has extra required choices (bit depth, colour space); their defaults come from the node. */
function save(g: Graph, video: Link, nodes: NodeInfos, dest: Destination) {
  if (dest.folder) g.add("FletchSaveVideo", { video, folder: dest.folder, filename_prefix: "fletch" });
  else g.add("SaveVideo", { video, filename_prefix: dest.subfolder ? `${dest.subfolder}/fletch` : "fletch", ...saveVideoInputs(nodes) });
}

/** The lightx2v / Lightning 4-step LoRA pair for a Wan 2.2 14B model, if both halves are installed. */
export function findSpeedLoras(family: string, loraNames: string[]): { high: string; low: string } | undefined {
  const kind = family === "wan22_i2v" ? "i2v" : "t2v";
  const other = kind === "i2v" ? "t2v" : "i2v";
  const speed = loraNames.filter((n) => {
    const b = n.toLowerCase();
    return /lightx2v|lightning|4.?steps?/.test(b) && !b.includes(other);
  });
  const pick = (half: "high" | "low") =>
    speed.find((n) => loraTarget(n) === half && n.toLowerCase().includes(kind)) ?? speed.find((n) => loraTarget(n) === half);
  const [high, low] = [pick("high"), pick("low")];
  return high && low ? { high, low } : undefined;
}

// LTX-2.3's distilled schedule from ComfyUI's template: 8 steps at half size, then 3 after the 2x upscale.
const LTX_SIGMAS_FIRST = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0";
const LTX_SIGMAS_REFINE = "0.85, 0.7250, 0.4219, 0.0";

/**
 * LTX-2.3 makes video and sound together. Pass one samples a half-size video latent joined with
 * an audio latent; the video half is upscaled 2x and the pair is refined in a second pass. A start
 * picture is pressed into the latent before each pass (LTXVImgToVideoInplace).
 */
export function compileLtx(s: Settings, parts: Parts, nodes: NodeInfos = {}, dest: Destination = {}): ApiWorkflow {
  const p = PROFILES[s.family];
  const v = p.video!;
  const r = resolve(s);
  const g = new Graph();
  const ckpt = s.model.name;

  const loader = g.add("CheckpointLoaderSimple", { ckpt_name: ckpt });
  const VAE: Link = [loader, 2];
  let MODEL: Link = [g.add("LoraLoaderModelOnly", { model: [loader, 0], lora_name: parts.distillLora, strength_model: 0.5 }), 0];
  for (const l of s.loras) if (l.name) MODEL = [g.add("LoraLoaderModelOnly", { model: MODEL, lora_name: l.name, strength_model: l.weight }), 0];
  const AUDIO_VAE: Link = [g.add("LTXVAudioVAELoader", { ckpt_name: ckpt }), 0];
  const CLIP: Link = [g.add("LTXAVTextEncoderLoader", { text_encoder: parts.gemma, ckpt_name: ckpt, device: "default" }), 0];

  const cond = g.add("LTXVConditioning", {
    positive: [g.add("CLIPTextEncode", { clip: CLIP, text: withPrefix(p.positivePrefix, s.positive) }), 0],
    negative: [g.add("CLIPTextEncode", { clip: CLIP, text: withPrefix(p.negativePrefix, s.negative) }), 0],
    frame_rate: v.fps,
  });
  const [POSITIVE, NEGATIVE]: Link[] = [[cond, 0], [cond, 1]];

  const start = s.addons?.source?.image;
  let picture: Link | null = null;
  if (start) {
    const scaled = g.add("ImageScale", { image: [loadImage(g, start), 0], upscale_method: "lanczos", width: r.width, height: r.height, crop: "center" });
    picture = [g.add("LTXVPreprocess", { image: [scaled, 0], img_compression: 18 }), 0];
  }
  const withPicture = (latent: Link, strength: number): Link =>
    picture ? [g.add("LTXVImgToVideoInplace", { vae: VAE, image: picture, latent, strength, bypass: false }), 0] : latent;

  const sampleAv = (latent: Link, positive: Link, negative: Link, sigmas: string, seed: number): Link => [
    g.add("SamplerCustomAdvanced", {
      noise: [g.add("RandomNoise", { noise_seed: seed }), 0],
      guider: [g.add("CFGGuider", { model: MODEL, positive, negative, cfg: r.cfg }), 0],
      sampler: [g.add("KSamplerSelect", { sampler_name: r.sampler }), 0],
      sigmas: [g.add("ManualSigmas", { sigmas }), 0],
      latent_image: latent,
    }),
    0,
  ];

  // Pass 1: half size, video and sound together.
  const small: Link = [g.add("EmptyLTXVLatentVideo", { width: Math.round(r.width / 2), height: Math.round(r.height / 2), length: r.frames, batch_size: 1 }), 0];
  const audio: Link = [g.add("LTXVEmptyLatentAudio", { audio_vae: AUDIO_VAE, frames_number: r.frames, frame_rate: v.fps, batch_size: 1 }), 0];
  const first = sampleAv(
    [g.add("LTXVConcatAVLatent", { video_latent: withPicture(small, 0.7), audio_latent: audio }), 0],
    POSITIVE,
    NEGATIVE,
    LTX_SIGMAS_FIRST,
    s.seed,
  );
  const split1 = g.add("LTXVSeparateAVLatent", { av_latent: first });

  // Pass 2: upscale the video latent 2x and refine both.
  const upscaled: Link = [
    g.add("LTXVLatentUpsampler", {
      samples: [split1, 0],
      upscale_model: [g.add("LatentUpscaleModelLoader", { model_name: parts.upscaler }), 0],
      vae: VAE,
    }),
    0,
  ];
  const crop = g.add("LTXVCropGuides", { positive: POSITIVE, negative: NEGATIVE, latent: [split1, 0] });
  const second = sampleAv(
    [g.add("LTXVConcatAVLatent", { video_latent: withPicture(upscaled, 1), audio_latent: [split1, 1] }), 0],
    [crop, 0],
    [crop, 1],
    LTX_SIGMAS_REFINE,
    s.seed + 1,
  );
  const split2 = g.add("LTXVSeparateAVLatent", { av_latent: second });

  const frames: Link = [
    g.add("VAEDecodeTiled", { samples: [split2, 0], vae: VAE, tile_size: 768, overlap: 64, temporal_size: 4096, temporal_overlap: 4 }),
    0,
  ];
  const sound: Link = [g.add("LTXVAudioVAEDecode", { samples: [split2, 1], audio_vae: AUDIO_VAE }), 0];
  save(g, [g.add("CreateVideo", { ...nodeDefaults(nodes.CreateVideo), images: frames, audio: sound, fps: v.fps }), 0], nodes, dest);
  return g.nodes;
}
