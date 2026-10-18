// Video workflows (Wan 2.2), built the way ComfyUI's own templates wire them.
//
// Wan 2.2 14B is two models: the high-noise one lays out motion and composition for the first
// half of the steps, then the low-noise one finishes the detail. Each gets its own LoRAs and
// ModelSamplingSD3 shift, and two KSamplerAdvanced nodes hand the latent from one to the other.

import { PROFILES } from "./profiles";
import {
  Graph,
  isAnimated,
  loadImage,
  nodeDefaults,
  resolve,
  withPrefix,
  type ApiWorkflow,
  type Destination,
  type InputImage,
  type Link,
  type NodeInfos,
  type Parts,
  type Settings,
} from "./workflow";

/** Video models take frame counts in steps of 4 (Wan) or 8 (LTX), plus one. */
export const frameStep = (family: Settings["family"]) => (PROFILES[family].arch === "ltx23" ? 8 : 4);

/** How many of a GIF's frames to restyle: up to the chosen length, rounded down to a count the model takes. */
export function gifFrames(gif: InputImage, seconds: number, step: number): number {
  const want = Math.min(gif.frames ?? 1, Math.max(1, Math.round(seconds * (gif.fps ?? 10))));
  return Math.floor((want - 1) / step) * step + 1;
}

/** The GIF's frames, trimmed and scaled to the size being made, and encoded as a video latent. */
function gifLatent(g: Graph, gif: InputImage, frames: number, width: number, height: number, vae: Link): { pictures: Link; latent: Link } {
  const picked = g.add("ImageFromBatch", { image: [loadImage(g, gif), 0], batch_index: 0, length: frames });
  const pictures: Link = [g.add("ImageScale", { image: [picked, 0], upscale_method: "lanczos", width, height, crop: "center" }), 0];
  return { pictures, latent: [g.add("VAEEncode", { pixels: pictures, vae }), 0] };
}

function saveGif(g: Graph, images: Link, fps: number, dest: Destination) {
  g.add("FletchSaveGif", {
    images,
    fps,
    folder: dest.folder ?? "",
    filename_prefix: !dest.folder && dest.subfolder ? `${dest.subfolder}/fletch` : "fletch",
  });
}

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
  const gif = isAnimated(start) ? start! : null;
  const size = { width: r.width, height: r.height, length: r.frames, batch_size: 1 };
  let LATENT: Link;
  if (gif) {
    LATENT = gifLatent(g, gif, gifFrames(gif, r.seconds, 4), r.width, r.height, VAE).latent;
  } else if (p.latent === "EmptyHunyuanLatentVideo") {
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
  // Restyling a GIF skips the first part of the schedule, like img2img: the more change, the earlier it starts.
  const denoise = gif ? s.addons!.source!.strength : 1;
  const skip = Math.min(r.steps - 1, Math.round(r.steps * (1 - denoise)));
  let SAMPLES: Link;
  if (v.experts && skip >= Math.floor(r.steps / 2)) {
    // Little change asked for: only the low-noise model's half of the schedule is left.
    SAMPLES = [
      g.add("KSamplerAdvanced", {
        model: model(parts.low, "low"),
        add_noise: "enable",
        noise_seed: s.seed,
        steps: r.steps,
        ...common,
        latent_image: LATENT,
        start_at_step: skip,
        end_at_step: 10000,
        return_with_leftover_noise: "disable",
      }),
      0,
    ];
  } else if (v.experts) {
    const split = Math.max(1, Math.floor(r.steps / 2));
    const first = g.add("KSamplerAdvanced", {
      model: model(parts.high, "high"),
      add_noise: "enable",
      noise_seed: s.seed,
      steps: r.steps,
      ...common,
      latent_image: LATENT,
      start_at_step: skip,
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
      g.add("KSampler", { model: model(s.model.name, null), seed: s.seed, steps: r.steps, ...common, latent_image: LATENT, denoise }),
      0,
    ];
  }

  const frames: Link = [g.add("VAEDecode", { samples: SAMPLES, vae: VAE }), 0];
  if (gif && s.addons?.source?.output !== "video") {
    saveGif(g, frames, gif.fps!, dest);
    return g.nodes;
  }
  save(g, [g.add("CreateVideo", { ...nodeDefaults(nodes.CreateVideo), images: frames, fps: gif ? gif.fps! : v.fps }), 0], nodes, dest);
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

/** The distilled schedule from `change` down: start partway, keep the template's later steps. */
export function ltxRestyleSigmas(change: number): string {
  const rest = LTX_SIGMAS_FIRST.split(", ").map(Number).filter((x) => x < change);
  return [Math.min(1, change), ...rest].map((x) => (Number.isInteger(x) ? x.toFixed(1) : String(x))).join(", ");
}

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
  const start = s.addons?.source?.image;
  if (isAnimated(start)) return ltxRestyle(g, s, start!, { MODEL, VAE, AUDIO_VAE, CLIP }, nodes, dest);

  const cond = g.add("LTXVConditioning", {
    positive: [g.add("CLIPTextEncode", { clip: CLIP, text: withPrefix(p.positivePrefix, s.positive) }), 0],
    negative: [g.add("CLIPTextEncode", { clip: CLIP, text: withPrefix(p.negativePrefix, s.negative) }), 0],
    frame_rate: v.fps,
  });
  const [POSITIVE, NEGATIVE]: Link[] = [[cond, 0], [cond, 1]];

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

/**
 * LTX-2.3 video to video: the GIF's frames are encoded at full size and sampled once from
 * partway down the distilled schedule, so the motion stays and the look follows the prompt.
 * To add sound only, the video is masked off so it stays as it is and only the sound is made.
 */
function ltxRestyle(
  g: Graph,
  s: Settings,
  gif: InputImage,
  m: { MODEL: Link; VAE: Link; AUDIO_VAE: Link; CLIP: Link },
  nodes: NodeInfos,
  dest: Destination,
): ApiWorkflow {
  const output = s.addons?.source?.output ?? "gif";
  const p = PROFILES[s.family];
  const r = resolve(s);
  const fps = gif.fps!;
  const frames = gifFrames(gif, r.seconds, 8);
  const cond = g.add("LTXVConditioning", {
    positive: [g.add("CLIPTextEncode", { clip: m.CLIP, text: withPrefix(p.positivePrefix, s.positive) }), 0],
    negative: [g.add("CLIPTextEncode", { clip: m.CLIP, text: withPrefix(p.negativePrefix, s.negative) }), 0],
    frame_rate: fps,
  });
  const audio: Link = [g.add("LTXVEmptyLatentAudio", { audio_vae: m.AUDIO_VAE, frames_number: frames, frame_rate: fps, batch_size: 1 }), 0];
  const source = gifLatent(g, gif, frames, r.width, r.height, m.VAE);
  const keep = output === "sound";
  const video: Link = keep
    ? [g.add("SetLatentNoiseMask", { samples: source.latent, mask: [g.add("SolidMask", { value: 0, width: r.width, height: r.height }), 0] }), 0]
    : source.latent;
  const latent: Link = [g.add("LTXVConcatAVLatent", { video_latent: video, audio_latent: audio }), 0];
  const sampled = g.add("SamplerCustomAdvanced", {
    noise: [g.add("RandomNoise", { noise_seed: s.seed }), 0],
    guider: [g.add("CFGGuider", { model: m.MODEL, positive: [cond, 0], negative: [cond, 1], cfg: r.cfg }), 0],
    sampler: [g.add("KSamplerSelect", { sampler_name: r.sampler }), 0],
    sigmas: [g.add("ManualSigmas", { sigmas: keep ? LTX_SIGMAS_FIRST : ltxRestyleSigmas(s.addons!.source!.strength) }), 0],
    latent_image: latent,
  });
  const split = g.add("LTXVSeparateAVLatent", { av_latent: [sampled, 0] });
  // Sound only: the GIF's own frames go straight into the video, untouched by the VAE.
  const images: Link = keep
    ? source.pictures
    : [g.add("VAEDecodeTiled", { samples: [split, 0], vae: m.VAE, tile_size: 768, overlap: 64, temporal_size: 4096, temporal_overlap: 4 }), 0];
  if (output === "gif") {
    saveGif(g, images, fps, dest);
    return g.nodes;
  }
  const sound: Link = [g.add("LTXVAudioVAEDecode", { samples: [split, 1], audio_vae: m.AUDIO_VAE }), 0];
  save(g, [g.add("CreateVideo", { ...nodeDefaults(nodes.CreateVideo), images, audio: sound, fps }), 0], nodes, dest);
  return g.nodes;
}
