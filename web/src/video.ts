// Video workflows (Wan 2.2), built the way ComfyUI's own templates wire them.
//
// Wan 2.2 14B is two models: the high-noise one lays out motion and composition for the first
// half of the steps, then the low-noise one finishes the detail. Each gets its own LoRAs and
// ModelSamplingSD3 shift, and two KSamplerAdvanced nodes hand the latent from one to the other.

import { PROFILES } from "./profiles";
import { Graph, loadImage, resolve, withPrefix, type ApiWorkflow, type Destination, type Link, type NodeInfos, type Parts, type Settings } from "./workflow";

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
  const video: Link = [g.add("CreateVideo", { images: frames, fps: v.fps }), 0];
  if (dest.folder) g.add("FletchSaveVideo", { video, folder: dest.folder, filename_prefix: "fletch" });
  else
    g.add("SaveVideo", {
      video,
      filename_prefix: dest.subfolder ? `${dest.subfolder}/fletch` : "fletch",
      ...saveVideoInputs(nodes),
    });
  return g.nodes;
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
