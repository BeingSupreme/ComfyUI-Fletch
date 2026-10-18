import { describe, expect, it } from "vitest";
import { WAN_NEGATIVE } from "./profiles";
import { gifFrames, loraTarget, ltxRestyleSigmas } from "./video";
import { base, byType, expectWellFormed, follow, one } from "./testutil";
import { compile, partnerName, pickParts, resolve, runProblems, type Settings } from "./workflow";

const HIGH = "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors";
const LOW = "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors";
const t2v: Settings = { ...base, model: { folder: "diffusion_models", name: HIGH }, family: "wan22_t2v", negative: "" };
const parts = { umt5: "umt5_xxl_fp8_e4m3fn_scaled.safetensors", vae: "wan_2.1_vae.safetensors", high: HIGH, low: LOW };
const start = { name: "fletch/cat.png", width: 1000, height: 1000 };

describe("Wan 2.2 14B", () => {
  it("builds text to video the way ComfyUI's template does", () => {
    const wf = compile(t2v, parts);
    expectWellFormed(wf);
    expect(one(wf, "CLIPLoader")[1].inputs).toEqual({ clip_name: parts.umt5, type: "wan", device: "default" });
    expect(one(wf, "EmptyHunyuanLatentVideo")[1].inputs).toEqual({ width: 640, height: 640, length: 81, batch_size: 1 });
    const [first, second] = byType(wf, "KSamplerAdvanced").map(([, n]) => n.inputs);
    expect(first).toMatchObject({ add_noise: "enable", noise_seed: 42, steps: 20, cfg: 3.5, start_at_step: 0, end_at_step: 10, return_with_leftover_noise: "enable" });
    expect(second).toMatchObject({ add_noise: "disable", start_at_step: 10, end_at_step: 10000, return_with_leftover_noise: "disable" });
    expect(second.latent_image).toEqual([byType(wf, "KSamplerAdvanced")[0][0], 0]);
    expect(follow(wf, follow(wf, first.model).inputs.model).inputs.unet_name).toBe(HIGH);
    expect(follow(wf, first.model).inputs.shift).toBe(5);
    expect(follow(wf, follow(wf, second.model).inputs.model).inputs.unet_name).toBe(LOW);
    expect(follow(wf, first.negative).inputs.text).toBe(WAN_NEGATIVE);
    expect(one(wf, "CreateVideo")[1].inputs.fps).toBe(16);
    expect(one(wf, "SaveVideo")[1].inputs).toMatchObject({ filename_prefix: "fletch", format: "auto", "format.codec": "auto" });
  });

  it("uses the 4-step speed LoRAs for Fast, each on its own half, and other LoRAs by name", () => {
    const speedLora = { high: "lightx2v_high_noise.safetensors", low: "lightx2v_low_noise.safetensors" };
    const loras = [
      { name: "style_high_noise.safetensors", weight: 0.7 },
      { name: "style_both.safetensors", weight: 0.5 },
    ];
    const wf = compile({ ...t2v, quality: "fast", speedLora, loras }, parts);
    expectWellFormed(wf);
    const [first, second] = byType(wf, "KSamplerAdvanced").map(([, n]) => n.inputs);
    expect(first).toMatchObject({ steps: 4, cfg: 1, end_at_step: 2 });
    const chain = (link: unknown): string[] => {
      const n = follow(wf, link);
      if (n.class_type === "UNETLoader") return [];
      return [...chain(n.inputs.model), ...(n.class_type === "LoraLoaderModelOnly" ? [n.inputs.lora_name as string] : [])];
    };
    expect(chain(first.model)).toEqual([speedLora.high, "style_high_noise.safetensors", "style_both.safetensors"]);
    expect(chain(second.model)).toEqual([speedLora.low, "style_both.safetensors"]);
  });

  it("animates a start picture with the image-to-video pair", () => {
    const i2v: Settings = { ...t2v, family: "wan22_i2v", addons: { source: { image: start, strength: 0.6 } }, seconds: 3 };
    const wf = compile(i2v, parts);
    expectWellFormed(wf);
    const [id, node] = one(wf, "WanImageToVideo");
    expect(node.inputs).toMatchObject({ width: 640, height: 640, length: 49 });
    expect(follow(wf, node.inputs.start_image).inputs.image).toBe("fletch/cat.png");
    const first = byType(wf, "KSamplerAdvanced")[0][1].inputs;
    expect([first.positive, first.negative, first.latent_image]).toEqual([[id, 0], [id, 1], [id, 2]]);
  });

  it("says what's missing: the start picture, the partner model, encoder and VAE", () => {
    expect(runProblems({ ...t2v, family: "wan22_i2v" }, parts)).toEqual([
      "This model turns a picture into a video. Turn on Start from an image and add one.",
    ]);
    expect(runProblems({ ...t2v, addons: { source: { image: start, strength: 1 } } }, parts)[0]).toMatch(/restyles an animated GIF/);
    expect(runProblems(t2v, { high: HIGH })).toEqual([
      "Add the UMT5 text encoder (umt5_xxl_fp8_e4m3fn_scaled.safetensors) to models/text_encoders.",
      "Add the Wan VAE (wan_2.1_vae.safetensors) to models/vae.",
      `Wan 2.2 14B comes as two models, high noise and low noise. Put the other one (${LOW}) in models/diffusion_models too.`,
    ]);
  });

  it("finds the partner whichever half is picked", () => {
    const enc = ["umt5_xxl_fp16.safetensors", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"];
    const vaes = ["wan2.2_vae.safetensors", "wan_2.1_vae.safetensors"];
    expect(pickParts("wan22_t2v", null, enc, vaes, HIGH, [HIGH, LOW])).toEqual(parts);
    expect(pickParts("wan22_t2v", null, enc, vaes, LOW, [HIGH, LOW])).toEqual(parts);
    expect(pickParts("wan22_t2v", null, enc, vaes, HIGH, [HIGH]).low).toBeUndefined();
    expect(partnerName("video/Wan2.2-I2V-HighNoise-Q8.gguf", "high")).toBe("video/Wan2.2-I2V-LowNoise-Q8.gguf");
    expect(pickParts("wan22_5b", null, enc, vaes).vae).toBe("wan2.2_vae.safetensors");
  });

  it("reads which half a LoRA is for from its name", () => {
    expect(loraTarget("wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")).toBe("high");
    expect(loraTarget("loras/Wan22_LowNoise_style.safetensors")).toBe("low");
    expect(loraTarget("cinematic.safetensors")).toBe("both");
  });
});

describe("Wan 2.2 5B", () => {
  const five: Settings = { ...t2v, model: { folder: "diffusion_models", name: "wan2.2_ti2v_5B_fp16.safetensors" }, family: "wan22_5b", shape: "wide" };
  const fiveParts = { umt5: parts.umt5, vae: "wan2.2_vae.safetensors" };

  it("runs one model with KSampler at 24 fps, with or without a start picture", () => {
    const wf = compile(five, fiveParts);
    expectWellFormed(wf);
    expect(one(wf, "Wan22ImageToVideoLatent")[1].inputs).toMatchObject({ width: 1280, height: 704, length: 121 });
    expect(one(wf, "KSampler")[1].inputs).toMatchObject({ steps: 20, cfg: 5, sampler_name: "uni_pc", scheduler: "simple" });
    expect(follow(wf, one(wf, "KSampler")[1].inputs.model).inputs.shift).toBe(8);
    expect(one(wf, "CreateVideo")[1].inputs.fps).toBe(24);
    const withImage = compile({ ...five, addons: { source: { image: start, strength: 0.6 } } }, fiveParts);
    expect(one(withImage, "Wan22ImageToVideoLatent")[1].inputs.start_image).toBeDefined();
  });

  it("saves outside ComfyUI's output with Fletch's video node, and speaks old SaveVideo", () => {
    expect(one(compile(five, fiveParts, undefined, {}, { folder: "D:/Clips" }), "FletchSaveVideo")[1].inputs.folder).toBe("D:/Clips");
    const old = { SaveVideo: { input: { required: { video: ["VIDEO"], filename_prefix: ["STRING"], format: [["auto", "mp4"]], codec: [["auto", "h264"]] } } } };
    expect(one(compile(five, fiveParts, undefined, old), "SaveVideo")[1].inputs).toMatchObject({ format: "auto", codec: "auto" });
  });

  it("offers 2, 3 or 5 seconds", () => {
    expect(resolve({ ...five, seconds: 2 }).frames).toBe(49);
    expect(resolve({ ...five, seconds: 7 }).frames).toBe(121);
  });
});

describe("speed LoRAs", () => {
  it("finds the matching 4-step pair", async () => {
    const { findSpeedLoras } = await import("./video");
    const names = [
      "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
      "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
      "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
      "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
      "detail.safetensors",
    ];
    expect(findSpeedLoras("wan22_t2v", names)).toEqual({ high: names[0], low: names[1] });
    expect(findSpeedLoras("wan22_i2v", names)).toEqual({ high: names[2], low: names[3] });
    expect(findSpeedLoras("wan22_i2v", names.slice(0, 3))).toBeUndefined();
  });
});

describe("LTX-2.3", () => {
  const CKPT = "ltx-2.3-22b-dev-fp8.safetensors";
  const ltx: Settings = { ...base, model: { folder: "checkpoints", name: CKPT }, family: "ltx23", shape: "wide", negative: "" };
  const ltxParts = {
    gemma: "gemma_3_12B_it_fp4_mixed.safetensors",
    distillLora: "ltx-2.3-22b-distilled-lora-384.safetensors",
    upscaler: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
  };

  it("builds the two-pass video and sound graph from ComfyUI's template", () => {
    const wf = compile({ ...ltx, loras: [{ name: "style.safetensors", weight: 0.8 }] }, ltxParts);
    expectWellFormed(wf);
    expect(one(wf, "EmptyLTXVLatentVideo")[1].inputs).toEqual({ width: 640, height: 360, length: 126, batch_size: 1 });
    expect(one(wf, "LTXVEmptyLatentAudio")[1].inputs).toMatchObject({ frames_number: 126, frame_rate: 25 });
    expect(one(wf, "LTXAVTextEncoderLoader")[1].inputs).toEqual({ text_encoder: ltxParts.gemma, ckpt_name: CKPT, device: "default" });
    expect(byType(wf, "ManualSigmas").map(([, n]) => n.inputs.sigmas)).toEqual([
      "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
      "0.85, 0.7250, 0.4219, 0.0",
    ]);
    const guiders = byType(wf, "CFGGuider").map(([, n]) => n.inputs);
    expect(guiders.map((g) => g.cfg)).toEqual([1, 1]);
    expect(follow(wf, guiders[1].positive).class_type).toBe("LTXVCropGuides");
    const lora = follow(wf, guiders[0].model);
    expect(lora.inputs.lora_name).toBe("style.safetensors");
    expect(follow(wf, lora.inputs.model).inputs).toMatchObject({ lora_name: ltxParts.distillLora, strength_model: 0.5 });
    expect(one(wf, "LatentUpscaleModelLoader")[1].inputs.model_name).toBe(ltxParts.upscaler);
    expect(byType(wf, "RandomNoise").map(([, n]) => n.inputs.noise_seed)).toEqual([42, 43]);
    const video = one(wf, "CreateVideo")[1].inputs;
    expect(follow(wf, video.audio).class_type).toBe("LTXVAudioVAEDecode");
    expect(follow(wf, video.images).class_type).toBe("VAEDecodeTiled");
    expect(video.fps).toBe(25);
    expect(byType(wf, "LTXVImgToVideoInplace")).toHaveLength(0);
  });

  it("presses a start picture into both passes", () => {
    const wf = compile({ ...ltx, addons: { source: { image: start, strength: 1 } } }, ltxParts);
    expectWellFormed(wf);
    expect(byType(wf, "LTXVImgToVideoInplace").map(([, n]) => n.inputs.strength)).toEqual([0.7, 1]);
    expect(one(wf, "ImageScale")[1].inputs).toMatchObject({ width: 960, height: 960 });
  });

  it("goes up to 20 seconds", () => {
    expect(resolve({ ...ltx, seconds: 10 }).frames).toBe(251);
    expect(resolve({ ...ltx, seconds: 20 }).frames).toBe(501);
  });

  it("finds its files and says which are missing", () => {
    const parts = pickParts("ltx23", null, ["gemma_3_12B_it_fp4_mixed.safetensors", "t5xxl.safetensors"], [], CKPT, [], {
      loras: ["detail.safetensors", ltxParts.distillLora],
      latentUpscalers: ["ltx-2-spatial-upscaler-x2-1.0.safetensors", ltxParts.upscaler],
    });
    expect(parts).toEqual(ltxParts);
    const problems = runProblems({ ...ltx, model: { folder: "diffusion_models", name: CKPT } }, {});
    expect(problems).toHaveLength(4);
    expect(problems[0]).toMatch(/models\/checkpoints/);
    expect(runProblems(ltx, ltxParts)).toEqual([]);
  });
});

describe("GIF to GIF", () => {
  const gif = { name: "fletch/dance.gif", width: 480, height: 270, frames: 40, fps: 10 };
  const restyle = (family: Settings["family"], model: Settings["model"], strength = 0.6): Settings => ({
    ...t2v, family, model, addons: { source: { image: gif, strength } },
  });

  it("uses as many frames as the model takes, up to the chosen length", () => {
    expect(gifFrames(gif, 5, 4)).toBe(37);
    expect(gifFrames(gif, 2, 4)).toBe(17);
    expect(gifFrames(gif, 5, 8)).toBe(33);
  });

  it("restyles with the Wan text-to-video pair, skipping into the schedule, and saves a GIF at the GIF's speed", () => {
    const wf = compile(restyle("wan22_t2v", t2v.model), parts, undefined, {}, { subfolder: "clips" });
    expectWellFormed(wf);
    expect(one(wf, "ImageFromBatch")[1].inputs).toMatchObject({ batch_index: 0, length: 37 });
    expect(one(wf, "ImageScale")[1].inputs).toMatchObject({ width: 832, height: 512, crop: "center" });
    expect(byType(wf, "EmptyHunyuanLatentVideo")).toHaveLength(0);
    const [first, second] = byType(wf, "KSamplerAdvanced").map(([, n]) => n.inputs);
    expect(follow(wf, first.latent_image).class_type).toBe("VAEEncode");
    expect(first).toMatchObject({ start_at_step: 8, end_at_step: 10 });
    expect(second).toMatchObject({ start_at_step: 10 });
    expect(one(wf, "FletchSaveGif")[1].inputs).toMatchObject({ fps: 10, folder: "", filename_prefix: "clips/fletch" });
    // A light touch leaves only the low-noise half.
    const light = byType(compile(restyle("wan22_t2v", t2v.model, 0.3), parts), "KSamplerAdvanced");
    expect(light).toHaveLength(1);
    expect(light[0][1].inputs).toMatchObject({ add_noise: "enable", start_at_step: 14 });
    expect(follow(compile(restyle("wan22_t2v", t2v.model, 0.3), parts), follow(compile(restyle("wan22_t2v", t2v.model, 0.3), parts), light[0][1].inputs.model).inputs.model).inputs.unet_name).toBe(LOW);
  });

  it("restyles with Wan 5B using denoise", () => {
    const wf = compile(restyle("wan22_5b", { folder: "diffusion_models", name: "wan2.2_ti2v_5B_fp16.safetensors" }), { umt5: parts.umt5, vae: "wan2.2_vae.safetensors" });
    expectWellFormed(wf);
    expect(one(wf, "KSampler")[1].inputs.denoise).toBe(0.6);
    expect(byType(wf, "Wan22ImageToVideoLatent")).toHaveLength(0);
  });

  it("restyles with LTX-2.3 in one pass from partway down its schedule", () => {
    const ltxParts = { gemma: "gemma_3_12B_it_fp4_mixed.safetensors", distillLora: "ltx_2.3_distilled_lora.safetensors", upscaler: "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" };
    const wf = compile(restyle("ltx23", { folder: "checkpoints", name: "ltx-2.3-22b-dev-fp8.safetensors" }), ltxParts, undefined, {}, { folder: "D:/Gifs" });
    expectWellFormed(wf);
    expect(one(wf, "ImageFromBatch")[1].inputs.length).toBe(33);
    expect(one(wf, "ManualSigmas")[1].inputs.sigmas).toBe("0.6, 0.421875, 0.0");
    expect(one(wf, "LTXVConditioning")[1].inputs.frame_rate).toBe(10);
    expect(byType(wf, "LTXVLatentUpsampler")).toHaveLength(0);
    expect(one(wf, "FletchSaveGif")[1].inputs).toMatchObject({ folder: "D:/Gifs", fps: 10 });
    expect(ltxRestyleSigmas(1)).toBe("1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0");
  });

  it("says when a GIF doesn't fit the model", () => {
    const i2v = restyle("wan22_i2v", t2v.model);
    expect(runProblems(i2v, parts)[0]).toMatch(/animates a still picture/);
    expect(runProblems({ ...base, addons: { source: { image: gif, strength: 0.6 } } }, {})[0]).toMatch(/Animated GIFs work with the video models/);
    expect(runProblems(restyle("wan22_t2v", t2v.model), parts)).toEqual([]);
  });
});
