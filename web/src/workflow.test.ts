import { describe, expect, it } from "vitest";
import { base, byType, expectWellFormed, fluxParts, follow, one, sampler } from "./testutil";
import { compile, modelProblems, nodeDefaults, pickParts, resolve, sizeFor, type Link } from "./workflow";

describe("compile", () => {
  it("builds plain SDXL text-to-image from the profile", () => {
    const wf = compile(base);
    expectWellFormed(wf);
    expect(sampler(wf)).toMatchObject({ seed: 42, steps: 30, cfg: 6, sampler_name: "dpmpp_2m", scheduler: "karras" });
    expect(one(wf, "EmptyLatentImage")[1].inputs).toEqual({ width: 1024, height: 1024, batch_size: 1 });
    expect(follow(wf, sampler(wf).positive).inputs.text).toBe("a red fox in snow");
    expect(follow(wf, sampler(wf).negative).inputs.text).toBe("blurry");
  });

  it("maps Shape and Quality through the family's profile", () => {
    const wf = compile({ ...base, family: "sd15", shape: "portrait", quality: "fast" });
    expect(one(wf, "EmptyLatentImage")[1].inputs).toMatchObject({ width: 512, height: 768 });
    expect(sampler(wf).steps).toBe(15);
  });

  it("chains LoRAs and feeds the last one to the sampler and both prompts", () => {
    const wf = compile({
      ...base,
      loras: [
        { name: "detail.safetensors", weight: 0.8 },
        { name: "style.safetensors", weight: 0.5 },
      ],
    });
    expectWellFormed(wf);
    const loras = byType(wf, "LoraLoader");
    expect(loras.map(([, n]) => n.inputs.lora_name)).toEqual(["detail.safetensors", "style.safetensors"]);
    const [[firstId], [lastId, last]] = loras;
    expect(last.inputs.model).toEqual([firstId, 0]);
    expect(sampler(wf).model).toEqual([lastId, 0]);
    expect(follow(wf, sampler(wf).positive).inputs.clip).toEqual([lastId, 1]);
    expect(follow(wf, sampler(wf).negative).inputs.clip).toEqual([lastId, 1]);
  });

  it("skips empty LoRA rows", () => {
    expect(byType(compile({ ...base, loras: [{ name: "", weight: 1 }] }), "LoraLoader")).toHaveLength(0);
  });

  it("gives Pony clip skip 2, its sampler and score tags", () => {
    const wf = compile({ ...base, family: "pony" });
    expectWellFormed(wf);
    const [skipId, skip] = one(wf, "CLIPSetLastLayer");
    expect(skip.inputs.stop_at_clip_layer).toBe(-2);
    const pos = follow(wf, sampler(wf).positive);
    expect(pos.inputs.clip).toEqual([skipId, 0]);
    expect(pos.inputs.text).toBe("score_9, score_8_up, score_7_up, a red fox in snow");
    expect(follow(wf, sampler(wf).negative).inputs.text).toBe("score_6, score_5, score_4, blurry");
    expect(sampler(wf)).toMatchObject({ sampler_name: "euler_ancestral", scheduler: "normal", steps: 25, cfg: 7 });
  });

  it("applies clip skip after LoRAs", () => {
    const wf = compile({ ...base, family: "illustrious", loras: [{ name: "x.safetensors", weight: 1 }] });
    const [loraId] = one(wf, "LoraLoader");
    expect(one(wf, "CLIPSetLastLayer")[1].inputs.clip).toEqual([loraId, 1]);
  });

  it("loads a diffusion-only Flux model with its own encoders, VAE and guidance", () => {
    const wf = compile({ ...base, model: { folder: "diffusion_models", name: "flux1-dev.safetensors" }, family: "flux_dev" }, fluxParts);
    expectWellFormed(wf);
    expect(one(wf, "UNETLoader")[1].inputs).toEqual({ unet_name: "flux1-dev.safetensors", weight_dtype: "default" });
    expect(one(wf, "DualCLIPLoader")[1].inputs).toEqual({
      clip_name1: "t5xxl_fp8_e4m3fn.safetensors",
      clip_name2: "clip_l.safetensors",
      type: "flux",
    });
    const [vaeId] = one(wf, "VAELoader");
    expect(one(wf, "VAEDecode")[1].inputs.vae).toEqual([vaeId, 0]);
    const pos = follow(wf, sampler(wf).positive);
    expect(pos.class_type).toBe("FluxGuidance");
    expect(pos.inputs.guidance).toBe(3.5);
    expect(follow(wf, sampler(wf).negative).inputs.text).toBe("");
    expect(byType(wf, "EmptySD3LatentImage")).toHaveLength(1);
    expect(sampler(wf)).toMatchObject({ cfg: 1, steps: 20, sampler_name: "euler", scheduler: "simple" });
  });

  it("runs an all-in-one Flux checkpoint through the checkpoint loader", () => {
    const wf = compile({ ...base, model: { folder: "checkpoints", name: "flux1-schnell-fp8.safetensors" }, family: "flux_schnell" });
    expectWellFormed(wf);
    expect(byType(wf, "UNETLoader")).toHaveLength(0);
    expect(byType(wf, "FluxGuidance")).toHaveLength(0);
    expect(sampler(wf).steps).toBe(4);
  });

  it("lets Advanced override the profile", () => {
    const wf = compile({ ...base, family: "pony", advanced: { steps: 12, cfg: 4, sampler: "euler", scheduler: "karras", clipSkip: 1 } });
    expect(sampler(wf)).toMatchObject({ steps: 12, cfg: 4, sampler_name: "euler", scheduler: "karras" });
    expect(byType(wf, "CLIPSetLastLayer")).toHaveLength(0);
  });

  it("refuses a diffusion-only model whose parts are missing", () => {
    const s = { ...base, model: { folder: "diffusion_models" as const, name: "flux1-dev.safetensors" }, family: "flux_dev" as const };
    expect(() => compile(s, { t5: "t5xxl.safetensors" })).toThrow(/CLIP-L.*Flux VAE/);
  });

  const klein = (family: "flux2_klein" | "flux2_klein_base") => ({
    ...base,
    model: { folder: "diffusion_models" as const, name: "flux-2-klein-4b-fp8.safetensors" },
    family,
    variant: "4b",
  });
  const kleinParts = { qwen: "qwen_3_4b.safetensors", vae: "flux2-vae.safetensors" };

  it("builds FLUX.2 Klein the way ComfyUI's own template does", () => {
    const wf = compile(klein("flux2_klein"), kleinParts);
    expectWellFormed(wf);
    expect(byType(wf, "KSampler")).toHaveLength(0);
    expect(one(wf, "CLIPLoader")[1].inputs).toEqual({ clip_name: "qwen_3_4b.safetensors", type: "flux2", device: "default" });
    expect(one(wf, "VAELoader")[1].inputs).toEqual({ vae_name: "flux2-vae.safetensors" });
    const sca = one(wf, "SamplerCustomAdvanced")[1].inputs;
    expect(follow(wf, sca.noise).inputs).toEqual({ noise_seed: 42 });
    expect(follow(wf, sca.sampler).inputs).toEqual({ sampler_name: "euler" });
    expect(follow(wf, sca.sigmas).inputs).toEqual({ steps: 4, width: 1024, height: 1024 });
    expect(follow(wf, sca.latent_image).class_type).toBe("EmptyFlux2LatentImage");
    const guider = follow(wf, sca.guider);
    expect(guider.inputs.cfg).toBe(1);
    // Distilled Klein has no negative prompt: the negative is the positive zeroed out.
    const neg = follow(wf, guider.inputs.negative);
    expect(neg.class_type).toBe("ConditioningZeroOut");
    expect(neg.inputs.conditioning).toEqual(guider.inputs.positive);
    expect(one(wf, "VAEDecode")[1].inputs.samples).toEqual([one(wf, "SamplerCustomAdvanced")[0], 0]);
  });

  it("gives Klein base a real negative prompt and CFG 5", () => {
    const wf = compile(klein("flux2_klein_base"), kleinParts);
    expectWellFormed(wf);
    const guider = follow(wf, one(wf, "SamplerCustomAdvanced")[1].inputs.guider);
    expect(guider.inputs.cfg).toBe(5);
    expect(follow(wf, guider.inputs.negative).inputs.text).toBe("blurry");
    expect(follow(wf, one(wf, "SamplerCustomAdvanced")[1].inputs.sigmas).inputs.steps).toBe(20);
  });

  it("refuses a checkpoint-folder Flux file that has no text encoder", () => {
    const s = { ...klein("flux2_klein"), model: { folder: "checkpoints" as const, name: "Flux Klein FP8.safetensors" } };
    expect(() => compile(s, kleinParts, false)).toThrow(/Move it to models\/diffusion_models/);
    expect(compile(s, kleinParts, true)).toBeTruthy();
  });
});

describe("Krea 2", () => {
  const krea = {
    ...base,
    model: { folder: "diffusion_models" as const, name: "krea2_turbo_fp8_scaled.safetensors" },
    family: "krea2_turbo" as const,
  };
  const kreaParts = { qwen: "qwen3vl_4b_fp8_scaled.safetensors", vae: "qwen_image_vae.safetensors" };

  it("builds Krea 2 Turbo the way ComfyUI's own template does", () => {
    const wf = compile({ ...krea, loras: [{ name: "krea2_darkbrush.safetensors", weight: 0.8 }] }, kreaParts);
    expectWellFormed(wf);
    expect(one(wf, "CLIPLoader")[1].inputs).toEqual({ clip_name: "qwen3vl_4b_fp8_scaled.safetensors", type: "krea2", device: "default" });
    expect(one(wf, "VAELoader")[1].inputs).toEqual({ vae_name: "qwen_image_vae.safetensors" });
    expect(byType(wf, "LoraLoader")).toHaveLength(0);
    const [loraId, lora] = one(wf, "LoraLoaderModelOnly");
    expect(lora.inputs).toMatchObject({ lora_name: "krea2_darkbrush.safetensors", strength_model: 0.8 });
    const ks = one(wf, "KSampler")[1].inputs;
    expect(ks).toMatchObject({ steps: 8, cfg: 1, sampler_name: "euler", scheduler: "simple", denoise: 1 });
    expect(ks.model).toEqual([loraId, 0]);
    expect(follow(wf, ks.negative).class_type).toBe("ConditioningZeroOut");
    expect(follow(wf, ks.latent_image).class_type).toBe("EmptyLatentImage");
  });

  it("uses a typed negative prompt only when CFG is raised above 1", () => {
    const neg = (cfg?: number) => {
      const wf = compile({ ...krea, negative: "blurry", advanced: { cfg } }, kreaParts);
      return follow(wf, one(wf, "KSampler")[1].inputs.negative);
    };
    expect(neg().class_type).toBe("ConditioningZeroOut");
    expect(neg(3).inputs.text).toBe("blurry");
  });

  it("says which Krea 2 files to add", () => {
    expect(modelProblems(krea, {})).toEqual([
      "Add the Qwen3-VL text encoder (qwen3vl_4b_fp8_scaled.safetensors) to models/text_encoders.",
      "Add the Qwen Image VAE (qwen_image_vae.safetensors) to models/vae.",
    ]);
  });

  it("finds the Qwen3-VL encoder and Qwen Image VAE, and keeps Klein off the VL encoder", () => {
    const encoders = ["qwen3vl_4b_fp8_scaled.safetensors", "qwen_3_4b.safetensors"];
    const vaes = ["flux2-vae.safetensors", "qwen_image_vae.safetensors"];
    expect(pickParts("krea2_turbo", null, encoders, vaes)).toEqual(kreaParts);
    expect(pickParts("flux2_klein", null, encoders, vaes).qwen).toBe("qwen_3_4b.safetensors");
  });
});

describe("save destination", () => {
  it("saves into ComfyUI's output by default, into a subfolder, or anywhere with Fletch's node", () => {
    expect(one(compile(base), "SaveImage")[1].inputs.filename_prefix).toBe("fletch");
    expect(one(compile(base, {}, undefined, {}, { subfolder: "portraits/2026" }), "SaveImage")[1].inputs.filename_prefix).toBe(
      "portraits/2026/fletch",
    );
    const wf = compile(base, {}, undefined, {}, { folder: "D:\\Art\\Fletch" });
    expectWellFormed(wf);
    expect(byType(wf, "SaveImage")).toHaveLength(0);
    expect(one(wf, "FletchSaveImage")[1].inputs).toMatchObject({ folder: "D:\\Art\\Fletch", filename_prefix: "fletch" });
  });
});

describe("resolve", () => {
  it("reports what the run will use", () => {
    expect(resolve({ family: "sdxl", shape: "wide", quality: "best", advanced: {} })).toEqual({
      width: 1216, height: 832, steps: 40, cfg: 6, seconds: 0, frames: 1, sampler: "dpmpp_2m", scheduler: "karras", clipSkip: 1,
    });
  });
});

describe("pickParts", () => {
  it("prefers fp8 T5 and finds clip_l and ae for FLUX.1", () => {
    expect(
      pickParts(
        "flux_dev",
        null,
        ["clip_g.safetensors", "t5xxl_fp16.safetensors", "t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors"],
        ["sdxl_vae.safetensors", "flux2-vae.safetensors", "flux/ae.safetensors"],
      ),
    ).toEqual({ t5: "t5xxl_fp8_e4m3fn.safetensors", clipL: "clip_l.safetensors", vae: "flux/ae.safetensors" });
  });

  it("falls back to any FLUX.1 file with flux in its name for the VAE", () => {
    expect(pickParts("flux_dev", null, [], ["flux2-vae.safetensors", "flux_vae.safetensors"]).vae).toBe("flux_vae.safetensors");
  });

  it("matches Klein's Qwen3 encoder to its size", () => {
    const encoders = ["t5xxl_fp16.safetensors", "qwen_3_4b.safetensors", "qwen_3_8b_fp8mixed.safetensors"];
    const vaes = ["ae.safetensors", "flux2-vae.safetensors"];
    expect(pickParts("flux2_klein", "4b", encoders, vaes)).toEqual({ qwen: "qwen_3_4b.safetensors", vae: "flux2-vae.safetensors" });
    expect(pickParts("flux2_klein", "9b", encoders, vaes).qwen).toBe("qwen_3_8b_fp8mixed.safetensors");
  });

  it("says what to add", () => {
    const s = { model: { folder: "diffusion_models" as const, name: "x" }, family: "flux_dev" as const };
    expect(modelProblems(s, {})).toHaveLength(3);
    expect(modelProblems({ ...s, family: "flux2_klein", variant: "9b" }, {})).toEqual([
      "Add the Qwen3 text encoder (qwen_3_8b_fp8mixed.safetensors) to models/text_encoders.",
      "Add the Flux 2 VAE (flux2-vae.safetensors) to models/vae.",
    ]);
    expect(modelProblems({ ...s, family: "sdxl" }, {})[0]).toMatch(/full checkpoint/);
  });
});

const photo = { name: "fletch/beach.png", width: 1200, height: 800 };

describe("add-ons", () => {
  it("starts from an image: scaled to the family's size at the image's aspect, encoded, partly denoised", () => {
    const wf = compile({ ...base, addons: { source: { image: photo, strength: 0.6 } } });
    expectWellFormed(wf);
    expect(byType(wf, "EmptyLatentImage")).toHaveLength(0);
    expect(one(wf, "LoadImage")[1].inputs).toEqual({ image: "fletch/beach.png" });
    expect(one(wf, "ImageScale")[1].inputs).toMatchObject({ width: 1280, height: 832, crop: "disabled" });
    expect(follow(wf, sampler(wf).latent_image).class_type).toBe("VAEEncode");
    expect(sampler(wf).denoise).toBe(0.6);
  });

  it("inpaints only the painted area", () => {
    const masked = { ...photo, name: "fletch/beach-mask.png" };
    const wf = compile({ ...base, addons: { source: { image: photo, masked, strength: 0.9 } } });
    expectWellFormed(wf);
    const [loadId, load] = one(wf, "LoadImage");
    expect(load.inputs.image).toBe("fletch/beach-mask.png");
    expect(one(wf, "GrowMask")[1].inputs.mask).toEqual([loadId, 1]);
    expect(follow(wf, sampler(wf).latent_image).class_type).toBe("SetLatentNoiseMask");
  });

  it("does img2img on Klein by trimming the FLUX.2 schedule", () => {
    const s = {
      ...base,
      model: { folder: "diffusion_models" as const, name: "klein.safetensors" },
      family: "flux2_klein" as const,
      variant: "4b",
      addons: { source: { image: photo, strength: 0.5 } },
    };
    const wf = compile(s, { qwen: "qwen_3_4b.safetensors", vae: "flux2-vae.safetensors" });
    expectWellFormed(wf);
    const sca = one(wf, "SamplerCustomAdvanced")[1].inputs;
    expect(sca.sigmas).toEqual([one(wf, "SplitSigmasDenoise")[0], 1]);
    expect(one(wf, "SplitSigmasDenoise")[1].inputs.denoise).toBe(0.5);
    expect(follow(wf, sca.latent_image).class_type).toBe("VAEEncode");
  });

  it("hires fix enlarges, re-encodes and refines with a second pass", () => {
    const wf = compile({ ...base, addons: { hires: { scale: 1.5, strength: 0.45 } } });
    expectWellFormed(wf);
    const passes = byType(wf, "KSampler");
    expect(passes).toHaveLength(2);
    expect(passes[0][1].inputs.denoise).toBe(1);
    expect(passes[1][1].inputs.denoise).toBe(0.45);
    expect(follow(wf, passes[1][1].inputs.latent_image).class_type).toBe("VAEEncode");
    expect(one(wf, "ImageScaleBy")[1].inputs.scale_by).toBe(1.5);
    const saved = one(wf, "SaveImage")[1].inputs.images as Link;
    expect(follow(wf, follow(wf, saved).inputs.samples)).toBe(passes[1][1]);
  });

  it("upscales last, after face fix", () => {
    const nodes = {
      FaceDetailer: { input: { required: { image: ["IMAGE"], guide_size: ["FLOAT", { default: 512 }], wildcard: ["STRING", { multiline: true }], sam_detection_hint: [["center-1", "none"]], denoise: ["FLOAT", { default: 0.5 }] } } },
      UltralyticsDetectorProvider: { input: { required: { model_name: [["bbox/face_yolov8m.pt"]] } } },
    };
    const wf = compile(
      { ...base, addons: { faceFix: { detector: "bbox/face_yolov8m.pt", strength: 0.4 }, upscale: { model: "4x-UltraSharp.pth" } } },
      {},
      undefined,
      nodes,
    );
    expectWellFormed(wf);
    const [faceId, face] = one(wf, "FaceDetailer");
    expect(face.inputs).toMatchObject({ guide_size: 512, wildcard: "", sam_detection_hint: "center-1", denoise: 0.4, seed: 42 });
    expect(follow(wf, face.inputs.bbox_detector).inputs.model_name).toBe("bbox/face_yolov8m.pt");
    const up = one(wf, "ImageUpscaleWithModel")[1];
    expect(up.inputs.image).toEqual([faceId, 0]);
    expect(follow(wf, one(wf, "SaveImage")[1].inputs.images)).toBe(up);
  });

  it("applies a guide to both prompts, with the right preprocessor", () => {
    const guide = { image: photo, mode: "canny" as const, model: "controlnet-canny-sdxl.safetensors", strength: 0.7 };
    const wf = compile({ ...base, addons: { guide } });
    expectWellFormed(wf);
    const [cnId, cn] = one(wf, "ControlNetApplyAdvanced");
    expect(follow(wf, cn.inputs.image).class_type).toBe("Canny");
    expect(cn.inputs.strength).toBe(0.7);
    expect(sampler(wf).positive).toEqual([cnId, 0]);
    expect(sampler(wf).negative).toEqual([cnId, 1]);

    const pose = compile({ ...base, addons: { guide: { ...guide, mode: "pose" } } }, {}, undefined, {
      DWPreprocessor: { input: { required: { image: ["IMAGE"], detect_hand: [["enable", "disable"]], resolution: ["INT", { default: 512 }] } } },
    });
    expectWellFormed(pose);
    expect(one(pose, "DWPreprocessor")[1].inputs).toMatchObject({ detect_hand: "enable", resolution: 512 });

    const asis = compile({ ...base, addons: { guide: { ...guide, mode: "asis" } } });
    expect(follow(asis, one(asis, "ControlNetApplyAdvanced")[1].inputs.image).class_type).toBe("LoadImage");
  });
});

describe("sizeFor", () => {
  it("keeps the source image's shape at the family's pixel count", () => {
    expect(sizeFor("sdxl", "square", { width: 1200, height: 800 })).toEqual([1280, 832]);
    expect(sizeFor("sd15", "square", { width: 800, height: 1200 })).toEqual([448, 640]);
    expect(sizeFor("sdxl", "wide")).toEqual([1216, 832]);
  });
});

describe("nodeDefaults", () => {
  it("fills widgets from defaults and skips links", () => {
    expect(
      nodeDefaults({
        input: {
          required: {
            image: ["IMAGE"],
            n: ["INT", { default: 3 }],
            mode: ["COMBO", { options: ["a", "b"], default: "b" }],
            old: [["x", "y"]],
            flag: ["BOOLEAN", { default: true }],
          },
        },
      }),
    ).toEqual({ n: 3, mode: "b", old: "x", flag: true });
  });
});
