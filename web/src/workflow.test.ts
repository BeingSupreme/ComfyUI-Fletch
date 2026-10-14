import { describe, expect, it } from "vitest";
import { compile, modelProblems, pickParts, resolve, type ApiWorkflow, type Link, type Settings } from "./workflow";

const base: Settings = {
  model: { folder: "checkpoints", name: "juggernautXL.safetensors" },
  family: "sdxl",
  positive: "a red fox in snow",
  negative: "blurry",
  shape: "square",
  quality: "balanced",
  loras: [],
  seed: 42,
  advanced: {},
};

const fluxParts = { t5: "t5xxl_fp8_e4m3fn.safetensors", clipL: "clip_l.safetensors", vae: "ae.safetensors" };

// Output counts per node class, to check every link lands on a real output.
const OUTPUTS: Record<string, number> = {
  CheckpointLoaderSimple: 3,
  UNETLoader: 1,
  DualCLIPLoader: 1,
  VAELoader: 1,
  LoraLoader: 2,
  CLIPSetLastLayer: 1,
  CLIPTextEncode: 1,
  FluxGuidance: 1,
  EmptyLatentImage: 1,
  EmptySD3LatentImage: 1,
  EmptyFlux2LatentImage: 1,
  CLIPLoader: 1,
  ConditioningZeroOut: 1,
  RandomNoise: 1,
  KSamplerSelect: 1,
  Flux2Scheduler: 1,
  CFGGuider: 1,
  SamplerCustomAdvanced: 2,
  KSampler: 1,
  VAEDecode: 1,
  SaveImage: 0,
};

const isLink = (v: unknown): v is Link =>
  Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && typeof v[1] === "number";

function expectWellFormed(wf: ApiWorkflow) {
  for (const node of Object.values(wf)) {
    expect(OUTPUTS, node.class_type).toHaveProperty(node.class_type);
    for (const v of Object.values(node.inputs)) {
      if (!isLink(v)) continue;
      expect(wf[v[0]], `link to node ${v[0]}`).toBeDefined();
      expect(v[1]).toBeLessThan(OUTPUTS[wf[v[0]].class_type]);
    }
  }
  expect(Object.values(wf).filter((n) => n.class_type === "SaveImage")).toHaveLength(1);
}

const byType = (wf: ApiWorkflow, type: string) => Object.entries(wf).filter(([, n]) => n.class_type === type);
const one = (wf: ApiWorkflow, type: string) => {
  const found = byType(wf, type);
  expect(found, type).toHaveLength(1);
  return found[0];
};
const sampler = (wf: ApiWorkflow) => one(wf, "KSampler")[1].inputs;
const follow = (wf: ApiWorkflow, link: unknown) => wf[(link as Link)[0]];

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

describe("resolve", () => {
  it("reports what the run will use", () => {
    expect(resolve({ family: "sdxl", shape: "wide", quality: "best", advanced: {} })).toEqual({
      width: 1216, height: 832, steps: 40, cfg: 6, sampler: "dpmpp_2m", scheduler: "karras", clipSkip: 1,
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
