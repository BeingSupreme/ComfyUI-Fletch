import { expect } from "vitest";
import type { ApiWorkflow, Link, Settings } from "./workflow";

export const base: Settings = {
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

export const fluxParts = { t5: "t5xxl_fp8_e4m3fn.safetensors", clipL: "clip_l.safetensors", vae: "ae.safetensors" };

// Output counts per node class, to check every link lands on a real output.
export const OUTPUTS: Record<string, number> = {
  CheckpointLoaderSimple: 3,
  UNETLoader: 1,
  DualCLIPLoader: 1,
  VAELoader: 1,
  LoraLoader: 2,
  LoraLoaderModelOnly: 1,
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
  LoadImage: 2,
  ImageScale: 1,
  ImageScaleBy: 1,
  VAEEncode: 1,
  GrowMask: 1,
  SetLatentNoiseMask: 1,
  SplitSigmasDenoise: 2,
  ControlNetLoader: 1,
  ControlNetApplyAdvanced: 2,
  Canny: 1,
  DWPreprocessor: 1,
  UpscaleModelLoader: 1,
  ImageUpscaleWithModel: 1,
  UltralyticsDetectorProvider: 2,
  FaceDetailer: 4,
  KSampler: 1,
  VAEDecode: 1,
  SaveImage: 0,
  FletchSaveImage: 0,
  ModelSamplingSD3: 1,
  EmptyHunyuanLatentVideo: 1,
  WanImageToVideo: 3,
  Wan22ImageToVideoLatent: 1,
  KSamplerAdvanced: 1,
  CreateVideo: 1,
  SaveVideo: 1,
  FletchSaveVideo: 0,
  LTXVAudioVAELoader: 1,
  LTXAVTextEncoderLoader: 1,
  LTXVConditioning: 2,
  LTXVPreprocess: 1,
  LTXVImgToVideoInplace: 1,
  ManualSigmas: 1,
  EmptyLTXVLatentVideo: 1,
  LTXVEmptyLatentAudio: 1,
  LTXVConcatAVLatent: 1,
  LTXVSeparateAVLatent: 2,
  LatentUpscaleModelLoader: 1,
  LTXVLatentUpsampler: 1,
  LTXVCropGuides: 3,
  VAEDecodeTiled: 1,
  LTXVAudioVAEDecode: 1,
};

export const isLink = (v: unknown): v is Link =>
  Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && typeof v[1] === "number";

export function expectWellFormed(wf: ApiWorkflow) {
  for (const node of Object.values(wf)) {
    expect(OUTPUTS, node.class_type).toHaveProperty(node.class_type);
    for (const v of Object.values(node.inputs)) {
      if (!isLink(v)) continue;
      expect(wf[v[0]], `link to node ${v[0]}`).toBeDefined();
      expect(v[1]).toBeLessThan(OUTPUTS[wf[v[0]].class_type]);
    }
  }
  const saves = ["SaveImage", "FletchSaveImage", "SaveVideo", "FletchSaveVideo"];
  expect(Object.values(wf).filter((n) => saves.includes(n.class_type))).toHaveLength(1);
}

export const byType = (wf: ApiWorkflow, type: string) => Object.entries(wf).filter(([, n]) => n.class_type === type);
export const one = (wf: ApiWorkflow, type: string) => {
  const found = byType(wf, type);
  expect(found, type).toHaveLength(1);
  return found[0];
};
export const sampler = (wf: ApiWorkflow) => one(wf, "KSampler")[1].inputs;
export const follow = (wf: ApiWorkflow, link: unknown) => wf[(link as Link)[0]];

