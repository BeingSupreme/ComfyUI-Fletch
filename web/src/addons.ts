// Which add-ons can run with what's installed, and the UI's add-on choices.

import type { Catalog, LoraInfo } from "./comfy";
import type { Profile } from "./profiles";
import { GUIDE_PREPROCESSORS, type Addons, type GifOutput, type GuideMode, type InputImage } from "./workflow";

export interface AddonChoices {
  source: { on: boolean; image?: InputImage; masked?: InputImage; strength: number; output?: GifOutput };
  hires: { on: boolean; scale: number; strength: number };
  upscale: { on: boolean; model: string };
  faceFix: { on: boolean; detector: string; strength: number };
  guide: { on: boolean; image?: InputImage; mode: GuideMode; model: string; strength: number };
}

export const DEFAULT_ADDONS: AddonChoices = {
  source: { on: false, strength: 0.6 },
  hires: { on: false, scale: 1.5, strength: 0.45 },
  upscale: { on: false, model: "" },
  faceFix: { on: false, detector: "", strength: 0.4 },
  guide: { on: false, mode: "canny", model: "", strength: 0.7 },
};

export type AddonKey = keyof AddonChoices;

export const GUIDE_MODES: [GuideMode, string][] = [
  ["canny", "Edges"],
  ["pose", "Pose"],
  ["depth", "Depth"],
  ["asis", "Use as-is"],
];

const LORA_BASE_WORDS: Record<string, string> = { sd15: "SD 1.5", sdxl: "SDXL", flux: "Flux.1", flux2: "Flux.2", krea2: "Krea 2" };

export const faceDetectors = (catalog: Catalog) => catalog.detectors.filter((d) => /face/i.test(d));

export const controlnetsFor = (profile: Profile, controlnets: LoraInfo[]) =>
  controlnets.filter((c) => c.family === profile.loraBase || c.family === null);

export function guideModeAvailable(mode: GuideMode, catalog: Catalog) {
  const needs = GUIDE_PREPROCESSORS[mode];
  return needs.length === 0 || needs.some((n) => n === "Canny" || catalog.nodes[n]);
}

/** null when an add-on can run, otherwise what's missing, in words a person can act on. */
export function unavailable(profile: Profile | null, catalog: Catalog, controlnets: LoraInfo[]): Record<AddonKey, string | null> {
  if (profile?.video) {
    const notVideo = "Not available for video yet.";
    return {
      source: null,
      hires: notVideo,
      upscale: notVideo,
      faceFix: notVideo,
      guide: notVideo,
    };
  }
  const flux2 = profile?.arch === "flux2" ? "Not available for Flux.2 models yet." : null;
  // No ControlNets exist for Krea 2 yet.
  const noGuide = profile?.arch === "krea2" ? "Not available for Krea 2 yet." : flux2;
  return {
    source: null,
    hires: null,
    upscale: catalog.upscaleModels.length ? null : "Add an upscale model (such as 4x-UltraSharp.pth) to models/upscale_models.",
    faceFix:
      flux2 ??
      (!catalog.nodes.FaceDetailer || !catalog.nodes.UltralyticsDetectorProvider
        ? "Needs the Impact Pack and Impact Subpack custom nodes (install them from ComfyUI Manager)."
        : faceDetectors(catalog).length
          ? null
          : "Add a face detection model (face_yolov8m.pt) to models/ultralytics/bbox."),
    guide:
      noGuide ??
      (profile && controlnetsFor(profile, controlnets).length === 0
        ? `Add a ${LORA_BASE_WORDS[profile.loraBase]} ControlNet model to models/controlnet.`
        : null),
  };
}

/** The add-ons a run will use: switched on, available and fully filled in. */
export function addonsForRun(
  a: AddonChoices,
  blocked: Record<AddonKey, string | null>,
  catalog: Catalog,
  allowedControlnets: Set<string>,
): Addons {
  const out: Addons = {};
  if (a.source.on && a.source.image && !blocked.source) out.source = { image: a.source.image, masked: a.source.masked, strength: a.source.strength, output: a.source.output };
  if (a.hires.on && !blocked.hires) out.hires = { scale: a.hires.scale, strength: a.hires.strength };
  if (a.upscale.on && !blocked.upscale) {
    const model = catalog.upscaleModels.includes(a.upscale.model) ? a.upscale.model : catalog.upscaleModels[0];
    out.upscale = { model };
  }
  if (a.faceFix.on && !blocked.faceFix) {
    const faces = faceDetectors(catalog);
    out.faceFix = { detector: faces.includes(a.faceFix.detector) ? a.faceFix.detector : faces[0], strength: a.faceFix.strength };
  }
  const g = a.guide;
  if (g.on && !blocked.guide && g.image && allowedControlnets.has(g.model) && guideModeAvailable(g.mode, catalog)) {
    out.guide = { image: g.image, mode: g.mode, model: g.model, strength: g.strength };
  }
  return out;
}
