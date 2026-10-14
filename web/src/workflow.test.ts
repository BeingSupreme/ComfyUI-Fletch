import { describe, expect, it } from "vitest";
import { buildTxt2Img, SDXL_DEFAULTS, type ApiWorkflow, type Link } from "./workflow";

const settings = {
  ...SDXL_DEFAULTS,
  checkpoint: "juggernautXL.safetensors",
  positive: "a red fox in snow",
  negative: "blurry",
  seed: 42,
};

// Every link must point at a node that exists and an output that node has.
const OUTPUTS: Record<string, number> = {
  CheckpointLoaderSimple: 3,
  CLIPTextEncode: 1,
  EmptyLatentImage: 1,
  KSampler: 1,
  VAEDecode: 1,
  SaveImage: 0,
};

const isLink = (v: unknown): v is Link =>
  Array.isArray(v) && v.length === 2 && typeof v[0] === "string" && typeof v[1] === "number";

function links(wf: ApiWorkflow): Link[] {
  return Object.values(wf).flatMap((n) => Object.values(n.inputs).filter(isLink));
}

describe("buildTxt2Img", () => {
  const wf = buildTxt2Img(settings);

  it("wires every link to an existing node output", () => {
    for (const [id, out] of links(wf)) {
      expect(wf[id], `node ${id}`).toBeDefined();
      expect(out).toBeLessThan(OUTPUTS[wf[id].class_type]);
    }
  });

  it("passes the settings through to the sampler", () => {
    const k = wf["5"].inputs;
    expect(k).toMatchObject({ seed: 42, steps: 30, cfg: 6, sampler_name: "dpmpp_2m", scheduler: "karras" });
    expect(k.positive).toEqual(["2", 0]);
    expect(k.negative).toEqual(["3", 0]);
  });

  it("encodes both prompts with the checkpoint's CLIP", () => {
    expect(wf["2"].inputs).toEqual({ clip: ["1", 1], text: "a red fox in snow" });
    expect(wf["3"].inputs).toEqual({ clip: ["1", 1], text: "blurry" });
  });

  it("ends in exactly one SaveImage", () => {
    expect(Object.values(wf).filter((n) => n.class_type === "SaveImage")).toHaveLength(1);
  });
});
