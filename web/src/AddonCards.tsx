import { useState } from "react";
import { inputImageUrl, uploadImage, type Catalog, type LoraInfo } from "./comfy";
import { controlnetsFor, faceDetectors, GUIDE_MODES, guideModeAvailable, type AddonChoices, type AddonKey } from "./addons";
import type { Profile } from "./profiles";
import type { InputImage } from "./workflow";

function Toggle(props: { label: string; on: boolean; disabled?: boolean; onChange: (on: boolean) => void }) {
  return (
    <label className={props.disabled ? "toggle disabled" : "toggle"}>
      <span>{props.label}</span>
      <input
        type="checkbox" role="switch" aria-label={props.label} checked={props.on && !props.disabled} disabled={props.disabled}
        onChange={(e) => props.onChange(e.target.checked)}
      />
    </label>
  );
}

function Slider(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="slider small">
      <span>{props.label}</span>
      <input
        type="range" min={props.min} max={props.max} step={props.step} value={props.value} aria-label={props.label}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
      <span className="weight">{props.value.toFixed(2)}</span>
    </label>
  );
}

/** A drop zone that uploads the picture to ComfyUI and shows it once it's there. */
export function ImageDrop(props: { label: string; image?: InputImage; onImage: (img: InputImage) => void; onError: (msg: string) => void }) {
  const [busy, setBusy] = useState(false);
  async function take(file: File | undefined) {
    if (!file) return;
    setBusy(true);
    try {
      props.onImage(await uploadImage(file, file.name));
    } catch (e) {
      props.onError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <label
      className="drop"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        take(e.dataTransfer.files[0]);
      }}
    >
      {props.image ? (
        <img src={inputImageUrl(props.image)} alt="" />
      ) : (
        <span className="small muted">{busy ? "Uploading…" : props.label}</span>
      )}
      <input type="file" accept="image/*" hidden aria-label={props.label} onChange={(e) => take(e.target.files?.[0])} />
    </label>
  );
}

export function AddonCards(props: {
  addons: AddonChoices;
  set: <K extends AddonKey>(key: K, patch: Partial<AddonChoices[K]>) => void;
  blocked: Record<AddonKey, string | null>;
  profile: Profile | null;
  catalog: Catalog;
  controlnets: LoraInfo[];
  onPaint: () => void;
  onError: (msg: string) => void;
}) {
  const { addons: a, set, blocked, catalog } = props;
  const nets = props.profile ? controlnetsFor(props.profile, props.controlnets) : [];
  const note = (key: AddonKey) => blocked[key] && <p className="muted small note">{blocked[key]}</p>;

  return (
    <>
      <Toggle label="Start from an image" on={a.source.on} onChange={(on) => set("source", { on })} />
      {a.source.on && (
        <div className="sub">
          <ImageDrop
            label="Drop a picture here, or click to choose one"
            image={a.source.image}
            onImage={(image) => set("source", { image, masked: undefined })}
            onError={props.onError}
          />
          {a.source.image && (
            <>
              <Slider label="Change" value={a.source.strength} min={0.05} max={1} step={0.05} onChange={(strength) => set("source", { strength })} />
              <div className="row small">
                <button className="link" onClick={props.onPaint}>
                  {a.source.masked ? "Repaint area again" : "Paint an area to change (inpaint)"}
                </button>
                {a.source.masked && (
                  <button className="link" onClick={() => set("source", { masked: undefined })}>Change the whole image</button>
                )}
              </div>
              {a.source.masked && <p className="muted small note">Only the painted area will change.</p>}
            </>
          )}
        </div>
      )}

      <Toggle label="Hires fix" on={a.hires.on} onChange={(on) => set("hires", { on })} />
      {a.hires.on && (
        <div className="sub">
          <div className="chips">
            {[1.5, 2].map((scale) => (
              <button key={scale} className={a.hires.scale === scale ? "chip on" : "chip"} onClick={() => set("hires", { scale })}>
                {scale}x
              </button>
            ))}
          </div>
          <Slider label="Refine" value={a.hires.strength} min={0.2} max={0.7} step={0.05} onChange={(strength) => set("hires", { strength })} />
        </div>
      )}

      <Toggle label="Face fix" on={a.faceFix.on} disabled={!!blocked.faceFix} onChange={(on) => set("faceFix", { on })} />
      {note("faceFix")}
      {a.faceFix.on && !blocked.faceFix && (
        <div className="sub">
          {faceDetectors(catalog).length > 1 && (
            <select aria-label="Face detector" value={a.faceFix.detector} onChange={(e) => set("faceFix", { detector: e.target.value })}>
              {faceDetectors(catalog).map((d) => <option key={d}>{d}</option>)}
            </select>
          )}
          <Slider label="Strength" value={a.faceFix.strength} min={0.1} max={0.8} step={0.05} onChange={(strength) => set("faceFix", { strength })} />
        </div>
      )}

      <Toggle label="Pose or edge guide" on={a.guide.on} disabled={!!blocked.guide} onChange={(on) => set("guide", { on })} />
      {note("guide")}
      {a.guide.on && !blocked.guide && (
        <div className="sub">
          <ImageDrop
            label="Drop the picture to follow"
            image={a.guide.image}
            onImage={(image) => set("guide", { image })}
            onError={props.onError}
          />
          <div className="chips">
            {GUIDE_MODES.filter(([m]) => guideModeAvailable(m, catalog)).map(([mode, label]) => (
              <button key={mode} className={a.guide.mode === mode ? "chip on" : "chip"} onClick={() => set("guide", { mode })}>
                {label}
              </button>
            ))}
          </div>
          <select aria-label="ControlNet model" value={a.guide.model} onChange={(e) => set("guide", { model: e.target.value })}>
            <option value="">Choose a ControlNet model</option>
            {nets.map((n) => <option key={n.name}>{n.name}</option>)}
          </select>
          <Slider label="Strength" value={a.guide.strength} min={0.1} max={1.5} step={0.05} onChange={(strength) => set("guide", { strength })} />
        </div>
      )}

      <Toggle label="Upscale" on={a.upscale.on} disabled={!!blocked.upscale} onChange={(on) => set("upscale", { on })} />
      {note("upscale")}
      {a.upscale.on && !blocked.upscale && (
        <div className="sub">
          <select aria-label="Upscale model" value={a.upscale.model} onChange={(e) => set("upscale", { model: e.target.value })}>
            {catalog.upscaleModels.map((m) => <option key={m}>{m}</option>)}
          </select>
        </div>
      )}
    </>
  );
}
