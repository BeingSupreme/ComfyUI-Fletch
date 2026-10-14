import { useEffect, useMemo, useState } from "react";
import { AddonCards } from "./AddonCards";
import { addonsForRun, controlnetsFor, DEFAULT_ADDONS, unavailable, type AddonChoices, type AddonKey } from "./addons";
import { HistoryBar } from "./HistoryBar";
import { MaskPainter } from "./MaskPainter";
import {
  connect,
  imageUrl,
  inputImageUrl,
  listControlnets,
  listLoras,
  loadCatalog,
  modelInfo,
  queuePrompt,
  uploadImage,
  watch,
  type Catalog,
  type ImageRef,
  type LoraInfo,
  getOutputFolder,
  type OutputFolder,
} from "./comfy";
import { FAMILIES, isFamily, PROFILES, UNSUPPORTED, type Family, type Quality, type Shape } from "./profiles";
import { load, save } from "./store";
import {
  compile,
  modelProblems,
  pickParts,
  randomSeed,
  resolve,
  type Advanced,
  type LoraChoice,
  type Destination,
  type ModelFolder,
  type Settings,
} from "./workflow";

/** Fletch's own save node is only needed for folders outside ComfyUI's output. */
const destination = (o: OutputFolder | null): Destination =>
  !o ? {} : o.inside_output ? { subfolder: o.subfolder || undefined } : { folder: o.path };

type Status = { kind: "idle" } | { kind: "running"; step: number; of: number } | { kind: "error"; message: string };

interface HistoryEntry {
  image: ImageRef;
  settings: Settings;
  /** True when the family was picked by hand rather than detected. */
  familyOverridden?: boolean;
}

/** What the user has chosen; the family and seed are worked out when a run starts. */
interface Choices {
  model: { folder: ModelFolder; name: string } | null;
  positive: string;
  negative: string;
  shape: Shape;
  quality: Quality;
  lorasOn: boolean;
  loras: LoraChoice[];
  advanced: Advanced;
  /** The family the Advanced numbers were set for; they don't carry over to another family. */
  advancedFor?: Family | null;
  /** Fixed seed, or null for a new random seed every run. */
  seed: number | null;
  addons: AddonChoices;
}

const DEFAULT_CHOICES: Choices = {
  model: null,
  positive: "",
  negative: "blurry, low quality",
  shape: "square",
  quality: "balanced",
  lorasOn: false,
  loras: [{ name: "", weight: 0.8 }],
  advanced: {},
  seed: null,
  addons: DEFAULT_ADDONS,
};

/** Saved choices from an older version may lack newer fields; fill them from the defaults. */
function loadChoices(): Choices {
  const saved = load<Partial<Choices>>("choices", {});
  const addons = Object.fromEntries(
    (Object.keys(DEFAULT_ADDONS) as AddonKey[]).map((k) => [k, { ...DEFAULT_ADDONS[k], ...saved.addons?.[k] }]),
  ) as unknown as AddonChoices;
  return { ...DEFAULT_CHOICES, ...saved, addons };
}

const EMPTY_CATALOG: Catalog = {
  checkpoints: [],
  diffusionModels: [],
  textEncoders: [],
  vaes: [],
  samplers: [],
  schedulers: [],
  upscaleModels: [],
  detectors: [],
  nodes: {},
};

const modelKey = (m: { folder: string; name: string }) => `${m.folder}:${m.name}`;

const SHAPES: [Shape, string][] = [["square", "Square"], ["portrait", "Portrait"], ["wide", "Wide"]];
const QUALITIES: [Quality, string][] = [["fast", "Fast"], ["balanced", "Balanced"], ["best", "Best"]];

function Chips<T extends string>(props: { options: [T, string][]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="chips">
      {props.options.map(([v, label]) => (
        <button key={v} className={v === props.value ? "chip on" : "chip"} onClick={() => props.onChange(v)}>
          {label}
        </button>
      ))}
    </div>
  );
}

function Toggle(props: { label: string; on: boolean; onChange: (on: boolean) => void }) {
  return (
    <label className="toggle">
      <span>{props.label}</span>
      <input type="checkbox" role="switch" checked={props.on} onChange={(e) => props.onChange(e.target.checked)} />
    </label>
  );
}

const numberOrUndefined = (v: string) => (v.trim() === "" || Number.isNaN(Number(v)) ? undefined : Number(v));

export function App() {
  const [connected, setConnected] = useState(false);
  const [catalog, setCatalog] = useState<Catalog>(EMPTY_CATALOG);
  const [loras, setLoras] = useState<LoraInfo[]>([]);
  const [choices, setChoices] = useState<Choices>(loadChoices);
  const [controlnets, setControlnets] = useState<LoraInfo[]>([]);
  const [painting, setPainting] = useState(false);
  const [detected, setDetected] = useState<{
    key: string;
    family: string | null;
    variant?: string | null;
    hasTextEncoders?: boolean;
    reason?: string;
  } | null>(null);
  const [overrides, setOverrides] = useState<Record<string, Family>>(() => load("familyOverrides", {}));
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [preview, setPreview] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => load("history", []));
  const [selected, setSelected] = useState<HistoryEntry | null>(null);
  const [output, setOutput] = useState<OutputFolder | null>(null);
  useEffect(() => {
    getOutputFolder().then(setOutput, () => setOutput(null));
  }, []);

  const update = (patch: Partial<Choices>) => setChoices((c) => ({ ...c, ...patch }));
  const setAddon = <K extends AddonKey>(key: K, patch: Partial<AddonChoices[K]>) =>
    setChoices((c) => ({ ...c, addons: { ...c.addons, [key]: { ...c.addons[key], ...patch } } }));

  useEffect(() => save("choices", choices), [choices]);
  useEffect(() => save("history", history), [history]);
  useEffect(() => save("familyOverrides", overrides), [overrides]);
  useEffect(() => connect(setConnected), []);

  useEffect(() => {
    if (!connected) return;
    loadCatalog().then((c) => {
      setCatalog(c);
      setChoices((ch) => {
        const all = [
          ...c.checkpoints.map((name) => ({ folder: "checkpoints" as const, name })),
          ...c.diffusionModels.map((name) => ({ folder: "diffusion_models" as const, name })),
        ];
        const stillThere = ch.model && all.some((m) => modelKey(m) === modelKey(ch.model!));
        return stillThere ? ch : { ...ch, model: all[0] ?? null };
      });
    });
    listLoras().then(setLoras, () => setLoras([]));
    listControlnets().then(setControlnets, () => setControlnets([]));
  }, [connected]);

  // Detect the family of the chosen model.
  const key = choices.model ? modelKey(choices.model) : "";
  useEffect(() => {
    if (!choices.model || !connected) return;
    const k = modelKey(choices.model);
    modelInfo(choices.model.folder, choices.model.name).then(
      (info) =>
        setDetected({ key: k, family: info.family, variant: info.variant, hasTextEncoders: info.has_text_encoders, reason: info.reason }),
      (e) => setDetected({ key: k, family: null, reason: String(e.message ?? e) }),
    );
  }, [key, connected]);

  const detectedFamily = detected?.key === key && isFamily(detected.family) ? detected.family : null;
  const family: Family | null = overrides[key] ?? detectedFamily;
  // Steps or CFG typed for SDXL would be wrong for Krea or Flux, so overrides stay with their family.
  const advanced: Advanced = family && choices.advancedFor === family ? choices.advanced : {};
  const setAdvanced = (patch: Partial<Advanced>) =>
    setChoices((c) => ({ ...c, advanced: { ...(c.advancedFor === family ? c.advanced : {}), ...patch }, advancedFor: family }));
  const profile = family ? PROFILES[family] : null;
  const current = detected?.key === key ? detected : null;
  const variant = current?.variant ?? null;
  const parts = useMemo(
    () => (family ? pickParts(family, variant, catalog.textEncoders, catalog.vaes) : {}),
    [family, variant, catalog],
  );
  const problems = choices.model && family ? modelProblems({ model: choices.model, family, variant }, parts, current?.hasTextEncoders) : [];
  const compatibleLoras = profile ? loras.filter((l) => l.family === profile.loraBase) : [];
  const unknownLoras = loras.filter((l) => l.family === null);
  const usableLoras = new Set([...compatibleLoras, ...unknownLoras].map((l) => l.name));
  const blocked = unavailable(profile, catalog, controlnets);
  const usableControlnets = new Set(profile ? controlnetsFor(profile, controlnets).map((c) => c.name) : []);
  const runAddons = addonsForRun(choices.addons, blocked, catalog, usableControlnets);
  const numbers = family
    ? resolve({ family, shape: choices.shape, quality: choices.quality, advanced, addons: runAddons })
    : null;

  function settingsForRun(): Settings | null {
    if (!choices.model || !family) return null;
    return {
      model: choices.model,
      family,
      variant,
      positive: choices.positive,
      negative: choices.negative,
      shape: choices.shape,
      quality: choices.quality,
      // Drop LoRAs left over from a model of another family.
      loras: choices.lorasOn ? choices.loras.filter((l) => usableLoras.has(l.name)) : [],
      seed: choices.seed ?? randomSeed(),
      advanced,
      addons: runAddons,
    };
  }

  async function generate() {
    const settings = settingsForRun();
    if (!settings || !numbers) return;
    setStatus({ kind: "running", step: 0, of: numbers.steps });
    setPreview(null);
    setSelected(null);
    try {
      const id = await queuePrompt(compile(settings, parts, current?.hasTextEncoders, catalog.nodes, destination(output)), { fletch: settings });
      const stop = watch(id, (e) => {
        if (e.kind === "progress") setStatus({ kind: "running", step: e.value, of: e.max });
        else if (e.kind === "preview") setPreview(e.url);
        else if (e.kind === "image") {
          const entry = { image: e.image, settings, familyOverridden: !!overrides[modelKey(settings.model)] };
          setPreview(imageUrl(e.image));
          setSelected(entry);
          setHistory((h) => [entry, ...h].slice(0, 40));
        } else if (e.kind === "error") {
          setStatus({ kind: "error", message: e.message });
          stop();
        } else if (e.kind === "done") {
          setStatus({ kind: "idle" });
          stop();
        }
      });
    } catch (e) {
      setStatus({ kind: "error", message: String((e as Error).message ?? e) });
    }
  }

  function reuse(entry: HistoryEntry) {
    const s = entry.settings;
    if (entry.familyOverridden) setOverrides((o) => ({ ...o, [modelKey(s.model)]: s.family }));
    setChoices((c) => ({
      ...c,
      model: s.model,
      positive: s.positive,
      negative: s.negative,
      shape: s.shape,
      quality: s.quality,
      lorasOn: s.loras.length > 0,
      loras: s.loras.length ? s.loras : c.loras,
      advanced: s.advanced,
      advancedFor: s.family,
      seed: s.seed,
      addons: {
        source: { ...c.addons.source, on: !!s.addons?.source, ...s.addons?.source },
        hires: { ...c.addons.hires, on: !!s.addons?.hires, ...s.addons?.hires },
        upscale: { ...c.addons.upscale, on: !!s.addons?.upscale, ...s.addons?.upscale },
        faceFix: { ...c.addons.faceFix, on: !!s.addons?.faceFix, ...s.addons?.faceFix },
        guide: { ...c.addons.guide, on: !!s.addons?.guide, ...s.addons?.guide },
      },
    }));
  }

  /** Use a finished image as the start image for the next run. */
  async function startFrom(entry: HistoryEntry) {
    try {
      const blob = await (await fetch(imageUrl(entry.image))).blob();
      setAddon("source", { on: true, image: await uploadImage(blob, entry.image.filename), masked: undefined });
    } catch (e) {
      setStatus({ kind: "error", message: String((e as Error).message ?? e) });
    }
  }

  async function saveMask(masked: Blob) {
    setPainting(false);
    const src = choices.addons.source.image;
    if (!src) return;
    try {
      const base = src.name.split("/").pop()!.replace(/\.[^.]+$/, "");
      setAddon("source", { masked: await uploadImage(masked, `${base}-mask.png`), strength: Math.max(choices.addons.source.strength, 0.85) });
    } catch (e) {
      setStatus({ kind: "error", message: String((e as Error).message ?? e) });
    }
  }

  const setLora = (i: number, patch: Partial<LoraChoice>) =>
    update({ loras: choices.loras.map((l, j) => (j === i ? { ...l, ...patch } : l)) });

  const running = status.kind === "running";
  const canGenerate = connected && !!family && problems.length === 0 && choices.positive.trim() !== "" && !running;
  const familyName = (f: Family) => PROFILES[f].label;

  return (
    <div className="app">
      <header>
        <strong>Fletch</strong>
        <span className={connected ? "dot on" : "dot"} />
        <span className="muted">{connected ? "Connected to ComfyUI" : "Connecting to ComfyUI…"}</span>
      </header>

      <aside>
        <section>
          <span className="label">Model</span>
          <select
            value={key}
            onChange={(e) => {
              const [folder, ...rest] = e.target.value.split(":");
              update({ model: { folder: folder as ModelFolder, name: rest.join(":") } });
            }}
          >
            {!choices.model && <option value="">No models found</option>}
            {catalog.checkpoints.length > 0 && (
              <optgroup label="Checkpoints">
                {catalog.checkpoints.map((n) => (
                  <option key={n} value={`checkpoints:${n}`}>{n}</option>
                ))}
              </optgroup>
            )}
            {catalog.diffusionModels.length > 0 && (
              <optgroup label="Diffusion models">
                {catalog.diffusionModels.map((n) => (
                  <option key={n} value={`diffusion_models:${n}`}>{n}</option>
                ))}
              </optgroup>
            )}
          </select>
          {choices.model && (
            <div className="family">
              <select
                className={family ? "badge" : "badge warn"}
                aria-label="Model family"
                value={overrides[key] ?? ""}
                onChange={(e) => {
                  const v = e.target.value;
                  setOverrides((o) => {
                    const next = { ...o };
                    if (isFamily(v)) next[key] = v;
                    else delete next[key];
                    return next;
                  });
                }}
              >
                <option value="">
                  {detectedFamily ? `${familyName(detectedFamily)} (detected)` : detected?.key === key ? "Pick a family" : "Detecting…"}
                </option>
                {FAMILIES.map((f) => (
                  <option key={f} value={f}>{familyName(f)}</option>
                ))}
              </select>
            </div>
          )}
          {numbers && profile && (
            <p className="muted small">
              Auto: {numbers.width} x {numbers.height}, {numbers.steps} steps, CFG {numbers.cfg}
              {numbers.clipSkip > 1 && `, clip skip ${numbers.clipSkip}`}
              {profile.positivePrefix && ", adds quality tags"}
            </p>
          )}
          {detected?.key === key && !detectedFamily && !overrides[key] && (
            <p className="warn-text small">
              {detected.family
                ? `This looks like a ${UNSUPPORTED[detected.family] ?? detected.family} model, which Fletch doesn't support yet.`
                : "Couldn't tell what kind of model this is."}{" "}
              Pick its family above to use it anyway.
            </p>
          )}
          {problems.map((p) => (
            <p key={p} className="warn-text small">{p}</p>
          ))}
        </section>

        <section>
          <span className="label">Shape</span>
          {runAddons.source ? (
            <p className="muted small">Follows your start image</p>
          ) : (
            <Chips options={SHAPES} value={choices.shape} onChange={(shape) => update({ shape })} />
          )}
        </section>

        <section>
          <span className="label">Quality</span>
          <Chips options={QUALITIES} value={choices.quality} onChange={(quality) => update({ quality, advanced: { ...advanced, steps: undefined }, advancedFor: family })} />
        </section>

        <section>
          <span className="label">Add-ons</span>
          <Toggle label="LoRA" on={choices.lorasOn} onChange={(lorasOn) => update({ lorasOn })} />
          {choices.lorasOn && (
            <div className="sub">
              {compatibleLoras.length + unknownLoras.length === 0 && (
                <p className="muted small">No {profile ? familyName(family!) : ""} LoRAs found in models/loras.</p>
              )}
              {choices.loras.map((l, i) => (
                <div key={i} className="lora">
                  <select value={l.name} onChange={(e) => setLora(i, { name: e.target.value })}>
                    <option value="">Choose a LoRA</option>
                    {compatibleLoras.length > 0 && (
                      <optgroup label={`For ${profile!.loraBase === "sdxl" ? "SDXL, Pony and Illustrious" : profile!.loraBase === "flux2" ? "Flux.2 Klein" : profile!.loraBase === "krea2" ? "Krea 2" : familyName(family!)}`}>
                        {compatibleLoras.map((x) => (
                          <option key={x.name}>{x.name}</option>
                        ))}
                      </optgroup>
                    )}
                    {unknownLoras.length > 0 && (
                      <optgroup label="Base model unknown">
                        {unknownLoras.map((x) => (
                          <option key={x.name}>{x.name}</option>
                        ))}
                      </optgroup>
                    )}
                  </select>
                  <div className="row">
                    <input
                      type="range" min={0} max={1.5} step={0.05} value={l.weight} aria-label="LoRA weight"
                      onChange={(e) => setLora(i, { weight: Number(e.target.value) })}
                    />
                    <span className="small weight">{l.weight.toFixed(2)}</span>
                    {choices.loras.length > 1 && (
                      <button className="link" onClick={() => update({ loras: choices.loras.filter((_, j) => j !== i) })}>
                        Remove
                      </button>
                    )}
                  </div>
                </div>
              ))}
              <button className="link" onClick={() => update({ loras: [...choices.loras, { name: "", weight: 0.8 }] })}>
                Add another LoRA
              </button>
            </div>
          )}

          <AddonCards
            addons={choices.addons}
            set={setAddon}
            blocked={blocked}
            profile={profile}
            catalog={catalog}
            controlnets={controlnets}
            onPaint={() => setPainting(true)}
            onError={(message) => setStatus({ kind: "error", message })}
          />
        </section>

        <details>
          <summary>Advanced</summary>
          <div className="advanced">
            <label>
              <span className="label">Steps</span>
              <input
                inputMode="numeric" placeholder={numbers ? String(profile!.steps[choices.quality]) : ""}
                value={advanced.steps ?? ""} onChange={(e) => setAdvanced({ steps: numberOrUndefined(e.target.value) })}
              />
            </label>
            <label>
              <span className="label">CFG</span>
              <input
                inputMode="decimal" placeholder={profile ? String(profile.cfg) : ""}
                value={advanced.cfg ?? ""} onChange={(e) => setAdvanced({ cfg: numberOrUndefined(e.target.value) })}
              />
            </label>
            <label>
              <span className="label">Sampler</span>
              <select value={advanced.sampler ?? ""} onChange={(e) => setAdvanced({ sampler: e.target.value || undefined })}>
                <option value="">Auto{profile ? ` (${profile.sampler})` : ""}</option>
                {catalog.samplers.map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
            <label>
              <span className="label">Scheduler</span>
              <select
                value={advanced.scheduler ?? ""}
                disabled={profile?.arch === "flux2"}
                onChange={(e) => setAdvanced({ scheduler: e.target.value || undefined })}
              >
                <option value="">{profile?.arch === "flux2" ? "Flux 2 schedule" : `Auto${profile ? ` (${profile.scheduler})` : ""}`}</option>
                {catalog.schedulers.map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
            <label>
              <span className="label">Clip skip</span>
              <select
                value={advanced.clipSkip ?? ""}
                onChange={(e) => setAdvanced({ clipSkip: numberOrUndefined(e.target.value) })}
              >
                <option value="">Auto{profile ? ` (${profile.clipSkip})` : ""}</option>
                <option value="1">1</option>
                <option value="2">2</option>
              </select>
            </label>
            <label>
              <span className="label">Seed</span>
              <input
                inputMode="numeric" placeholder="Random each time" value={choices.seed ?? ""}
                onChange={(e) => update({ seed: numberOrUndefined(e.target.value) ?? null })}
              />
            </label>
            <button className="link" onClick={() => update({ advanced: {}, seed: null })}>Reset to auto</button>
          </div>
        </details>
      </aside>

      <main>
        <textarea
          placeholder="Describe the image you want"
          value={choices.positive}
          onChange={(e) => update({ positive: e.target.value })}
          onKeyDown={(e) => e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGenerate && generate()}
        />
        {(!profile || profile.negative) && (
          <input placeholder="Avoid" value={choices.negative} onChange={(e) => update({ negative: e.target.value })} />
        )}
        <div className="generate">
          <button className="primary" disabled={!canGenerate} onClick={generate}>
            {running ? "Generating…" : "Generate"}
          </button>
          <span className="muted small">{choices.seed === null ? "Random seed" : `Seed ${choices.seed}`}</span>
        </div>

        <div className="preview">
          {preview ? <img src={preview} alt="Latest result" /> : <span className="muted">Your image appears here</span>}
          {running && (
            <div className="progress">
              <span className="small muted">Step {status.step} of {status.of}</span>
              <div className="bar">
                <div style={{ width: `${(100 * status.step) / Math.max(status.of, 1)}%` }} />
              </div>
            </div>
          )}
        </div>
        {status.kind === "error" && <p className="error">{status.message}</p>}
        {selected && !running && (
          <div className="row">
            <span className="muted small">
              {familyName(selected.settings.family)}, seed {selected.settings.seed}
              {selected.settings.loras.length > 0 && `, ${selected.settings.loras.length} LoRA`}
            </span>
            <span className="actions">
              <button className="link" onClick={() => startFrom(selected)}>Start from this image</button>
              <button className="link" onClick={() => reuse(selected)}>Use these settings</button>
            </span>
          </div>
        )}

        <HistoryBar
          output={output}
          onOutput={setOutput}
          count={history.length}
          onClear={() => {
            setHistory([]);
            setSelected(null);
            setPreview(null);
          }}
          onError={(message) => setStatus({ kind: "error", message })}
        />
        {history.length > 0 && (
          <div className="history">
            {history.map((h) => (
              <img
                key={`${h.image.subfolder}/${h.image.filename}`}
                src={imageUrl(h.image)}
                alt=""
                className={selected === h ? "on" : ""}
                onClick={() => {
                  setPreview(imageUrl(h.image));
                  setSelected(h);
                }}
              />
            ))}
          </div>
        )}
      </main>
      {painting && choices.addons.source.image && (
        <MaskPainter
          src={inputImageUrl(choices.addons.source.image)}
          onDone={saveMask}
          onCancel={() => setPainting(false)}
        />
      )}
    </div>
  );
}
