import { useEffect, useState } from "react";
import { connect, imageUrl, listCheckpoints, queuePrompt, watch, type ImageRef } from "./comfy";
import { buildTxt2Img, randomSeed, SDXL_DEFAULTS } from "./workflow";

type Status = { kind: "idle" } | { kind: "running"; step: number; of: number } | { kind: "error"; message: string };

export function App() {
  const [connected, setConnected] = useState(false);
  const [checkpoints, setCheckpoints] = useState<string[]>([]);
  const [checkpoint, setCheckpoint] = useState("");
  const [positive, setPositive] = useState("");
  const [negative, setNegative] = useState("blurry, low quality");
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [preview, setPreview] = useState<string | null>(null);
  const [history, setHistory] = useState<ImageRef[]>([]);

  useEffect(() => connect(setConnected), []);

  useEffect(() => {
    if (!connected) return;
    listCheckpoints()
      .then((list) => {
        setCheckpoints(list);
        setCheckpoint((c) => c || list[0] || "");
      })
      .catch((e) => setStatus({ kind: "error", message: String(e.message ?? e) }));
  }, [connected]);

  async function generate() {
    setStatus({ kind: "running", step: 0, of: SDXL_DEFAULTS.steps });
    setPreview(null);
    try {
      const workflow = buildTxt2Img({ ...SDXL_DEFAULTS, checkpoint, positive, negative, seed: randomSeed() });
      const id = await queuePrompt(workflow);
      const stop = watch(id, (e) => {
        if (e.kind === "progress") setStatus({ kind: "running", step: e.value, of: e.max });
        else if (e.kind === "preview") setPreview(e.url);
        else if (e.kind === "image") {
          setPreview(imageUrl(e.image));
          setHistory((h) => [e.image, ...h].slice(0, 20));
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

  const running = status.kind === "running";
  const canGenerate = connected && !!checkpoint && positive.trim() !== "" && !running;

  return (
    <div className="app">
      <header>
        <strong>Fletch</strong>
        <span className={connected ? "dot on" : "dot"} />
        <span className="muted">{connected ? "Connected to ComfyUI" : "Connecting to ComfyUI…"}</span>
      </header>

      <aside>
        <label>
          <span className="label">Model</span>
          <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
            {checkpoints.length === 0 && <option value="">No checkpoints found</option>}
            {checkpoints.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
        <p className="muted small">
          SDXL defaults: {SDXL_DEFAULTS.width} x {SDXL_DEFAULTS.height}, {SDXL_DEFAULTS.steps} steps, CFG {SDXL_DEFAULTS.cfg}
        </p>
      </aside>

      <main>
        <textarea
          placeholder="Describe the image you want"
          value={positive}
          onChange={(e) => setPositive(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGenerate && generate()}
        />
        <input placeholder="Avoid" value={negative} onChange={(e) => setNegative(e.target.value)} />
        <div className="row">
          <span className="muted small">Random seed, batch of 1</span>
          <button disabled={!canGenerate} onClick={generate}>
            {running ? "Generating…" : "Generate"}
          </button>
        </div>

        <div className="preview">
          {preview ? <img src={preview} alt="Latest result" /> : <span className="muted">Your image appears here</span>}
          {running && (
            <div className="progress">
              <span className="small muted">
                Step {status.step} of {status.of}
              </span>
              <div className="bar">
                <div style={{ width: `${(100 * status.step) / Math.max(status.of, 1)}%` }} />
              </div>
            </div>
          )}
        </div>
        {status.kind === "error" && <p className="error">{status.message}</p>}

        {history.length > 0 && (
          <div className="history">
            {history.map((img) => (
              <img key={img.filename} src={imageUrl(img)} alt="" onClick={() => setPreview(imageUrl(img))} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
