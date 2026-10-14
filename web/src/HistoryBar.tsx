import { useState } from "react";
import { openOutputFolder, pickOutputFolder, setOutputFolder, type OutputFolder } from "./comfy";

const folderName = (o: OutputFolder) =>
  o.inside_output ? (o.subfolder ? `output/${o.subfolder}` : "ComfyUI output") : (o.path.split(/[\\/]/).filter(Boolean).pop() ?? o.path);

/** The row above the history: where images are saved, and clearing the history. */
export function HistoryBar(props: {
  output: OutputFolder | null;
  onOutput: (o: OutputFolder) => void;
  count: number;
  onClear: () => void;
  onError: (message: string) => void;
}) {
  const { output } = props;
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [canBrowse, setCanBrowse] = useState(true);
  const [busy, setBusy] = useState(false);

  async function choose(folder: string) {
    setBusy(true);
    try {
      props.onOutput(await setOutputFolder(folder));
      setEditing(false);
    } catch (e) {
      props.onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function browse() {
    setBusy(true);
    try {
      const picked = await pickOutputFolder();
      if (picked) await choose(picked);
    } catch (e) {
      if ((e as { status?: number }).status === 501) setCanBrowse(false);
      else props.onError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="history-bar">
      <div className="row">
        <span className="label">History{props.count > 0 && ` (${props.count})`}</span>
        <span className="actions">
          {output && (
            <button
              className="link"
              title={output.path}
              onClick={() => {
                setDraft(output.folder);
                setEditing(!editing);
              }}
            >
              Save to: {folderName(output)}
            </button>
          )}
          {output && (
            <button className="link" onClick={() => openOutputFolder().catch((e) => props.onError((e as Error).message))}>
              Open folder
            </button>
          )}
          <button
            className="link"
            disabled={props.count === 0}
            onClick={() => window.confirm("Clear the history? Your saved images stay in their folder.") && props.onClear()}
          >
            Clear history
          </button>
        </span>
      </div>
      {editing && output && (
        <div className="output-edit">
          <p className="muted small">
            Now saving to <code>{output.path}</code>. Type a full path, or a name to make a folder inside ComfyUI's output.
          </p>
          <div className="row">
            <input
              aria-label="Output folder"
              placeholder="ComfyUI's output folder"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && choose(draft)}
            />
            {canBrowse && (
              <button disabled={busy} onClick={browse}>
                Browse…
              </button>
            )}
            <button className="primary" disabled={busy} onClick={() => choose(draft)}>
              Save
            </button>
          </div>
          <div className="row">
            {output.folder !== "" && (
              <button className="link" onClick={() => choose("")}>
                Use ComfyUI's output folder
              </button>
            )}
            <button className="link" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
