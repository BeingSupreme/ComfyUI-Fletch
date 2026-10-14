"""Where Fletch saves images: ComfyUI's output folder (default), a folder inside it, or any
folder on the machine. The choice lives in fletch_settings.json next to this package."""

import json
import os
import subprocess
import sys

import folder_paths

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SETTINGS = os.path.join(ROOT, "fletch_settings.json")


def _load():
    try:
        with open(SETTINGS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _store(data):
    with open(SETTINGS, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def describe(folder):
    """Resolve a folder the way saving will: relative names land inside ComfyUI's output."""
    out = os.path.realpath(folder_paths.get_output_directory())
    path = os.path.realpath(os.path.join(out, os.path.expanduser(folder or "")))
    inside = os.path.commonpath([out, path]) == out
    sub = os.path.relpath(path, out).replace(os.sep, "/") if inside else None
    return {"folder": folder or "", "path": path, "subfolder": "" if sub == "." else sub, "inside_output": inside}


def current():
    return describe(_load().get("output_folder", ""))


def set_folder(folder):
    """Make the folder (if needed), check it can be written to, and remember it."""
    info = describe(folder.strip())
    os.makedirs(info["path"], exist_ok=True)
    probe = os.path.join(info["path"], ".fletch-write-test")
    with open(probe, "w") as f:
        f.write("ok")
    os.remove(probe)
    data = _load()
    data["output_folder"] = info["folder"]
    known = set(data.get("known_folders", []))
    if not info["inside_output"]:
        known.add(info["path"])
    data["known_folders"] = sorted(known)
    _store(data)
    return info


def is_known(path):
    """Folders outside ComfyUI's output that Fletch may serve images from: ones the user chose."""
    path = os.path.realpath(path)
    return path in {os.path.realpath(p) for p in _load().get("known_folders", [])}


def pick_folder(start):
    """Show the system folder picker on the machine running ComfyUI. None if cancelled."""
    import tkinter
    from tkinter import filedialog

    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        chosen = filedialog.askdirectory(initialdir=start, title="Choose where Fletch saves images")
    finally:
        root.destroy()
    return chosen or None


def open_folder(path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: S606 - opens Explorer on the user's own folder
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])
