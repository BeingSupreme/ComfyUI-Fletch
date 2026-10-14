import importlib
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def output(tmp_path, monkeypatch):
    out = tmp_path / "ComfyUI" / "output"
    out.mkdir(parents=True)
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(out)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    mod = importlib.reload(importlib.import_module("backend.output"))
    monkeypatch.setattr(mod, "SETTINGS", str(tmp_path / "fletch_settings.json"))
    return mod, out


def test_default_is_comfy_output(output):
    mod, out = output
    assert mod.current() == {"folder": "", "path": os.path.realpath(out), "subfolder": "", "inside_output": True}


def test_relative_name_is_a_subfolder_of_output(output):
    mod, out = output
    info = mod.set_folder("portraits/october")
    assert info["subfolder"] == "portraits/october" and info["inside_output"]
    assert (out / "portraits" / "october").is_dir()
    assert mod.current()["subfolder"] == "portraits/october"
    assert not mod.is_known(info["path"])  # served by ComfyUI's /view, not Fletch


def test_absolute_folder_elsewhere_is_remembered_and_servable(output, tmp_path):
    mod, _ = output
    elsewhere = tmp_path / "Art" / "Fletch"
    info = mod.set_folder(str(elsewhere))
    assert info["subfolder"] is None and not info["inside_output"]
    assert elsewhere.is_dir() and mod.is_known(str(elsewhere))
    assert not mod.is_known(str(tmp_path))


def test_escaping_output_with_dots_counts_as_outside(output):
    mod, _ = output
    assert not mod.describe("../elsewhere")["inside_output"]


def test_save_node_writes_png_with_metadata(tmp_path):
    np = pytest.importorskip("numpy")
    from PIL import Image

    from backend.nodes import FletchSaveImage

    class Fake:
        def __init__(self, a):
            self.a = a

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    folder = tmp_path / "out"
    images = [Fake(np.zeros((8, 8, 3), dtype=np.float32))] * 2
    result = FletchSaveImage().save(images, str(folder), prompt={"1": {}}, extra_pnginfo={"fletch": {"seed": 1}})
    names = [r["filename"] for r in result["ui"]["images"]]
    assert names == ["fletch_00001_.png", "fletch_00002_.png"]
    assert result["ui"]["images"][0] == {"filename": names[0], "subfolder": str(folder), "type": "fletch"}
    assert Image.open(folder / names[0]).text["fletch"] == '{"seed": 1}'
    again = FletchSaveImage().save(images[:1], str(folder))
    assert again["ui"]["images"][0]["filename"] == "fletch_00003_.png"
