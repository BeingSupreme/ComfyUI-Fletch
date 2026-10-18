import os
import sys
import types

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.nodes import FletchSaveGif  # noqa: E402


class Frame:
    """Stands in for a torch tensor: .cpu().numpy()."""

    def __init__(self, a):
        self.a = a

    def cpu(self):
        return self

    def numpy(self):
        return self.a


def frames(n):
    return [Frame(np.full((8, 12, 3), i / n, dtype=np.float32)) for i in range(n)]


def test_saves_a_looping_gif_into_comfy_output(tmp_path, monkeypatch):
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    ui = FletchSaveGif().save(frames(5), 12.5, "", "clips/fletch")["ui"]
    assert ui["images"] == [{"filename": "fletch_00001_.gif", "subfolder": "clips", "type": "output"}]
    gif = Image.open(tmp_path / "clips" / "fletch_00001_.gif")
    assert (gif.n_frames, gif.size, gif.info["duration"], gif.info["loop"]) == (5, (12, 8), 80, 0)


def test_saves_to_any_folder_and_counts_up(tmp_path):
    node = FletchSaveGif()
    node.save(frames(2), 10, str(tmp_path), "fletch")
    ui = node.save(frames(2), 10, str(tmp_path), "fletch")["ui"]
    assert ui["images"][0] == {"filename": "fletch_00002_.gif", "subfolder": str(tmp_path), "type": "fletch"}
