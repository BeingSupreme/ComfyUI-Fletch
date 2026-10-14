"""Fletch's nodes: save images or videos to any folder, for output folders outside ComfyUI's."""

import json
import os
import re

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _next_index(folder, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_\.png$")
    found = [int(m.group(1)) for n in os.listdir(folder) if (m := pattern.match(n))]
    return max(found, default=0) + 1


class FletchSaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "folder": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "fletch"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Fletch"

    def save(self, images, folder, filename_prefix="fletch", prompt=None, extra_pnginfo=None):
        os.makedirs(folder, exist_ok=True)
        index = _next_index(folder, filename_prefix)
        results = []
        for image in images:
            pixels = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
            meta = PngInfo()
            if prompt is not None:
                meta.add_text("prompt", json.dumps(prompt))
            for key, value in (extra_pnginfo or {}).items():
                meta.add_text(key, json.dumps(value))
            name = f"{filename_prefix}_{index:05}_.png"
            Image.fromarray(pixels).save(os.path.join(folder, name), pnginfo=meta, compress_level=4)
            results.append({"filename": name, "subfolder": folder, "type": "fletch"})
            index += 1
        return {"ui": {"images": results}}


class FletchSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "folder": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "fletch"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Fletch"

    def save(self, video, folder, filename_prefix="fletch", prompt=None, extra_pnginfo=None):
        from comfy_api.latest import Types  # only inside ComfyUI

        os.makedirs(folder, exist_ok=True)
        pattern = re.compile(rf"^{re.escape(filename_prefix)}_(\d+)_\.mp4$")
        index = max([int(m.group(1)) for n in os.listdir(folder) if (m := pattern.match(n))], default=0) + 1
        name = f"{filename_prefix}_{index:05}_.mp4"
        metadata = dict(extra_pnginfo or {})
        if prompt is not None:
            metadata["prompt"] = prompt
        video.save_to(
            os.path.join(folder, name),
            format=Types.VideoContainer("mp4"),
            codec=Types.VideoCodec("auto"),
            metadata=metadata or None,
        )
        return {"ui": {"images": [{"filename": name, "subfolder": folder, "type": "fletch"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"FletchSaveImage": FletchSaveImage, "FletchSaveVideo": FletchSaveVideo}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FletchSaveImage": "Save Image (Fletch, any folder)",
    "FletchSaveVideo": "Save Video (Fletch, any folder)",
}
