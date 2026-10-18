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


def write_gif(images, fps, path):
    """Frames (N x H x W x 3, 0-1) to a looping GIF, each frame with its own 256-colour palette."""
    frames = [
        Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)).convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
        for image in images
    ]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / max(fps, 1)), loop=0, disposal=1)


class FletchSaveGif:
    """Saves frames as an animated GIF: into ComfyUI's output (folder empty) or any folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 100.0}),
                "folder": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "fletch"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Fletch"

    def save(self, images, fps, folder="", filename_prefix="fletch"):
        if folder:
            target, subfolder, kind = folder, folder, "fletch"
        else:
            import folder_paths  # only inside ComfyUI

            subfolder, _, filename_prefix = filename_prefix.rpartition("/")
            target, kind = os.path.join(folder_paths.get_output_directory(), subfolder), "output"
        os.makedirs(target, exist_ok=True)
        pattern = re.compile(rf"^{re.escape(filename_prefix)}_(\d+)_\.gif$")
        index = max([int(m.group(1)) for n in os.listdir(target) if (m := pattern.match(n))], default=0) + 1
        name = f"{filename_prefix}_{index:05}_.gif"
        write_gif(images, fps, os.path.join(target, name))
        return {"ui": {"images": [{"filename": name, "subfolder": subfolder, "type": kind}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"FletchSaveImage": FletchSaveImage, "FletchSaveVideo": FletchSaveVideo, "FletchSaveGif": FletchSaveGif}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FletchSaveImage": "Save Image (Fletch, any folder)",
    "FletchSaveVideo": "Save Video (Fletch, any folder)",
    "FletchSaveGif": "Save GIF (Fletch)",
}
