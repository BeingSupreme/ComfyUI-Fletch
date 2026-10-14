"""Model family detection from safetensors headers.

Only the header (a JSON table of tensor names and shapes) is read, never the weights, so
checking a 6 GB checkpoint costs a few kilobytes of disk reads.
"""

import json
import os
import re
import struct

MAX_HEADER = 100 * 1024 * 1024

# Families Fletch has profiles for. "sd2" and "sd3" are recognised so the UI can say they
# aren't supported yet rather than guessing wrong.
FAMILIES = ("sd15", "sd2", "sdxl", "pony", "illustrious", "sd3", "flux_dev", "flux_schnell")


def read_header(path):
    """Return (tensors, metadata) where tensors maps name -> shape."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        if length > MAX_HEADER:
            raise ValueError(f"safetensors header too large ({length} bytes)")
        header = json.loads(f.read(length))
    metadata = header.pop("__metadata__", None) or {}
    tensors = {name: info.get("shape", []) for name, info in header.items()}
    return tensors, metadata


def _any(keys, needle):
    return any(needle in k for k in keys)


def _sdxl_flavour(name, metadata):
    """Pony and Illustrious are SDXL underneath; tell them apart by metadata, then filename."""
    text = " ".join(
        str(metadata.get(k, ""))
        for k in ("modelspec.architecture", "modelspec.title", "ss_base_model_version", "ss_sd_model_name")
    )
    text = f"{text} {os.path.basename(name)}".lower()
    if "pony" in text:
        return "pony"
    if re.search(r"illustrious|noob", text):
        return "illustrious"
    return "sdxl"


def _context_dim(tensors, pattern):
    """Cross-attention key projection input width: 768 = SD1, 1024 = SD2, 2048 = SDXL."""
    for key, shape in tensors.items():
        if pattern in key and len(shape) == 2:
            return shape[1]
    return None


def classify_model(name, tensors, metadata):
    """Classify a checkpoint or diffusion-model file.

    Returns {"family", "has_vae", "has_text_encoders"}; family is None when unknown.
    """
    keys = tensors.keys()
    has_vae = _any(keys, "first_stage_model.") or _any(keys, "vae.")
    has_te = (
        _any(keys, "cond_stage_model.") or _any(keys, "conditioner.embedders") or _any(keys, "text_encoders.")
    )

    if _any(keys, "double_blocks.") and _any(keys, "single_blocks."):
        family = "flux_dev" if _any(keys, "guidance_in.") else "flux_schnell"
    elif _any(keys, "joint_blocks."):
        family = "sd3"
    elif _any(keys, "conditioner.embedders.1") or _any(keys, "label_emb.0.0.weight"):
        family = _sdxl_flavour(name, metadata)
    else:
        dim = _context_dim(tensors, "attn2.to_k.weight")
        family = {768: "sd15", 1024: "sd2", 2048: _sdxl_flavour(name, metadata)}.get(dim)
    return {"family": family, "has_vae": has_vae, "has_text_encoders": has_te}


def classify_lora(name, tensors, metadata):
    """Return the base architecture a LoRA was trained for: "sd15", "sdxl", "flux" or None.

    Pony and Illustrious LoRAs come back as "sdxl" since they load on any SDXL model.
    """
    base = str(metadata.get("ss_base_model_version", "")).lower()
    if "flux" in base:
        return "flux"
    if "sdxl" in base or "pony" in base or "illustrious" in base:
        return "sdxl"
    if base.startswith("sd_v1") or base.startswith("sd_1"):
        return "sd15"

    keys = tensors.keys()
    if _any(keys, "double_blocks") or _any(keys, "single_transformer_blocks") or _any(keys, "single_blocks"):
        return "flux"
    if _any(keys, "lora_te2_"):
        return "sdxl"
    for key, shape in tensors.items():
        # kohya: ..._attn2_to_k.lora_down.weight; diffusers/peft: ...attn2.to_k.lora_A.weight
        if re.search(r"attn2[._]to_k\.lora_(down|A)\.weight$", key) and len(shape) == 2:
            return {768: "sd15", 2048: "sdxl"}.get(shape[1])
    return None


_cache = {}


def cached(kind, path):
    """Classify a file, caching by path and modification time."""
    stamp = os.stat(path).st_mtime_ns
    hit = _cache.get((kind, path))
    if hit and hit[0] == stamp:
        return hit[1]
    tensors, metadata = read_header(path)
    result = (classify_model if kind == "model" else classify_lora)(path, tensors, metadata)
    _cache[(kind, path)] = (stamp, result)
    return result
