"""Model family detection from safetensors headers.

Only the header (a JSON table of tensor names and shapes) is read, never the weights, so
checking a 6 GB checkpoint costs a few kilobytes of disk reads.
"""

import json
import os
import re
import struct

MAX_HEADER = 100 * 1024 * 1024

# Families Fletch has profiles for. "sd2", "sd3", "flux2" (FLUX.2 dev) and "krea2_raw" are
# recognised so the UI can say they aren't supported yet rather than guessing wrong.
FAMILIES = (
    "sd15", "sd2", "sdxl", "pony", "illustrious", "sd3",
    "flux_dev", "flux_schnell", "flux2", "flux2_klein", "flux2_klein_base",
    "krea2_turbo", "krea2_raw",
    "wan22_t2v", "wan22_i2v", "wan22_5b", "wan21",
    "ltx23", "ltx2", "ltxv",
)

# FLUX.2 text-input width = 3 x the text encoder's hidden size. FLUX.1 reads T5 at 4096.
FLUX2_TEXT_WIDTHS = {7680: "4b", 12288: "9b"}  # Qwen3 4B, Qwen3 8B (Klein)


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


def _shape(tensors, suffix):
    for key, shape in tensors.items():
        if key.endswith(suffix):
            return shape
    return None


def _flux(name, tensors):
    """FLUX.1 dev / schnell, or FLUX.2 Klein (4B or 9B, distilled or base)."""
    keys = tensors.keys()
    txt = _shape(tensors, "txt_in.weight")
    width = txt[1] if txt and len(txt) == 2 else None
    if _any(keys, "double_stream_modulation_img") or (width is not None and width != 4096):
        variant = FLUX2_TEXT_WIDTHS.get(width)
        if variant is None:
            return "flux2", None
        base = "base" in os.path.basename(name).lower()
        return ("flux2_klein_base" if base else "flux2_klein"), variant
    return ("flux_dev" if _any(keys, "guidance_in.") else "flux_schnell"), None


def _wan(name, tensors):
    """Wan video models. The patch embedding's shape gives size (width) and input channels:
    5120 wide is 14B (16 channels = text to video, 36 = image to video), 3072 x 48 is the
    Wan 2.2 5B. Wan 2.1 and 2.2 14B share a layout; 2.2 ships as high/low-noise pairs."""
    shape = _shape(tensors, "patch_embedding.weight") or []
    width, channels = (shape[0], shape[1]) if len(shape) >= 2 else (None, None)
    base = os.path.basename(name).lower()
    if _any(tensors.keys(), "vace_blocks."):
        return "wan21"
    if width == 3072 and channels == 48:
        return "wan22_5b"
    if width == 5120 and re.search(r"wan.?2\.?2|high.?noise|low.?noise", base):
        return "wan22_i2v" if channels == 36 else "wan22_t2v"
    return "wan21"


def classify_model(name, tensors, metadata):
    """Classify a checkpoint or diffusion-model file.

    Returns {"family", "variant", "has_vae", "has_text_encoders"}; family is None when unknown.
    variant is "4b" or "9b" for FLUX.2 Klein, which picks its text encoder.
    """
    keys = tensors.keys()
    has_vae = _any(keys, "first_stage_model.") or _any(keys, "vae.")
    has_te = (
        _any(keys, "cond_stage_model.") or _any(keys, "conditioner.embedders") or _any(keys, "text_encoders.")
    )

    variant = None
    if _any(keys, "adaln_single.emb.timestep_embedder.linear_1.bias"):
        # Lightricks LTX. The audio+video ones (LTX-2 family) carry an audio timestep head;
        # 2.0 and 2.3 share it, so the version comes from the filename.
        if not _any(keys, "audio_adaln_single."):
            family = "ltxv"
        else:
            family = "ltx23" if re.search(r"2[._-]?3", os.path.basename(name)) else "ltx2"
    elif _any(keys, "patch_embedding.weight") and _any(keys, "cross_attn."):
        family = _wan(name, tensors)
    elif _any(keys, "txtfusion.projector.weight"):
        # Krea 2: RAW and Turbo share one layout, so only the filename tells them apart.
        family = "krea2_raw" if re.search(r"raw|base", os.path.basename(name).lower()) else "krea2_turbo"
    elif _any(keys, "double_blocks.") and _any(keys, "single_blocks."):
        family, variant = _flux(name, tensors)
    elif _any(keys, "joint_blocks."):
        family = "sd3"
    elif _any(keys, "conditioner.embedders.1") or _any(keys, "label_emb.0.0.weight"):
        family = _sdxl_flavour(name, metadata)
    else:
        dim = _context_dim(tensors, "attn2.to_k.weight")
        family = {768: "sd15", 1024: "sd2", 2048: _sdxl_flavour(name, metadata)}.get(dim)
    return {"family": family, "variant": variant, "has_vae": has_vae, "has_text_encoders": has_te}


# Out-features of a single block's fused linear1: 7 x width for FLUX.1, 9 x width for FLUX.2.
FLUX1_LINEAR1 = {21504}
FLUX2_LINEAR1 = {27648, 36864}


def classify_lora(name, tensors, metadata):
    """Return the base a LoRA was trained for: "sd15", "sdxl", "flux", "flux2", "krea2", "wan", "ltx" or None.

    Pony and Illustrious LoRAs come back as "sdxl" since they load on any SDXL model.
    """
    base = str(metadata.get("ss_base_model_version", "")).lower()
    if "wan" in base:
        return "wan"
    if "ltx" in base:
        return "ltx"
    if "krea2" in base or "krea_2" in base or "krea-2" in base:
        return "krea2"
    if "flux2" in base or "flux.2" in base or "klein" in base:
        return "flux2"
    if "flux" in base:
        return "flux"
    if "sdxl" in base or "pony" in base or "illustrious" in base:
        return "sdxl"
    if base.startswith("sd_v1") or base.startswith("sd_1"):
        return "sd15"

    keys = tensors.keys()
    if any(re.search(r"blocks[._]\d+[._](cross_attn|self_attn)[._]", k) for k in keys):
        return "wan"
    if any(re.search(r"audio_attn|audio_ff|video_to_audio|audio_to_video", k) for k in keys) or "ltx" in os.path.basename(name).lower():
        return "ltx"
    # Krea 2 layer names: native (attn.wq, txtfusion) or diffusers (attn.to_gate, text_fusion).
    if any(re.search(r"txtfusion|text_fusion|attn[._](wq|to_gate)[._]", k) for k in keys):
        return "krea2"
    if _any(keys, "double_blocks") or _any(keys, "single_transformer_blocks") or _any(keys, "single_blocks"):
        for key, shape in tensors.items():
            if re.search(r"single_blocks[._]\d+[._]linear1\.(lora_up|lora_B)\.weight$", key) and shape:
                if shape[0] in FLUX2_LINEAR1:
                    return "flux2"
                if shape[0] in FLUX1_LINEAR1:
                    return "flux"
        return "flux"
    if _any(keys, "lora_te2_"):
        return "sdxl"
    for key, shape in tensors.items():
        # kohya: ..._attn2_to_k.lora_down.weight; diffusers/peft: ...attn2.to_k.lora_A.weight
        if re.search(r"attn2[._]to_k\.lora_(down|A)\.weight$", key) and len(shape) == 2:
            return {768: "sd15", 2048: "sdxl"}.get(shape[1])
    return None


def classify_controlnet(name, tensors, metadata):
    """Return the base a ControlNet was trained for: "sd15", "sdxl", "flux" or None."""
    keys = tensors.keys()
    if _any(keys, "double_blocks") or _any(keys, "single_transformer_blocks"):
        return "flux"
    dim = _context_dim(tensors, "attn2.to_k.weight")
    if dim in (768, 2048):
        return {768: "sd15", 2048: "sdxl"}[dim]
    if _any(keys, "add_embedding") or _any(keys, "label_emb"):
        return "sdxl"
    return None


CLASSIFIERS = {"model": classify_model, "lora": classify_lora, "controlnet": classify_controlnet}

_cache = {}


def cached(kind, path):
    """Classify a file, caching by path and modification time."""
    stamp = os.stat(path).st_mtime_ns
    hit = _cache.get((kind, path))
    if hit and hit[0] == stamp:
        return hit[1]
    tensors, metadata = read_header(path)
    result = CLASSIFIERS[kind](path, tensors, metadata)
    _cache[(kind, path)] = (stamp, result)
    return result
