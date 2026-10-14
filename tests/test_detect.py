import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend import detect  # noqa: E402


def write_safetensors(path, tensors, metadata=None):
    header = {name: {"dtype": "F16", "shape": shape, "data_offsets": [0, 0]} for name, shape in tensors.items()}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)


def model(tmp_path, name, tensors, metadata=None):
    path = tmp_path / name
    write_safetensors(path, tensors, metadata)
    return detect.cached("model", str(path))


def lora(tmp_path, name, tensors, metadata=None):
    path = tmp_path / name
    write_safetensors(path, tensors, metadata)
    return detect.cached("lora", str(path))


SD15 = {
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": [320, 768],
    "cond_stage_model.transformer.text_model.embeddings.position_ids": [1, 77],
    "first_stage_model.decoder.conv_in.weight": [512, 4, 3, 3],
}
SDXL = {
    "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": [640, 2048],
    "conditioner.embedders.1.model.ln_final.weight": [1280],
    "first_stage_model.decoder.conv_in.weight": [512, 4, 3, 3],
}
FLUX_DEV_UNET = {
    "double_blocks.0.img_attn.qkv.weight": [9216, 3072],
    "single_blocks.0.linear1.weight": [21504, 3072],
    "guidance_in.in_layer.weight": [3072, 256],
}


def test_sd15_checkpoint(tmp_path):
    assert model(tmp_path, "dreamshaper_8.safetensors", SD15) == {
        "family": "sd15",
        "variant": None,
        "has_vae": True,
        "has_text_encoders": True,
    }


def test_sd2_is_recognised(tmp_path):
    t = {"model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": [320, 1024]}
    assert model(tmp_path, "v2-1.safetensors", t)["family"] == "sd2"


def test_sdxl_checkpoint(tmp_path):
    assert model(tmp_path, "juggernautXL_v9.safetensors", SDXL)["family"] == "sdxl"


def test_pony_by_filename(tmp_path):
    assert model(tmp_path, "cyberrealisticPony_v150.safetensors", SDXL)["family"] == "pony"


def test_pony_by_metadata(tmp_path):
    meta = {"modelspec.architecture": "stable-diffusion-xl-v1-base", "modelspec.title": "Pony Diffusion V6 XL"}
    assert model(tmp_path, "pdxl.safetensors", SDXL, meta)["family"] == "pony"


def test_illustrious_by_filename(tmp_path):
    assert model(tmp_path, "waiNSFWIllustrious_v140.safetensors", SDXL)["family"] == "illustrious"
    assert model(tmp_path, "noobaiXL_vpred.safetensors", SDXL)["family"] == "illustrious"


def test_flux_dev_diffusion_model_has_no_encoders(tmp_path):
    assert model(tmp_path, "flux1-dev.safetensors", FLUX_DEV_UNET) == {
        "family": "flux_dev",
        "variant": None,
        "has_vae": False,
        "has_text_encoders": False,
    }


def test_flux_schnell(tmp_path):
    t = {k: v for k, v in FLUX_DEV_UNET.items() if not k.startswith("guidance_in")}
    assert model(tmp_path, "flux1-schnell.safetensors", t)["family"] == "flux_schnell"


def test_flux_all_in_one_checkpoint(tmp_path):
    t = {f"model.diffusion_model.{k}": v for k, v in FLUX_DEV_UNET.items()}
    t["vae.decoder.conv_in.weight"] = [512, 16, 3, 3]
    t["text_encoders.clip_l.transformer.text_model.final_layer_norm.weight"] = [768]
    r = model(tmp_path, "flux1-dev-fp8.safetensors", t)
    assert r == {"family": "flux_dev", "variant": None, "has_vae": True, "has_text_encoders": True}


def test_sd3(tmp_path):
    t = {"model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight": [4608, 1536]}
    assert model(tmp_path, "sd3.5_medium.safetensors", t)["family"] == "sd3"


def test_unknown(tmp_path):
    assert model(tmp_path, "mystery.safetensors", {"foo.weight": [3]})["family"] is None


def test_lora_by_metadata(tmp_path):
    assert lora(tmp_path, "a.safetensors", {}, {"ss_base_model_version": "sdxl_base_v1-0"}) == "sdxl"
    assert lora(tmp_path, "b.safetensors", {}, {"ss_base_model_version": "sd_v1"}) == "sd15"
    assert lora(tmp_path, "c.safetensors", {}, {"ss_base_model_version": "flux1"}) == "flux"


def test_lora_by_shape(tmp_path):
    sd15 = {"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn2_to_k.lora_down.weight": [16, 768]}
    sdxl = {"lora_unet_input_blocks_4_1_transformer_blocks_0_attn2_to_k.lora_down.weight": [16, 2048]}
    peft = {"unet.down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.weight": [8, 2048]}
    assert lora(tmp_path, "d.safetensors", sd15) == "sd15"
    assert lora(tmp_path, "e.safetensors", sdxl) == "sdxl"
    assert lora(tmp_path, "f.safetensors", peft) == "sdxl"


def test_lora_flux_and_sdxl_text_encoder_keys(tmp_path):
    assert lora(tmp_path, "g.safetensors", {"lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": [16, 3072]}) == "flux"
    assert lora(tmp_path, "h.safetensors", {"lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": [8, 1280]}) == "sdxl"


KLEIN_4B = {
    "double_blocks.0.img_attn.qkv.weight": [9216, 3072],
    "single_blocks.0.linear1.weight": [27648, 3072],
    "double_stream_modulation_img.lin.weight": [18432, 3072],
    "txt_in.weight": [3072, 7680],
}


def test_flux2_klein_4b_distilled(tmp_path):
    assert model(tmp_path, "flux-2-klein-4b-fp8.safetensors", KLEIN_4B) == {
        "family": "flux2_klein",
        "variant": "4b",
        "has_vae": False,
        "has_text_encoders": False,
    }


def test_flux2_klein_9b_base_by_text_width(tmp_path):
    t = {
        "double_blocks.0.img_attn.qkv.weight": [12288, 4096],
        "single_blocks.0.linear1.weight": [36864, 4096],
        "txt_in.weight": [4096, 12288],
    }
    r = model(tmp_path, "flux-2-klein-base-9b-fp8.safetensors", t)
    assert (r["family"], r["variant"]) == ("flux2_klein_base", "9b")


def test_flux1_is_not_mistaken_for_flux2(tmp_path):
    t = dict(FLUX_DEV_UNET, **{"txt_in.weight": [3072, 4096]})
    assert model(tmp_path, "flux1-dev-2.safetensors", t)["family"] == "flux_dev"


def test_flux2_dev_is_recognised_but_unsupported(tmp_path):
    t = {"double_blocks.0.a": [1], "single_blocks.0.a": [1], "txt_in.weight": [6144, 15360]}
    assert model(tmp_path, "flux2-dev.safetensors", t)["family"] == "flux2"


def test_lora_flux1_vs_flux2_by_linear1(tmp_path):
    f1 = {"lora_unet_single_blocks_0_linear1.lora_up.weight": [21504, 16]}
    f2 = {"single_blocks.0.linear1.lora_B.weight": [27648, 16]}
    assert lora(tmp_path, "i.safetensors", f1) == "flux"
    assert lora(tmp_path, "j.safetensors", f2) == "flux2"
    assert lora(tmp_path, "k.safetensors", {}, {"ss_base_model_version": "flux2_klein_9b"}) == "flux2"


def controlnet(tmp_path, name, tensors, metadata=None):
    path = tmp_path / name
    write_safetensors(path, tensors, metadata)
    return detect.cached("controlnet", str(path))


def test_controlnet_families(tmp_path):
    sd15 = {"control_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight": [320, 768]}
    sdxl = {"down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight": [640, 2048]}
    flux = {"double_blocks.0.img_attn.qkv.weight": [9216, 3072], "controlnet_blocks.0.weight": [3072, 3072]}
    assert controlnet(tmp_path, "cn15.safetensors", sd15) == "sd15"
    assert controlnet(tmp_path, "cnxl.safetensors", sdxl) == "sdxl"
    assert controlnet(tmp_path, "cnflux.safetensors", flux) == "flux"
    assert controlnet(tmp_path, "cn_unknown.safetensors", {"x": [1]}) is None
