# ComfyUI-Fletch

Fletch is a simple front end for [ComfyUI](https://github.com/comfyanonymous/ComfyUI): type a prompt, pick a model, press Generate. There's no node graph. Fletch builds the ComfyUI workflow for you and runs it on your own ComfyUI, so every model and custom node you already have keeps working.

> Status: Phase 4 in progress. Images (Phases 1 and 2) plus Wan 2.2 video.

## Install

1. Clone into ComfyUI's `custom_nodes` folder:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/BeingSupreme/ComfyUI-Fletch
   ```
2. Restart ComfyUI.
3. Open http://127.0.0.1:8188/fletch/ (use your ComfyUI address if it runs elsewhere).

The built app ships in `dist/`, so there's nothing to build to use it.

## What works now

- **Pick a model, Fletch sets it up.** It reads the model file's header (not the weights) to tell SD 1.5, SDXL, Pony, Illustrious, Flux.1, Flux.2 Klein and Krea 2 apart, then fills in size, steps, CFG, sampler, clip skip and any quality tags that family expects. The family badge under the model shows what was detected; change it if detection gets it wrong.
- **Flux without wiring.** Pick a Flux model from `models/diffusion_models` and Fletch finds its text encoders and VAE for you: T5 + CLIP-L + `ae` for Flux.1, the matching Qwen3 encoder (4B or 8B) + `flux2-vae` for Flux.2 Klein. If one is missing, it says which file to add. Flux files without a text encoder built in belong in `models/diffusion_models`, not `models/checkpoints`; Fletch tells you if one is in the wrong place.
- **Krea 2 Turbo.** Put `krea2_turbo_fp8_scaled.safetensors` in `models/diffusion_models`, `qwen3vl_4b_fp8_scaled.safetensors` in `models/text_encoders` and `qwen_image_vae.safetensors` in `models/vae` (all from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2)). Fletch runs it at 8 steps and CFG 1 like ComfyUI's template. Needs ComfyUI 0.25 or newer. Krea 2 style LoRAs work too.
- **Shape and Quality** instead of width, height and steps. Each family maps them to sizes and step counts that suit it.
- **LoRA** add-on. The list only shows LoRAs made for the current model's family (plus any it can't identify). Add as many as you like, each with its own weight.
- **Advanced** holds steps, CFG, sampler, scheduler, clip skip and seed, pre-filled from the family's defaults.
- **History** keeps your last 40 images. Click one, then "Use these settings" to get the exact same setup back, seed included. Every saved PNG also carries its Fletch settings in its metadata.
- **Save to** (above the history) picks where images go: ComfyUI's output folder by default, a named folder inside it, or any folder on your PC (Browse opens your system's folder picker when ComfyUI's Python has it; otherwise paste the path). **Open folder** shows it in your file manager, and **Clear history** empties the list without deleting any images.

### Video (Wan 2.2 and LTX-2.3)

Pick a Wan 2.2 model and Fletch switches to video: a **Length** choice (2, 3 or 5 seconds) appears and the result plays in the preview and history.

- **Wan 2.2 14B** comes as two files, a high-noise and a low-noise model (for example `wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors` and `..._low_noise_...`). Put both in `models/diffusion_models` and pick either one; Fletch finds its partner and splits the steps between them like ComfyUI's template. The text-to-video pair makes video from your prompt; the image-to-video pair animates a picture from **Start from an image**.
- **Fast** uses the lightx2v 4-step LoRAs (`wan2.2_t2v_lightx2v_4steps_lora_..._high_noise` / `_low_noise`, or the `i2v` ones) if they're in `models/loras`: 4 steps at CFG 1 instead of 14. Balanced and Best run 20 and 30 steps at CFG 3.5.
- **Wan 2.2 5B** (`wan2.2_ti2v_5B_fp16.safetensors`) is one file at 24 fps and 720p, with an optional start picture.
- Both need `umt5_xxl_fp8_e4m3fn_scaled.safetensors` in `models/text_encoders`; 14B uses `wan_2.1_vae.safetensors` and 5B uses `wan2.2_vae.safetensors` in `models/vae`.
- **LTX-2.3** makes video with sound, at 25 fps. It loads as a checkpoint and needs four files, the same ones ComfyUI's LTX-2.3 template uses:
  - `ltx-2.3-22b-dev-fp8.safetensors` in `models/checkpoints` ([Lightricks/LTX-2.3-fp8](https://huggingface.co/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors))
  - `gemma_3_12B_it_fp4_mixed.safetensors` in `models/text_encoders` ([Comfy-Org/ltx-2](https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors))
  - `ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors` in `models/loras` ([Comfy-Org/ltx-2.3](https://huggingface.co/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/ltx_2.3_22b_distilled_1.1_lora_dynamic_fro09_avg_rank_111_bf16.safetensors))
  - `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` in `models/latent_upscale_models` ([Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors))

  It runs the template's fixed schedule (8 steps at half size, then 3 more after a 2x upscale), so the Quality choice is hidden. A start picture from **Start from an image** becomes the first frame. LTX offers up to 20 seconds; longer clips need more memory and time.
- **Continue from last frame** (under a finished video) makes its final frame the start picture for the next run, so you can chain clips. Each clip saves as its own file.
- **GIF to GIF**: turn on **Start from an image** and drop in an animated GIF. Fletch restyles its frames with your prompt and saves a GIF at the same speed. **Change** sets how far it moves from the original. It works with the Wan 2.2 text-to-video pair, Wan 2.2 5B and LTX-2.3. Fletch uses the GIF's frames up to the chosen length, rounded down to a count the model takes, so a few frames at the end may be dropped.
  - **Make** picks what comes out: **GIF**, **Video** (with sound on LTX-2.3), or **Add sound only** (LTX-2.3), which keeps the GIF's picture exactly and makes a soundtrack from your prompt.
- Wan LoRAs work. One whose name says high noise or low noise goes on that half only; others go on both.
- The image add-ons (hires fix, face fix, guide, upscale) are off for video for now.

### Add-ons

Each add-on is a switch in the left column. Fletch wires the nodes in for you. If one can't run with your setup, its switch is greyed out and says what's missing.

- **Start from an image** (img2img). Drop a picture in and set how much should change. The output follows the picture's shape. **Paint an area to change** turns it into inpainting: only the painted part is redrawn. Under any finished image, "Start from this image" sends it back in.
- **Hires fix** renders, enlarges 1.5x or 2x, then refines the details in a second pass.
- **Face fix** finds faces and redraws them at higher detail. Needs [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack), [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack) and a face model such as `face_yolov8m.pt` in `models/ultralytics/bbox`.
- **Pose or edge guide** (ControlNet) makes the image follow another picture's edges, pose or depth. Put ControlNet models in `models/controlnet`; Fletch only lists ones that match your model's family. Edges work out of the box; pose and depth need [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux). "Use as-is" takes a picture that's already a pose or edge map.
- **Upscale** enlarges the final image with an upscale model from `models/upscale_models` (for example 4x-UltraSharp).

Face fix and the guide aren't available for Flux.2 Klein yet, and the guide isn't available for Krea 2.

SD 2.x, SD 3.5, FLUX.2 dev and Krea 2 RAW models are recognised but not supported yet.

## Develop

```
cd web
npm install
npm run dev     # http://localhost:5173/fletch/, proxies to ComfyUI at COMFY_URL (default http://127.0.0.1:8188)
npm test        # workflow compiler tests
cd .. && python -m pytest   # detection tests
npm run build   # writes ../dist, which ComfyUI serves
```

## Layout

- `__init__.py` registers the `/fletch/` route in ComfyUI and serves `dist/`
- `backend/output.py` and `backend/nodes.py` handle the output folder; the Fletch Save Image and Save Video nodes save to folders outside ComfyUI's output
- `backend/detect.py` classifies models, LoRAs and ControlNets from their safetensors headers; `backend/routes.py` serves the app and `/fletch/api/*`
- `web/src/profiles.ts` holds each model family's loaders and defaults
- `web/src/workflow.ts` turns settings into ComfyUI API-format workflow JSON; `web/src/video.ts` does the same for video
- `web/src/comfy.ts` talks to ComfyUI: `/object_info`, `/prompt`, `/ws`, `/view`
- `web/src/addons.ts` decides which add-ons can run with what is installed
- `web/src/App.tsx`, `AddonCards.tsx` and `MaskPainter.tsx` are the UI
