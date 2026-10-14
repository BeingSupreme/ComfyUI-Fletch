# ComfyUI-Fletch

Fletch is a simple front end for [ComfyUI](https://github.com/comfyanonymous/ComfyUI): type a prompt, pick a model, press Generate. There's no node graph. Fletch builds the ComfyUI workflow for you and runs it on your own ComfyUI, so every model and custom node you already have keeps working.

> Status: Phase 2. Model detection, per-family defaults, LoRA, history, and the image add-ons below.

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
- `backend/detect.py` classifies models, LoRAs and ControlNets from their safetensors headers; `backend/routes.py` serves the app and `/fletch/api/*`
- `web/src/profiles.ts` holds each model family's loaders and defaults
- `web/src/workflow.ts` turns settings into ComfyUI API-format workflow JSON
- `web/src/comfy.ts` talks to ComfyUI: `/object_info`, `/prompt`, `/ws`, `/view`
- `web/src/addons.ts` decides which add-ons can run with what is installed
- `web/src/App.tsx`, `AddonCards.tsx` and `MaskPainter.tsx` are the UI
