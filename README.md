# ComfyUI-Fletch

Fletch is a simple front end for [ComfyUI](https://github.com/comfyanonymous/ComfyUI): type a prompt, pick a model, press Generate. There's no node graph. Fletch builds the ComfyUI workflow for you and runs it on your own ComfyUI, so every model and custom node you already have keeps working.

> Status: Phase 1. Model detection, per-family defaults, LoRA and history.

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

- **Pick a model, Fletch sets it up.** It reads the model file's header (not the weights) to tell SD 1.5, SDXL, Pony, Illustrious, Flux.1 and Flux.2 Klein apart, then fills in size, steps, CFG, sampler, clip skip and any quality tags that family expects. The family badge under the model shows what was detected; change it if detection gets it wrong.
- **Flux without wiring.** Pick a Flux model from `models/diffusion_models` and Fletch finds its text encoders and VAE for you: T5 + CLIP-L + `ae` for Flux.1, the matching Qwen3 encoder (4B or 8B) + `flux2-vae` for Flux.2 Klein. If one is missing, it says which file to add. Flux files without a text encoder built in belong in `models/diffusion_models`, not `models/checkpoints`; Fletch tells you if one is in the wrong place.
- **Shape and Quality** instead of width, height and steps. Each family maps them to sizes and step counts that suit it.
- **LoRA** add-on. The list only shows LoRAs made for the current model's family (plus any it can't identify). Add as many as you like, each with its own weight.
- **Advanced** holds steps, CFG, sampler, scheduler, clip skip and seed, pre-filled from the family's defaults.
- **History** keeps your last 40 images. Click one, then "Use these settings" to get the exact same setup back, seed included. Every saved PNG also carries its Fletch settings in its metadata.

SD 2.x, SD 3.5 and FLUX.2 dev models are recognised but not supported yet.

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
- `backend/detect.py` classifies models and LoRAs from their safetensors headers; `backend/routes.py` serves the app and `/fletch/api/*`
- `web/src/profiles.ts` holds each model family's loaders and defaults
- `web/src/workflow.ts` turns settings into ComfyUI API-format workflow JSON
- `web/src/comfy.ts` talks to ComfyUI: `/object_info`, `/prompt`, `/ws`, `/view`
- `web/src/App.tsx` is the UI
