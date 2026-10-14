# ComfyUI-Fletch

Fletch is a simple front end for [ComfyUI](https://github.com/comfyanonymous/ComfyUI): type a prompt, pick a model, press Generate. There's no node graph. Fletch builds the ComfyUI workflow for you and runs it on your own ComfyUI, so every model and custom node you already have keeps working.

> Status: Phase 0. One fixed SDXL text-to-image workflow, to prove the loop end to end.

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

- Lists the checkpoints in your ComfyUI `models/checkpoints` folder
- Prompt and "avoid" (negative prompt) boxes
- Generates with SDXL defaults: 1024 x 1024, 30 steps, CFG 6, dpmpp_2m / karras, random seed
- Live step progress, live previews (if ComfyUI is started with `--preview-method auto`), and a history strip for this session

Use an SDXL checkpoint for now. Model detection and per-family defaults come in Phase 1.

## Develop

```
cd web
npm install
npm run dev     # http://localhost:5173/fletch/, proxies to ComfyUI at COMFY_URL (default http://127.0.0.1:8188)
npm test        # workflow compiler tests
npm run build   # writes ../dist, which ComfyUI serves
```

## Layout

- `__init__.py` registers the `/fletch/` route in ComfyUI and serves `dist/`
- `web/src/workflow.ts` turns settings into ComfyUI API-format workflow JSON
- `web/src/comfy.ts` talks to ComfyUI: `/object_info`, `/prompt`, `/ws`, `/view`
- `web/src/App.tsx` is the UI
