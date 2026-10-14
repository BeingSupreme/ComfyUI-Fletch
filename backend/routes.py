"""HTTP routes Fletch adds to ComfyUI's server: the app itself and model detection."""

import asyncio
import os

import folder_paths
from aiohttp import web

from . import detect

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DIST = os.path.join(ROOT, "dist")
INDEX = os.path.join(DIST, "index.html")

# Folders the app may ask about, as ComfyUI names them.
MODEL_FOLDERS = ("checkpoints", "diffusion_models")


def _classify(kind, folder, name):
    path = folder_paths.get_full_path(folder, name)
    if not path:
        return None
    if not path.endswith(".safetensors"):
        return {"family": None, "reason": "not a .safetensors file"} if kind == "model" else None
    try:
        return detect.cached(kind, path)
    except (OSError, ValueError) as e:
        return {"family": None, "reason": str(e)} if kind == "model" else None


def register(routes):
    @routes.get("/fletch")
    async def fletch_redirect(request):
        raise web.HTTPFound("/fletch/")

    @routes.get("/fletch/")
    async def fletch_index(request):
        if not os.path.isfile(INDEX):
            return web.Response(
                status=503,
                text="Fletch's web app isn't built. Run `npm install && npm run build` in custom_nodes/ComfyUI-Fletch/web.",
            )
        return web.FileResponse(INDEX, headers={"Cache-Control": "no-cache"})

    @routes.get("/fletch/api/model_info")
    async def model_info(request):
        folder = request.query.get("folder", "checkpoints")
        name = request.query.get("name", "")
        if folder not in MODEL_FOLDERS or not name:
            return web.json_response({"error": "folder and name are required"}, status=400)
        result = await asyncio.to_thread(_classify, "model", folder, name)
        if result is None:
            return web.json_response({"error": f"{name} not found in {folder}"}, status=404)
        return web.json_response(result)

    def file_families(kind, folder):
        async def handler(request):
            def scan():
                return [{"name": n, "family": _classify(kind, folder, n)} for n in folder_paths.get_filename_list(folder)]

            return web.json_response(await asyncio.to_thread(scan))

        return handler

    routes.get("/fletch/api/loras")(file_families("lora", "loras"))
    routes.get("/fletch/api/controlnets")(file_families("controlnet", "controlnet"))

    if os.path.isdir(os.path.join(DIST, "assets")):
        routes.static("/fletch/assets", os.path.join(DIST, "assets"))
