"""HTTP routes Fletch adds to ComfyUI's server: the app, model detection and the output folder."""

import asyncio
import os

import folder_paths
from aiohttp import web

from . import detect, output

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

    @routes.get("/fletch/api/output")
    async def get_output(request):
        return web.json_response(output.current())

    @routes.post("/fletch/api/output")
    async def set_output(request):
        folder = str((await request.json()).get("folder", ""))
        try:
            return web.json_response(await asyncio.to_thread(output.set_folder, folder))
        except OSError as e:
            return web.json_response({"error": f"Can't save to that folder: {e.strerror or e}"}, status=400)

    @routes.post("/fletch/api/output/pick")
    async def pick_output(request):
        try:
            chosen = await asyncio.to_thread(output.pick_folder, output.current()["path"])
        except Exception:  # no tkinter (e.g. embedded Python) or no desktop session
            return web.json_response({"error": "unavailable"}, status=501)
        return web.json_response({"folder": chosen})

    @routes.post("/fletch/api/output/open")
    async def open_output(request):
        path = output.current()["path"]
        try:
            os.makedirs(path, exist_ok=True)
            output.open_folder(path)
        except OSError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True})

    @routes.get("/fletch/api/image")
    async def image(request):
        folder, name = request.query.get("folder", ""), request.query.get("filename", "")
        path = os.path.join(folder, name)
        if not name or os.path.basename(name) != name or not output.is_known(folder) or not os.path.isfile(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    if os.path.isdir(os.path.join(DIST, "assets")):
        routes.static("/fletch/assets", os.path.join(DIST, "assets"))
