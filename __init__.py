"""Fletch: a simple, node-free front end for ComfyUI.

Installing this folder into ComfyUI/custom_nodes serves the app at http://<comfy-host>/fletch/.
It adds no nodes of its own; the app drives ComfyUI through its normal API.
"""

import os

from aiohttp import web
from server import PromptServer

DIST = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dist")
INDEX = os.path.join(DIST, "index.html")

routes = PromptServer.instance.routes


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


if os.path.isdir(os.path.join(DIST, "assets")):
    routes.static("/fletch/assets", os.path.join(DIST, "assets"))

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
