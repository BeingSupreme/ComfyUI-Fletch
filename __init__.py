"""Fletch: a simple, node-free front end for ComfyUI.

Installing this folder into ComfyUI/custom_nodes serves the app at http://<comfy-host>/fletch/.
The app drives ComfyUI through its normal API. Its one node saves images outside ComfyUI's
output folder when you pick a folder elsewhere.
"""

try:
    from server import PromptServer
except ImportError:  # imported outside ComfyUI, e.g. by the test runner
    PromptServer = None

if PromptServer is not None:
    from .backend.routes import register

    register(PromptServer.instance.routes)
    from .backend.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
