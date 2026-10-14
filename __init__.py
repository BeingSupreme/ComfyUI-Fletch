"""Fletch: a simple, node-free front end for ComfyUI.

Installing this folder into ComfyUI/custom_nodes serves the app at http://<comfy-host>/fletch/.
It adds no nodes of its own; the app drives ComfyUI through its normal API.
"""

try:
    from server import PromptServer
except ImportError:  # imported outside ComfyUI, e.g. by the test runner
    PromptServer = None

if PromptServer is not None:
    from .backend.routes import register

    register(PromptServer.instance.routes)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
