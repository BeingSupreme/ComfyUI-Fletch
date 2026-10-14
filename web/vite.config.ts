import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, point COMFY_URL at a running ComfyUI (default http://127.0.0.1:8188).
const comfy = process.env.COMFY_URL ?? "http://127.0.0.1:8188";

export default defineConfig({
  base: "/fletch/",
  plugins: [react()],
  build: { outDir: "../dist", emptyOutDir: true },
  server: {
    proxy: {
      "/object_info": comfy,
      "/prompt": comfy,
      "/view": comfy,
      "/ws": { target: comfy, ws: true },
    },
  },
});
