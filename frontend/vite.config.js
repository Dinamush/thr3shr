import { defineConfig } from "vite";
import { paletteCss } from "./src/theme/paletteCss.js";

// Serves the palette CSS generated from src/theme/themes.js, so a palette is
// defined once and the build, the Appearance menu and the tests all read it.
const PALETTES_ID = "virtual:thr3shr-palettes.css";
const RESOLVED_PALETTES_ID = `\0${PALETTES_ID}`;

function palettesPlugin() {
  return {
    name: "thr3shr-palettes",
    resolveId(id) {
      return id === PALETTES_ID ? RESOLVED_PALETTES_ID : null;
    },
    load(id) {
      return id === RESOLVED_PALETTES_ID ? paletteCss() : null;
    },
  };
}

export default defineConfig({
  plugins: [palettesPlugin()],
  server: {
    port: 5173,
    strictPort: true,
  },
  preview: {
    port: 4173,
    strictPort: true,
  },
});
