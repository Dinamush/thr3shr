import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
// Fonts ship with the app, so it looks the same offline and makes no
// third-party requests.
import "@fontsource/figtree/400.css";
import "@fontsource/figtree/500.css";
import "@fontsource/figtree/600.css";
import "@fontsource/figtree/700.css";
import "@fontsource/fragment-mono/400.css";
// Order matters: palette overrides must follow the token defaults.
import "./theme/tokens.css";
import "virtual:thr3shr-palettes.css";
import "./styles.css";
import "./theme/skin-yorha.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
